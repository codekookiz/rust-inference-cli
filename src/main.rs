mod layers;

use layers::{Layer, Dense, ReLU, Sigmoid};
use serde_json::Value;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::path::Path;

fn load_model_from_path(path: &str) -> Vec<Box<dyn Layer>> {
    let data = std::fs::read_to_string(path).expect("모델 파일을 읽을 수 없습니다.");
    let json: Value = serde_json::from_str(&data).expect("JSON 파싱에 실패했습니다.");
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();

    if let Some(arr) = json.as_array() {
        for item in arr {
            let layer_type = item["type"].as_str().expect("type 필드가 없습니다.");
            match layer_type {
                "Dense" => {
                    let d: Dense = serde_json::from_value(item.clone()).unwrap();
                    layers.push(Box::new(d));
                }
                "ReLU" => {
                    let r: ReLU = serde_json::from_value(item.clone()).unwrap();
                    layers.push(Box::new(r));
                }
                "Sigmoid" => {
                    let s: Sigmoid = serde_json::from_value(item.clone()).unwrap();
                    layers.push(Box::new(s));
                }
                _ => println!("경고: 알 수 없는 레이어 타입 '{}'", layer_type),
            }
        }
    }
    layers
}

fn save_model(path: &str, model: &[Box<dyn Layer>]) {
    let json_list: Vec<Value> = model.iter().map(|l| l.to_json()).collect();
    let json_string = serde_json::to_string_pretty(&json_list).expect("직렬화 실패");
    std::fs::write(path, json_string).expect("파일 쓰기 실패");
    println!("\n모델이 '{}'에 저장되었습니다!", path);
}

fn interpret_result(model_path: &str, input: f32, output: f32, loss: f32, is_prediction: bool) {
    let filename = Path::new(model_path).file_name().unwrap().to_str().unwrap();
    println!("\n--- 분석 결과 ({}) ---", filename);

    if filename.contains("exam") {
        let chance = (output * 100.0) as i32;
        println!("공부 시간: {:.1}", input);
        println!("합격 확률: {}%", chance);
        if chance > 70 { println!("상태: 합격권"); }
        else { println!("상태: 노력 필요"); }
    } else if filename.contains("sentiment") {
        println!("문장 점수: {:.2}", input);
        if output > 0.65 { println!("감정: 긍정"); }
        else if output < 0.35 { println!("감정: 부정"); }
        else { println!("감정: 중립"); }
    }
    
    if !is_prediction {
        println!("Loss: {:.6}", loss);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_init_path = "init_model.json";

    let (model_path, save_path) = if let Some(p) = args.get(1) {
        let path_str = p.as_str();
        (path_str.to_string(), format!("trained_{}", path_str))
    } else {
        let trained_default = "trained_model.json";
        if Path::new(trained_default).exists() {
            (trained_default.to_string(), trained_default.to_string())
        } else {
            (default_init_path.to_string(), "trained_model.json".to_string())
        }
    };

    let model_list = load_model_from_path(&model_path);
    let model = Arc::new(model_list);
    
    println!("모델 로드 완료: {}", model_path);
    println!("저장 예정 경로: {}", save_path);

    let (in_tx, in_rx) = mpsc::channel::<(f32, f32, bool)>();
    let (out_tx, out_rx) = mpsc::channel::<(usize, f32, f32, f32, bool)>();
    let shared_in_rx = Arc::new(Mutex::new(in_rx));
    let learning_rate = 0.1;

    for worker_id in 0..3 {
        let model_ref = Arc::clone(&model);
        let rx_ref = Arc::clone(&shared_in_rx);
        let tx_res = out_tx.clone();

        thread::spawn(move || {
            loop {
                let (input_val, target_val, is_pred) = {
                    let lock = rx_ref.lock().unwrap();
                    match lock.recv() {
                        Ok(v) => v,
                        Err(_) => break,
                    }
                };

                let mut current_input = input_val;
                let mut layer_inputs = vec![];

                for layer in model_ref.iter() {
                    layer_inputs.push(current_input);
                    current_input = layer.forward(current_input);
                }

                let final_output = current_input;
                let gradient = final_output - target_val;
                let loss = gradient.powi(2) * 0.5;

                if !is_pred {
                    let mut grad = gradient;
                    for (layer, &prev_input) in model_ref.iter().rev().zip(layer_inputs.iter().rev()) {
                        grad = layer.backward(prev_input, grad, learning_rate);
                    }
                }

                tx_res.send((worker_id, input_val, final_output, loss, is_pred)).unwrap();
            }
        });
    }

    let model_path_for_thread = model_path.clone();
    thread::spawn(move || {
        for (_id, input, res, loss, is_pred) in out_rx {
            interpret_result(&model_path_for_thread, input, res, loss, is_pred);
            print!("\n입력(값:정답, 값, 종료:q, 저장:s) > ");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    });

    if let Some(data_path) = args.get(2) {
        if let Ok(content) = std::fs::read_to_string(data_path) {
            for part in content.split(',') {
                let pair: Vec<&str> = part.split(':').collect();
                if pair.len() == 2 {
                    if let (Ok(v), Ok(t)) = (pair[0].trim().parse::<f32>(), pair[1].trim().parse::<f32>()) {
                        in_tx.send((v, t, false)).unwrap();
                    }
                }
            }
        }
    }

    loop {
        print!("입력 > ");
        use std::io::{stdin, stdout, Write};
        stdout().flush().unwrap();

        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read");
        let input = input.trim();

        if input == "q" { break; }
        if input == "s" {
            save_model(&save_path, &model);
            continue;
        }

        for part in input.split(',') {
            let pair: Vec<&str> = part.split(':').collect();
            if pair.len() == 2 {
                if let (Ok(v), Ok(t)) = (pair[0].trim().parse::<f32>(), pair[1].trim().parse::<f32>()) {
                    in_tx.send((v, t, false)).unwrap();
                }
            } else if pair.len() == 1 {
                if let Ok(v) = pair[0].trim().parse::<f32>() {
                    in_tx.send((v, 0.0, true)).unwrap();
                }
            }
        }
    }

    drop(in_tx);
    save_model(&save_path, &model);
}