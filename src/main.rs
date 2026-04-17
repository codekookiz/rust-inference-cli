mod layers;

use layers::{Layer, Dense, ReLU, Sigmoid};
use serde_json::Value;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::path::Path;

fn load_model() -> Vec<Box<dyn Layer>> {
    let trained_path = "trained_model.json";
    let init_path = "init_model.json";

    let path = if Path::new(trained_path).exists() {
        println!("기존 학습된 모델('{}')을 불러옵니다.", trained_path);
        trained_path
    } else {
        println!("초기 모델('{}')을 불러옵니다.", init_path);
        init_path
    };

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

fn main() {
    let model_list = load_model();
    let model = Arc::new(model_list);
    println!("학습 엔진 로드 완료 (레이어 {}개)", model.len());

    let (in_tx, in_rx) = mpsc::channel::<(f32, f32)>();
    let (out_tx, out_rx) = mpsc::channel::<(usize, f32, f32)>();
    let shared_in_rx = Arc::new(Mutex::new(in_rx));

    let learning_rate = 0.1;

    let mut handles = vec![];

    for worker_id in 0..3 {
        let model_ref = Arc::clone(&model);
        let rx_ref = Arc::clone(&shared_in_rx);
        let tx_res = out_tx.clone();

        let handle = thread::spawn(move || {
            loop {
                let (input_val, target_val) = {
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
                let mut gradient = final_output - target_val;
                let loss = gradient.powi(2) * 0.5;

                for (layer, &prev_input) in model_ref.iter().rev().zip(layer_inputs.iter().rev()) {
                    gradient = layer.backward(prev_input, gradient, learning_rate);
                }

                tx_res.send((worker_id, final_output, loss)).unwrap();
            }
        });
        handles.push(handle);
    }

    thread::spawn(move || {
        for (id, res, loss) in out_rx {
            println!("\n[Worker {}] 결과: {:.4} | Loss: {:.4}", id, res, loss);
            print!("입력(값:정답, 종료:q, 저장:s) > ");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    });

    println!("학습 데이터를 입력하세요. 예: 1.0:2.0 (입력 1.0, 정답 2.0)");
    println!("여러 개 입력 시 쉼표 사용 (저장: s, 종료: q)");

    loop {
        print!("입력 > ");
        use std::io::{stdin, stdout, Write};
        stdout().flush().unwrap();

        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read");
        let input = input.trim();

        if input == "q" {
            break;
        }
        if input == "s" {
            save_model("trained_model.json", &model);
            continue;
        }

        for part in input.split(',') {
            let pair: Vec<&str> = part.split(':').collect();
            if pair.len() == 2 {
                let val = pair[0].trim().parse::<f32>();
                let tar = pair[1].trim().parse::<f32>();
                if let (Ok(v), Ok(t)) = (val, tar) {
                    in_tx.send((v, t)).unwrap();
                }
            } else {
                println!("형식이 틀렸습니다. (값:정답)");
            }
        }
    }

    drop(in_tx);
    
    save_model("trained_model.json", &model);
    println!("시스템을 안전하게 종료합니다...");
}