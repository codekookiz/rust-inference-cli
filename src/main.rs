mod layers;

use layers::{Layer, Dense, ReLU, Sigmoid};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::path::Path;
use std::io::{Write, stdin, stdout};

fn load_sentiment_map(path: &str) -> HashMap<String, f32> {
    if let Ok(data) = std::fs::read_to_string(path) {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        HashMap::new()
    }
}

fn normalize_input(v: Vec<f32>) -> Vec<f32> {
    if v.len() == 3 {
        let mut normalized = v;
        normalized[0] = ((normalized[0] + 1.0).ln() / 5.0).clamp(0.0, 1.0);

        normalized[1] = (normalized[1] / 100.0).clamp(0.0, 1.0);
        normalized[2] = (normalized[2] / 20.0).clamp(0.0, 1.0);
        normalized
    } else {
        v
    }
}

fn stem_korean(word: &str) -> String {
    let mut stemmed = word.to_string();
    
    let endings = ["은", "는", "이", "가", "을", "를", "으로", "로", "에서", "보다", "부터", "까지"];
    for &end in &endings {
        if stemmed.ends_with(end) && stemmed.chars().count() > end.chars().count() {
            stemmed = stemmed[..stemmed.len() - end.len()].to_string();
            break; 
        }
    }

    let verb_endings = ["다", "요", "네", "어", "아", "니", "게", "해"];
    for &end in &verb_endings {
        if stemmed.ends_with(end) && stemmed.chars().count() > 1 {
            stemmed = stemmed[..stemmed.len() - end.len()].to_string();
            break;
        }
    }

    stemmed
}

fn text_to_vector(text: &str, map: &HashMap<String, f32>) -> Vec<f32> {
    let mut total_score = 0.0;
    let mut last_score = 0.0;
    let multiplier = 1.5;

    for word in text.split_whitespace() {
        let stemmed = stem_korean(word);
        
        let score = map.get(word)
            .or_else(|| map.get(&stemmed));

        if let Some(&current_score) = score {
            if last_score != 0.0 && current_score != 0.0 {
                if (last_score > 0.0 && current_score > 0.0) || (last_score < 0.0 && current_score < 0.0) {
                    total_score += current_score * multiplier;
                } else {
                    total_score += current_score;
                }
            } else {
                total_score += current_score;
            }
            last_score = current_score;
        } else {
            last_score = 0.0;
        }
    }
    vec![total_score]
}

fn load_model_from_path(path: &str) -> Vec<Box<dyn Layer>> {
    let data = std::fs::read_to_string(path).expect("File error");
    let json: Value = serde_json::from_str(&data).expect("JSON error");
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();

    if let Some(arr) = json.as_array() {
        for item in arr {
            let layer_type = item["type"].as_str().expect("No type");
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
                _ => {}
            }
        }
    }
    layers
}

fn save_model(path: &str, model: &[Box<dyn Layer>]) {
    let json_list: Vec<Value> = model.iter().map(|l| l.to_json()).collect();
    let json_string = serde_json::to_string_pretty(&json_list).unwrap();
    std::fs::write(path, json_string).unwrap();
    println!("\n[시스템] 모델 저장 완료: {}", path);
}

fn interpret_result(model_path: &str, input: &[f32], output: f32, loss: f32, is_prediction: bool) {
    let filename = Path::new(model_path).file_name().unwrap().to_str().unwrap();
    println!("\n--- 분석 결과 ({}) ---", filename);

    if filename.contains("exam") {
        println!("입력 데이터 (공부, 성적, 컨디션): {:?}", input);
        println!("합격 확률: {:.1}%", output * 100.0);
    } else if filename.contains("sentiment") {
        println!("입력 벡터: {:?}", input);
        print!("분석 결과: ");
        if output > 0.65 { println!("긍정 (확률: {:.1}%)", output * 100.0); }
        else if output < 0.35 { println!("부정 (확률: {:.1}%)", (1.0 - output) * 100.0); }
        else { println!("중립 ({:.1}%)", output * 100.0); }
    }
    
    if !is_prediction {
        println!("Loss: {:.6}", loss);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let (model_path, save_path, is_predict_only) = if let Some(p) = args.get(1) {
        let is_only = args.iter().any(|arg| arg == "--predict");
        (p.clone(), format!("trained_{}", p), is_only)
    } else {
        ("exam_init.json".to_string(), "trained_exam.json".to_string(), false)
    };

    let model = Arc::new(load_model_from_path(&model_path));
    let sentiment_map = load_sentiment_map("sentiment_map.json");
    
    let (in_tx, in_rx) = mpsc::channel::<(Vec<f32>, f32, bool)>();
    let (out_tx, out_rx) = mpsc::channel::<(Vec<f32>, f32, f32, bool)>();
    let shared_in_rx = Arc::new(Mutex::new(in_rx));
    let learning_rate = 0.01;

    let input_hint = if model_path.contains("exam") {
        "공부(0+),성적(0~100),컨디션(1~10)"
    } else if model_path.contains("sentiment") {
        "감정 문장 또는 수치"
    } else {
        "v1,v2,v3..."
    };

    let mut prepped_data = Vec::new();
    if !is_predict_only {
        if let Some(data_path) = args.get(2) {
            if let Ok(content) = std::fs::read_to_string(data_path) {
                let parts: Vec<&str> = content.split(';').filter(|s| !s.trim().is_empty()).collect();
                for part in parts {
                    let pair: Vec<&str> = part.split(':').collect();
                    if pair.len() == 2 {
                        let mut v = if model_path.contains("sentiment") && !pair[0].contains(',') && pair[0].trim().parse::<f32>().is_err() {
                            text_to_vector(pair[0].trim(), &sentiment_map)
                        } else {
                            pair[0].split(',').map(|s| s.trim().parse::<f32>().unwrap_or(0.0)).collect()
                        };
                        
                        if model_path.contains("exam") { v = normalize_input(v); }
                        
                        let t = pair[1].trim().parse::<f32>().unwrap_or(0.0);
                        prepped_data.push((v, t));
                    }
                }
            }
        }
    }
    let total_expected = prepped_data.len();

    for _ in 0..3 {
        let model_ref = Arc::clone(&model);
        let rx_ref = Arc::clone(&shared_in_rx);
        let tx_res = out_tx.clone();

        thread::spawn(move || {
            loop {
                let (input_vec, target_val, is_pred) = {
                    let lock = rx_ref.lock().unwrap();
                    match lock.recv() {
                        Ok(v) => v,
                        Err(_) => break,
                    }
                };

                let mut current_input = input_vec.clone();
                let mut layer_inputs = vec![];

                for layer in model_ref.iter() {
                    layer_inputs.push(current_input.clone());
                    current_input = layer.forward(current_input);
                }

                let final_output = current_input[0];
                let gradient = final_output - target_val;
                let loss = gradient.powi(2) * 0.5;

                if !is_pred {
                    let mut grad = vec![gradient];
                    for (layer, prev_input) in model_ref.iter().rev().zip(layer_inputs.into_iter().rev()) {
                        grad = layer.backward(prev_input, grad, learning_rate);
                    }
                }
                tx_res.send((input_vec, final_output, loss, is_pred)).unwrap();
            }
        });
    }

    let model_path_for_thread = model_path.clone();
    let hint_for_thread = input_hint.to_string();
    thread::spawn(move || {
        let mut count = 0;
        for (input_vec, res, loss, is_pred) in out_rx {
            count += 1;
            if is_pred || count % 10 == 0 || (total_expected > 0 && count == total_expected) {
                interpret_result(&model_path_for_thread, &input_vec, res, loss, is_pred);
                if !is_pred { println!("[진행] {}/{}개 처리 중...", count, total_expected); }
                print!("\n입력({}) > ", hint_for_thread);
                stdout().flush().unwrap();
            }
        }
    });

    if !is_predict_only {
        for _ in 0..1000 {
        for (v, t) in &prepped_data {
            in_tx.send((v.clone(), *t, false)).unwrap();
        }
    }
    } else {
        println!("[모드] 예측 전용 모드 활성화.");
        print!("입력({}) > ", input_hint);
        stdout().flush().unwrap();
    }

    loop {
        let mut input_str = String::new();
        stdin().read_line(&mut input_str).unwrap();
        let trimmed = input_str.trim();

        if trimmed == "q" { break; }
        if trimmed == "s" { save_model(&save_path, &model); continue; }

        if model_path.contains("sentiment") && !trimmed.contains(':') && !trimmed.contains(',') {
            let v = text_to_vector(trimmed, &sentiment_map);
            in_tx.send((v, 0.0, true)).unwrap();
            continue;
        }

        for part in trimmed.split(';') {
            let pair: Vec<&str> = part.split(':').collect();
            let (vec_str, target, is_pred) = if pair.len() == 2 {
                (pair[0], pair[1].parse::<f32>().unwrap_or(0.0), false)
            } else {
                (pair[0], 0.0, true)
            };

            let mut v: Vec<f32> = vec_str.split(',')
                .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
                .collect();
            
            if model_path.contains("exam") { v = normalize_input(v); }
            
            if !v.is_empty() { in_tx.send((v, target, is_pred)).unwrap(); }
        }
    }

    drop(in_tx);
    if !is_predict_only {
        save_model(&save_path, &model);
    }
}