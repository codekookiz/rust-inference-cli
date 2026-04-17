mod layers;

use layers::{Layer, Dense, ReLU};
use serde_json::Value;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

fn load_model(path: &str) -> Vec<Box<dyn Layer>> {
    let data = std::fs::read_to_string(path)
        .expect("model.json 파일을 찾을 수 없습니다.");
    
    let json: Value = serde_json::from_str(&data)
        .expect("JSON 파싱에 실패했습니다.");

    let mut layers: Vec<Box<dyn Layer>> = Vec::new();

    if let Some(arr) = json.as_array() {
        for item in arr {
            let layer_type = item["type"].as_str().expect("type 필드가 없습니다.");
            match layer_type {
                "Dense" => {
                    let d: Dense = serde_json::from_value(item.clone()).unwrap();
                    layers.push(Box::new(d));
                },
                "ReLU" => {
                    let r: ReLU = serde_json::from_value(item.clone()).unwrap();
                    layers.push(Box::new(r));
                },
                _ => println!("경고: 알 수 없는 레이어 타입 '{}'", layer_type),
            }
        }
    }
    layers
}

fn main() {
    let model_list = load_model("model.json");
    let model = Arc::new(model_list);
    println!("🚀 모델 로드 완료: {}개의 레이어", model.len());

    let (in_tx, in_rx) = mpsc::channel::<f32>();
    let (out_tx, out_rx) = mpsc::channel::<(usize, f32)>();
    let shared_in_rx = Arc::new(Mutex::new(in_rx));

    let mut handles = vec![];

    for worker_id in 0..3 {
        let model_ref = Arc::clone(&model);
        let rx_ref = Arc::clone(&shared_in_rx);
        let tx_res = out_tx.clone();

        let handle = thread::spawn(move || {
            loop {
                let val = {
                    let lock = rx_ref.lock().unwrap();
                    match lock.recv() {
                        Ok(v) => v,
                        Err(_) => break,
                    }
                };

                let mut result = val;
                for layer in model_ref.iter() {
                    result = layer.run(result);
                }

                tx_res.send((worker_id, result)).unwrap();
            }
        });
        handles.push(handle);
    }

    let inputs = vec![-1.0, 2.0, -3.0, 4.0, 5.0, 10.0];
    for i in inputs {
        in_tx.send(i).unwrap();
    }

    drop(in_tx);
    drop(out_tx);

    println!("\n--- 추론 결과 수집 시작 ---");
    for (id, res) in out_rx {
        println!("Worker {} 가 연산한 결과: {}", id, res);
    }

    for handle in handles {
        handle.join().unwrap();
    }
    println!("\n✅ 모든 작업이 완료되었습니다.");
}