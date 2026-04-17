mod layers;

use layers::{Layer, Dense, ReLU};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use serde_json::Value;

fn load_model(path: &str) -> Vec<Box<dyn Layer>> {
    let data = std::fs::read_to_string(path)
        .expect("model.json 파일을 찾을 수 없습니다.");
    
    let json: Value = serde_json::from_str(&data)
        .expect("JSON 형식이 올바르지 않습니다.");

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
    let model: Arc<Vec<Box<dyn Layer>>> = Arc::new(vec![
        Box::new(Dense { name: "Linear_1".into(), factor: 2.0 }),
        Box::new(ReLU { name: "ReLU_1".into() }),
    ]);

    println!("모델 로드 완료: {}개의 레이어", model.len());
}