use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub trait Layer: Send + Sync {
    fn forward(&self, input: f32) -> f32;
    fn backward(&self, input: f32, grad_output: f32, lr: f32) -> f32;
    fn name(&self) -> String;
    fn to_json(&self) -> serde_json::Value;
}

#[derive(Deserialize, Serialize)]
pub struct Dense {
    pub name: String,
    pub weight: Mutex<f32>,
}

impl Layer for Dense {
    fn forward(&self, input: f32) -> f32 {
        let w = self.weight.lock().unwrap();
        input * (*w)
    }

    fn backward(&self, input: f32, grad_output: f32, lr: f32) -> f32 {
        let mut w = self.weight.lock().unwrap();
        let grad_input = *w * grad_output;
        let grad_weight = input * grad_output;
        *w -= lr * grad_weight;
        grad_input
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn to_json(&self) -> serde_json::Value {
        let w = self.weight.lock().unwrap();
        serde_json::json!({
            "type": "Dense",
            "name": self.name,
            "weight": *w
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct ReLU {
    pub name: String,
}

impl Layer for ReLU {
    fn forward(&self, input: f32) -> f32 {
        if input > 0.0 { input } else { 0.0 }
    }

    fn backward(&self, input: f32, grad_output: f32, _lr: f32) -> f32 {
        if input > 0.0 { grad_output } else { 0.0 }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "ReLU",
            "name": self.name
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct Sigmoid {
    pub name: String,
}

impl Layer for Sigmoid {
    fn forward(&self, input: f32) -> f32 {
        1.0 / (1.0 + (-input).exp())
    }

    fn backward(&self, input: f32, grad_output: f32, _lr: f32) -> f32 {
        let s = self.forward(input);
        grad_output * s * (1.0 - s)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "Sigmoid",
            "name": self.name
        })
    }
}