use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub trait Layer: Send + Sync {
    fn forward(&self, input: Vec<f32>) -> Vec<f32>;
    fn backward(&self, input: Vec<f32>, grad_output: Vec<f32>, lr: f32) -> Vec<f32>;
    fn name(&self) -> String;
    fn to_json(&self) -> serde_json::Value;
}

#[derive(Deserialize, Serialize)]
pub struct Dense {
    pub name: String,
    pub weights: Mutex<Vec<f32>>,
    pub bias: Mutex<f32>,
}

impl Layer for Dense {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        let w = self.weights.lock().unwrap();
        let b = self.bias.lock().unwrap();
        let sum: f32 = input.iter().zip(w.iter()).map(|(i, w_val)| i * w_val).sum();
        vec![sum + *b]
    }

    fn backward(&self, input: Vec<f32>, grad_output: Vec<f32>, lr: f32) -> Vec<f32> {
        let mut w = self.weights.lock().unwrap();
        let mut b = self.bias.lock().unwrap();
        let g_out = grad_output[0];

        let grad_input: Vec<f32> = w.iter().map(|w_val| w_val * g_out).collect();
        
        for (i, x) in input.iter().enumerate() {
            w[i] -= lr * x * g_out;
        }
        *b -= lr * g_out;

        grad_input
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn to_json(&self) -> serde_json::Value {
        let w = self.weights.lock().unwrap();
        let b = self.bias.lock().unwrap();
        serde_json::json!({
            "type": "Dense",
            "name": self.name,
            "weights": *w,
            "bias": *b
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct ReLU {
    pub name: String,
}

impl Layer for ReLU {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| if x > 0.0 { x } else { 0.0 }).collect()
    }

    fn backward(&self, input: Vec<f32>, grad_output: Vec<f32>, _lr: f32) -> Vec<f32> {
        input.into_iter().zip(grad_output.into_iter())
            .map(|(x, g)| if x > 0.0 { g } else { 0.0 })
            .collect()
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
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    fn backward(&self, input: Vec<f32>, grad_output: Vec<f32>, _lr: f32) -> Vec<f32> {
        let s_vec = self.forward(input);
        grad_output.into_iter().zip(s_vec.into_iter())
            .map(|(g, s)| g * s * (1.0 - s))
            .collect()
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