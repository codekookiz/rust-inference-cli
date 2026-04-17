use serde::Deserialize;

pub trait Layer: Send + Sync {
    fn run(&self, input: f32) -> f32;
    fn name(&self) -> String;
}

#[derive(Deserialize)]
pub struct Dense {
    pub name: String,
    pub factor: f32,
}

impl Layer for Dense {
    fn run(&self, input: f32) -> f32 {
        input * self.factor
    }
    fn name(&self) -> String {
        self.name.clone()
    }
}

#[derive(Deserialize)]
pub struct ReLU {
    pub name: String,
}

impl Layer for ReLU {
    fn run(&self, input: f32) -> f32 {
        if input > 0.0 { input } else { 0.0 }
    }
    fn name(&self) -> String {
        self.name.clone()
    }
}