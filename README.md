# 🦀 Rust Neural Engine

A lightweight neural inference engine built from scratch in **Rust**, focusing on **multi-threaded execution**, **parallel computation**, and **ownership-safe concurrency**.

> **Role**
>
> Individual Project
>
> - Designed a lightweight neural network inference engine
> - Implemented multi-threaded data processing using Rust concurrency primitives
> - Built custom forward propagation and weight management logic
> - Explored Rust's ownership model for safe parallel computation

---

# 📖 About

This project was developed to better understand how neural network inference works at a lower level without relying on existing machine learning frameworks.

Instead of using libraries such as PyTorch or TensorFlow, I implemented the core inference pipeline myself while exploring Rust's ownership system, memory safety, and concurrency model.

The project also served as an opportunity to learn how high-performance backend systems can efficiently utilize multi-threaded architectures.

---

# 🏗️ Architecture

```text
Input Data
      │
      ▼
Feature Mapping
      │
      ▼
Worker Pool
 (Multiple Threads)
      │
      ▼
Dense Layers
      │
      ▼
Activation Functions
      │
      ▼
Prediction
      │
      ▼
Model Serialization
```

---

# ⚙️ Tech Stack

### Language

- Rust

### Concurrency

- std::thread
- Arc
- Mutex
- mpsc Channel

### Serialization

- serde
- serde_json

### Optimization

- Stochastic Gradient Descent (SGD)

---

# ✨ Key Features

## ⚡ Multi-threaded Inference

- Built a worker pool using Rust threads
- Distributed inference workloads across multiple workers
- Shared model parameters safely using `Arc` and `Mutex`

---

## 🧠 Neural Network Engine

Implemented core neural network components including:

- Dense Layer
- Forward Propagation
- Backpropagation
- Weight Updates
- Bias Updates

without relying on external deep learning frameworks.

---

## 📊 Vector-based Processing

- Processed multi-dimensional feature vectors
- Supported batch-style computations
- Designed extensible layer interfaces for future expansion

---

## 💾 Model Persistence

- Saved model weights as JSON
- Loaded trained parameters for inference
- Implemented lightweight serialization using Serde

---

# 📂 Project Structure

```text
.
├── src/
├── sentiment_map.json
├── Cargo.toml
└── README.md
```

---

# 📚 What I Learned

This project helped me understand how neural network inference operates beneath high-level machine learning libraries.

More importantly, I gained hands-on experience with Rust's ownership model, thread-safe memory sharing, and concurrent programming.

Implementing matrix operations and worker pools from scratch also strengthened my understanding of performance-oriented backend programming and how parallel computation can be safely managed without sacrificing reliability.

---

# 🚀 Future Improvements

- SIMD-based matrix computation
- Mini-batch processing
- Generic layer abstraction
- Model checkpointing
- ONNX model import
- Benchmark against existing Rust ML libraries
