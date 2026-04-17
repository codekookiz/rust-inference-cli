# Rust Multi-Threaded Neural Engine (RMNE)

RMNE는 Rust의 소유권(Ownership)과 강력한 동시성 모델을 활용하여 바닥부터(From Scratch) 구현된 **병렬 학습 및 추론 신경망 엔진**입니다. 단순한 단일 값 연산을 넘어 다차원 벡터 연산과 자연어 감성 분석 매핑 기능을 지원합니다.

## Key Features

- **Vector-Based Computation**: 모든 레이어 간 데이터 흐름을 `Vec<f32>` 기반으로 처리하여 다차원 피처(Multi-features) 학습 지원.
- **Advanced Multi-Threading**: `std::thread`와 `MPSC` 채널을 이용한 워커 풀(Worker Pool) 구조로 데이터 병렬 학습 최적화.
- **Sentiment Mapping Engine**: 자연어 문장을 수치 벡터로 변환하는 외부 매핑 딕셔너리(`sentiment_map.json`) 연동 기능.
- **Optimized I/O Pipeline**: 학습 진행률 카운터 및 비동기 출력 제어를 통해 대량 데이터 학습 시 터미널 병목 현상 제거.
- **Flexible Persistence**: 학습된 가중치(Weights)와 편향(Bias)을 JSON 포맷으로 실시간 직렬화/역직렬화 지원.

## Architecture

1. **Main Interface**: 사용자로부터 자연어 문장이나 수치 벡터를 입력받아 전처리 후 워커 채널로 주입.
2. **Worker Pool (3 Threads)**: `Arc<Vec<Box<dyn Layer>>>`를 공유하며, 각각의 스레드가 독립적으로 순전파(Forward) 및 역전파(Backward) 수행.
3. **Internal Logic**:
   - **Dense Layer**: 가중치 행렬 내적(Dot Product) 및 편향(Bias) 연산, `Mutex` 기반의 Thread-safe 가중치 업데이트.
   - **Activation Layers**: 비선형성 확보를 위한 `ReLU`, `Sigmoid` 미분 정의 및 역전파 구현.

## Technical Stack

- **Language**: Rust
- **Concurrency**: std::sync::{Arc, Mutex, mpsc}
- **Serialization**: serde, serde_json
- **Optimization**: Stochastic Gradient Descent (SGD) simulation