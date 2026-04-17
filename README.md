# Rust Multi-Threaded Learning & Inference Engine

이 프로젝트는 Rust를 활용하여 병렬 AI 추론 및 역전파 기반의 학습 파이프라인을 구현한 CLI 도구입니다. 멀티스레딩과 MPSC 채널 통신을 통해 데이터 병렬 학습을 수행하며, 가중치 업데이트와 모델 상태 보존 기능을 지원합니다.

## Key Features

- **Parallel Training**: std::thread와 mpsc 채널을 활용하여 여러 워커가 동시에 역전파를 수행하고 가중치를 업데이트하는 병렬 학습 아키텍처.
- **Backpropagation**: 각 레이어별 미분 정의를 통한 경사 하강법(Gradient Descent) 구현.
- **Dynamic Configuration & Persistence**: init_model.json을 통한 모델 초기화 및 학습 완료된 상태를 trained_model.json으로 자동 저장 및 로드.
- **Thread-Safe State Management**: Arc와 Mutex를 활용하여 여러 스레드가 동시에 가중치를 안전하게 업데이트할 수 있는 구조.
- **Modular Design**: Layer 트레이트를 기반으로 Dense, ReLU, Sigmoid 등 다양한 레이어를 유연하게 확장 및 조합 가능.

## Architecture

- **Main Thread**: 모델 파일 로드, 사용자 입력(값:정답) 파싱 및 워커 스레드로 데이터 배분. 프로그램 종료 시 최신 가중치 저장 관리.
- **Worker Threads**: 입력을 받아 순전파(Forward Pass) 연산 후 오차를 계산하고, 역순으로 역전파(Backward Pass)를 수행하여 실시간 가중치 업데이트.
- **Layers**: 
  - Dense: 가중치(Weight) 기반 선형 연산 및 가중치 업데이트 미분 구현.
  - ReLU: 양수 신호 유지 및 음수 차단을 통한 비선형 활성화 함수.
  - Sigmoid: 출력을 0과 1 사이로 압착하여 확률 기반 예측 및 비선형성 추가.

## Model Configuration (JSON)

모델의 구조는 JSON 형식을 통해 정의됩니다. 각 레이어는 고유한 이름과 타입, 초기 가중치를 가집니다.

```json
[
  { "type": "Dense", "name": "L1", "weight": 0.5 },
  { "type": "ReLU", "name": "R1" },
  { "type": "Dense", "name": "L2", "weight": -0.8 },
  { "type": "Sigmoid", "name": "S1" }
]
```

## Technical Stack

- **Language**: Rust
- **Concurrency**: std::sync::{Arc, Mutex, mpsc}
- **Serialization**: serde, serde_json
- **Optimization**: Stochastic Gradient Descent (SGD) simulation