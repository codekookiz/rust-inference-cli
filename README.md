# Rust Multi-Threaded Inference Engine

이 프로젝트는 **Rust**를 활용하여 고성능 AI 추론 파이프라인을 시뮬레이션하는 CLI 도구입니다. 
멀티스레딩과 메시지 패싱(MPSC)을 활용하여 대량의 데이터를 병렬로 처리하며, JSON 설정을 통해 유연하게 모델 구조를 변경할 수 있습니다.

## Key Features

- **Parallel Processing**: `std::thread`와 `mpsc` 채널을 활용한 워커 풀(Worker Pool) 아키텍처.
- **Dynamic Configuration**: 코드 수정 없이 `model.json` 파일만으로 레이어(Dense, ReLU 등) 순서와 파라미터 변경 가능.
- **Memory Safety**: Rust의 소유권(Ownership) 및 스마트 포인터(`Arc`, `Mutex`, `Box`)를 활용한 안전한 리소스 공유.
- **Modular Design**: 트레이트(Trait) 기반 추상화로 새로운 레이어 타입을 손쉽게 확장 가능.

## Architecture

- **Main Thread**: 모델 로드 및 데이터 투입(Producer).
- **Worker Threads**: 입력을 받아 순전파(Forward Pass) 연산 수행 및 결과 반환(Consumer).
- **Layers**: 
  - `Dense`: 가중치(factor) 기반 선형 연산.
  - `ReLU`: 비선형 활성화 함수 시뮬레이션.