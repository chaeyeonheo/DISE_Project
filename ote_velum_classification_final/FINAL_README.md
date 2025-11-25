# OTE/Velum/None 분류 프로젝트 - 최종 통합 패키지

수면 무호흡 검사(DISE) 비디오에서 OTE, Velum, None 영역을 자동 분류하는 딥러닝 시스템

---

## 📦 패키지 구성

### 🔥 핵심 파일 (필수)

#### 1. 데이터 전처리
- **preprocess_with_analyzer.py** ⭐ **추천** - video_analyzer.py 활용, 자동 품질 분석
- **preprocess_videos_robust.py** - 6가지 지표 종합 분석, 시각화 제공
- **preprocess_videos_smart.py** - 밝기/선명도 기반 자동 검사 구간 감지
- **preprocess_videos.py** - 기본 전처리 (고정 시간 기반)

#### 2. 모델 학습 & 추론
- **dataset.py** - PyTorch 데이터셋 & 데이터로더
- **model.py** - ResNet, EfficientNet 등 모델 정의
- **train.py** - 학습 스크립트
- **inference.py** - 추론 스크립트 (비디오/이미지)

#### 3. 모델 공유
- **download_model.py** - 사전학습 모델 다운로드
- **upload_model.py** - 모델 업로드 (Google Drive, Hugging Face)

### 📚 가이드 문서

- **COMPLETE_GUIDE.md** - 전체 사용 가이드 (필독!)
- **ANALYZER_GUIDE.md** - video_analyzer.py 활용 가이드
- **SMART_DETECTION_GUIDE.md** - 자동 검사 구간 감지 가이드
- **NONE_STRATEGY_GUIDE.md** - None 레이블 추출 전략
- **README.md** - 프로젝트 개요

### 📋 기타
- **requirements.txt** - 필요한 Python 패키지
- **example_download.py** - 모델 다운로드 예제

---

## 🚀 빠른 시작 (5단계)

### 1️⃣ 환경 설정
```bash
pip install -r requirements.txt
```

### 2️⃣ 데이터 준비
```bash
mkdir -p dataset/OTE dataset/Velum

# 비디오 파일들을 해당 폴더에 복사
# dataset/OTE/ 에 OTE 비디오들
# dataset/Velum/ 에 Velum 비디오들
```

### 3️⃣ 데이터 전처리 (3가지 방법 중 선택)

#### 방법 A: video_analyzer.py 활용 ⭐ 추천
```bash
# video_analyzer.py가 있는 경우
python preprocess_with_analyzer.py
```

#### 방법 B: Robust 전처리 (시각화 포함)
```bash
python preprocess_videos_robust.py --method combined
```

#### 방법 C: Smart 전처리 (자동 검사 구간 감지)
```bash
python preprocess_videos_smart.py
```

### 4️⃣ 모델 학습
```bash
python train.py
```

### 5️⃣ 추론
```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input test_video.mp4 \
    --visualize
```

---

## 📖 상세 가이드

### 전처리 방법 비교

| 방법 | 파일 | 특징 | 추천 상황 |
|------|------|------|----------|
| **A** | preprocess_with_analyzer.py | video_analyzer.py 활용, ROI 검출 | video_analyzer.py 있을 때 ⭐ |
| **B** | preprocess_videos_robust.py | 6가지 지표, 시각화 제공 | 정밀 분석 필요 시 |
| **C** | preprocess_videos_smart.py | 자동 검사 구간 감지 | 빠른 프로토타입 |
| **D** | preprocess_videos.py | 고정 시간 기반 | 단순한 경우 |

### None 레이블 추출 전략

**None = OTE/Velum이 아닌 모든 영역**
- 내시경 삽입 시 노이즈
- 내시경 제거 시 노이즈
- 전환 구간
- 밝거나 흐린 프레임

자세한 내용: `NONE_STRATEGY_GUIDE.md`

### 학습 설정 커스터마이징

`train.py` 파일 수정:
```python
config = {
    'model_name': 'resnet50',     # 'resnet18', 'efficientnet_b0'
    'batch_size': 32,             # GPU 메모리에 따라 조정
    'epochs': 50,
    'learning_rate': 1e-4,
    'img_size': 224,
}
```

---

## 📁 프로젝트 구조

```
project/
├── dataset/                          # 원본 비디오
│   ├── OTE/
│   │   └── *.mp4
│   └── Velum/
│       └── *.mp4
│
├── processed_dataset/                # 전처리 후 생성
│   ├── OTE/
│   ├── Velum/
│   ├── None/
│   ├── annotations.json
│   └── visualizations/              # (Robust 방법 사용 시)
│
├── checkpoints/                      # 학습 후 생성
│   ├── best_model.pth
│   ├── training_history.json
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── test_results.json
│
└── [코드 파일들]
```

---

## 🔧 주요 기능

### 1. 다양한 전처리 방법
- ROI 기반 자동 검출
- 품질 점수 계산 (밝기, 선명도, 대비 등)
- 자동 검사 구간 감지
- 시각화 제공

### 2. 여러 모델 지원
- ResNet-50, ResNet-18
- EfficientNet-B0
- Custom CNN

### 3. 상세한 평가
- Class-wise metrics
- Confusion matrix
- Training history plots

### 4. 모델 공유
- Google Drive 업로드/다운로드
- Hugging Face Hub 지원

---

## 💡 각 전처리 방법 상세

### A. preprocess_with_analyzer.py

**장점**:
- video_analyzer.py의 ROI 검출 활용
- 프레임별 품질 분석
- 자동 OTE/Velum/None 분류

**사용법**:
```bash
python preprocess_with_analyzer.py --dataset dataset --output processed_dataset
```

**요구사항**: `video_analyzer.py` 필요

### B. preprocess_videos_robust.py

**장점**:
- 6가지 지표 종합 분석
- 상세한 시각화 (8개 그래프)
- 3가지 검사 구간 감지 방법

**사용법**:
```bash
# Combined 방법 (권장)
python preprocess_videos_robust.py --method combined

# 시각화 없이 빠르게
python preprocess_videos_robust.py --no-viz
```

### C. preprocess_videos_smart.py

**장점**:
- 밝기/선명도 기반 자동 감지
- 빠른 처리
- 품질 프로파일 시각화

**사용법**:
```bash
python preprocess_videos_smart.py
```

### D. preprocess_videos.py

**장점**:
- 단순하고 빠름
- 고정 시간 기반

**사용법**:
```bash
python preprocess_videos.py
```

---

## 📊 예상 결과

### 전처리 후
```
Total frames: 2,450

Class distribution:
  OTE: 820 frames (33.5%)
  Velum: 980 frames (40.0%)
  None: 650 frames (26.5%)
```

### 학습 후
```
Epoch 50/50
Train Loss: 0.1234 | Train Acc: 95.43%
Val Loss: 0.2156 | Val Acc: 92.22%

Test Results:
  OTE    - Precision: 0.93, Recall: 0.91, F1: 0.92
  Velum  - Precision: 0.94, Recall: 0.93, F1: 0.94
  None   - Precision: 0.89, Recall: 0.91, F1: 0.90
  
Overall Accuracy: 92.0%
```

---

## 🔍 문제 해결

### 1. GPU 메모리 부족
```python
# train.py 수정
config['batch_size'] = 16  # 32 → 16
config['img_size'] = 128   # 224 → 128
```

### 2. None 클래스 성능 낮음
- None 데이터 추가 수집
- frame_interval 감소 (더 많은 프레임)
- Class weight 사용

### 3. 전처리가 너무 느림
```bash
# 시각화 끄기
python preprocess_videos_robust.py --no-viz

# 또는 더 간단한 방법 사용
python preprocess_videos.py
```

자세한 문제 해결: `COMPLETE_GUIDE.md` 참조

---

## 📝 워크플로우 요약

```
1. 비디오 준비 (dataset/OTE, dataset/Velum)
   ↓
2. 전처리 (preprocess_*.py 중 하나 선택)
   ↓
3. 결과 확인 (processed_dataset/ 및 시각화)
   ↓
4. 학습 (train.py)
   ↓
5. 평가 (checkpoints/ 결과 확인)
   ↓
6. 추론 (inference.py)
```

---

## 🎯 추천 시나리오

### 시나리오 1: video_analyzer.py 있음
```bash
# 1. 전처리
python preprocess_with_analyzer.py

# 2. 학습
python train.py

# 3. 추론
python inference.py --model checkpoints/best_model.pth --input test.mp4
```

### 시나리오 2: 정밀 분석 필요
```bash
# 1. Robust 전처리 (시각화 포함)
python preprocess_videos_robust.py --method combined

# 2. 시각화 확인
ls processed_dataset/visualizations/

# 3. 학습
python train.py
```

### 시나리오 3: 빠른 프로토타입
```bash
# 1. 기본 전처리
python preprocess_videos.py

# 2. 경량 모델로 빠른 학습
# train.py에서 model_name='resnet18' 설정

python train.py
```

---

## 📚 참고 문서

- **COMPLETE_GUIDE.md**: 전체 사용 가이드 (필독!)
- **ANALYZER_GUIDE.md**: video_analyzer.py 활용
- **SMART_DETECTION_GUIDE.md**: 자동 검사 구간 감지
- **NONE_STRATEGY_GUIDE.md**: None 레이블 전략

---

## 🎓 고급 기능

### 모델 다운로드
```bash
python download_model.py --model resnet50_ote_velum_v1
```

### 모델 업로드
```bash
# Google Drive
python upload_model.py --model checkpoints/best_model.pth --platform gdrive

# Hugging Face
python upload_model.py --model checkpoints/best_model.pth --platform huggingface --hf-repo username/model-name
```

---

## ✅ 체크리스트

### 전처리 전
- [ ] dataset/OTE와 dataset/Velum에 비디오 준비
- [ ] requirements.txt 패키지 설치
- [ ] 전처리 방법 선택

### 전처리 후
- [ ] processed_dataset/ 폴더 확인
- [ ] annotations.json 생성 확인
- [ ] (Robust 사용 시) visualizations/ 확인

### 학습 후
- [ ] training_history.png 확인
- [ ] confusion_matrix.png 확인
- [ ] test_results.json 확인

### 배포 전
- [ ] 실제 비디오로 inference 테스트
- [ ] 오분류 패턴 분석
- [ ] 필요시 데이터 추가 수집

---

## 🆘 지원

문제가 발생하면:
1. COMPLETE_GUIDE.md의 문제 해결 섹션 확인
2. 각 가이드 문서 참조
3. 코드 내 주석 확인

---

## 📄 라이센스

이 프로젝트는 연구 목적으로 제공됩니다.

---

**🎉 이제 시작하세요!**

```bash
python preprocess_with_analyzer.py  # 또는 다른 전처리 방법
python train.py
python inference.py --model checkpoints/best_model.pth --input test.mp4 --visualize
```
