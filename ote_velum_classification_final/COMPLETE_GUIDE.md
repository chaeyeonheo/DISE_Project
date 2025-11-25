# 📚 OTE/Velum 분류 모델 - 완전 사용 가이드

## 🎯 개요

이 프로젝트는 수면 무호흡 검사(DISE) 비디오에서 OTE, Velum, None 영역을 자동 분류하는 딥러닝 모델입니다.

---

## 📁 프로젝트 구조

```
project/
├── dataset/                          # 원본 비디오 (사용자가 준비)
│   ├── OTE/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── Velum/
│       ├── video1.mp4
│       └── video2.mp4
│
├── processed_dataset/                # 전처리 후 생성됨
│   ├── OTE/                         # OTE 프레임 이미지
│   ├── Velum/                       # Velum 프레임 이미지
│   ├── None/                        # None 프레임 이미지
│   ├── annotations.json             # 전체 레이블 정보
│   └── visualizations/              # 분석 그래프
│       ├── video1_analysis.png
│       └── video2_analysis.png
│
├── checkpoints/                      # 학습 후 생성됨
│   ├── best_model.pth               # 최고 성능 모델
│   ├── training_history.json
│   └── confusion_matrix.png
│
└── [코드 파일들]
    ├── preprocess_videos_robust.py  # 전처리 (추천)
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── inference.py
    └── requirements.txt
```

---

## 🚀 빠른 시작 (5단계)

### 1단계: 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2단계: 데이터 준비

```bash
# 폴더 구조 생성
mkdir -p dataset/OTE dataset/Velum

# 비디오 파일을 해당 폴더에 복사
# dataset/OTE/ 에 OTE 비디오들
# dataset/Velum/ 에 Velum 비디오들
```

### 3단계: 데이터 전처리 ⭐ 중요!

```bash
# 방법 1: Robust 전처리 (추천) - 자동 검사 구간 감지
python preprocess_videos_robust.py

# 결과 확인
ls processed_dataset/               # 프레임 이미지들
ls processed_dataset/visualizations/  # 분석 그래프들
```

**중요**: `visualizations` 폴더의 그래프들을 **꼭 확인**하세요!
- 각 비디오의 품질 프로파일과 자동 감지된 검사 구간 확인
- 이상한 패턴이 있으면 수동 조정 필요

### 4단계: 모델 학습

```bash
python train.py
```

학습 중 출력:
```
Epoch 1/50
Train Loss: 0.8234 | Train Acc: 65.43%
Val Loss: 0.7156 | Val Acc: 71.22%
✓ Best model saved with val_acc: 71.22%
```

### 5단계: 추론

```bash
# 비디오 분석
python inference.py \
    --model checkpoints/best_model.pth \
    --input test_video.mp4 \
    --visualize

# 결과 확인
# - results.json: 프레임별 예측 결과
# - test_video_predictions.mp4: 시각화된 비디오
```

---

## 📊 전처리 상세 가이드

### Robust 전처리의 특징

**6가지 지표를 종합 분석**:

1. **밝기 (Brightness)**: 화면의 평균 밝기
2. **선명도 (Sharpness)**: Laplacian variance로 측정
3. **대비 (Contrast)**: 픽셀 값의 표준편차
4. **엣지 밀도 (Edge Density)**: Canny edge 비율
5. **색상 분산 (Color Variance)**: RGB 채널 표준편차
6. **움직임 (Motion)**: 프레임 간 차이

**종합 점수 계산**:
```python
combined_score = (
    0.25 × brightness +
    0.25 × sharpness +
    0.20 × contrast +
    0.15 × edge_density +
    0.10 × color_variance +
    0.05 × (1 - motion)  # 움직임 적을수록 좋음
)
```

### 3가지 검사 구간 감지 방법

#### 1) threshold (빠름)
```bash
python preprocess_videos_robust.py --method threshold
```
- 종합 점수가 0.6 이상인 구간을 검사 구간으로 간주
- 가장 빠르지만 단순함

#### 2) derivative (중간)
```bash
python preprocess_videos_robust.py --method derivative
```
- 점수의 변화율(1차 미분) 분석
- 급상승 지점 = 검사 시작
- 급하강 지점 = 검사 종료

#### 3) combined (권장) ⭐
```bash
python preprocess_videos_robust.py --method combined
```
- 적응형 임계값 + 연속성 분석
- 가장 긴 고품질 구간을 검사 구간으로 선택
- **가장 robust하고 정확함**

### 시각화 그래프 읽는 법

생성된 그래프 (`processed_dataset/visualizations/*.png`)를 확인하세요:

```
[그래프 구성]
┌──────────────┬──────────────┐
│ Brightness   │ Sharpness    │
├──────────────┼──────────────┤
│ Contrast     │ Edge Density │
├──────────────┼──────────────┤
│ Color Var    │ Motion       │
├──────────────┼──────────────┤
│ Combined     │ Detection    │
└──────────────┴──────────────┘

초록 세로선 = 검사 시작
빨강 세로선 = 검사 종료
```

**좋은 예시**:
```
Combined Score
1.0 │         ┌──────────────┐
0.8 │         │  검사 구간    │
0.6 │     ┌───┤              ├───┐
0.4 │  ┌──┘   │              │   └──┐
0.2 │──┘      │              │      └──
    └─────────┴──────────────┴─────────→
    삽입      검사 시작      검사 끝    제거
```

**나쁜 예시** (수동 확인 필요):
```
Combined Score
1.0 │ ┌──┐    ┌──┐    ┌──┐
0.8 │ │  │    │  │    │  │  ← 계속 변동
0.6 │─┘  └────┘  └────┘  └───
    └────────────────────────→
    (검사 구간 불명확)
```

---

## ⚙️ 주요 설정 조정

### 전처리 설정

```bash
# 시각화 끄기 (빠른 처리)
python preprocess_videos_robust.py --no-viz

# 다른 감지 방법 시도
python preprocess_videos_robust.py --method threshold
python preprocess_videos_robust.py --method derivative
```

### 학습 설정

`train.py` 파일 수정:

```python
config = {
    'model_name': 'resnet50',        # 'resnet18', 'efficientnet_b0'
    'batch_size': 32,                # GPU 메모리에 따라 조정
    'epochs': 50,                    # 더 길게/짧게
    'learning_rate': 1e-4,           # 학습률
    'img_size': 224,                 # 이미지 크기
}
```

### 추론 설정

```bash
# 프레임 간격 조정 (빠른 처리)
python inference.py \
    --model checkpoints/best_model.pth \
    --input video.mp4 \
    --frame-interval 5  # 5프레임마다 1개만 분석

# 세그먼트 분석 (3초 단위)
python inference.py \
    --model checkpoints/best_model.pth \
    --input video.mp4 \
    --segment-analysis \
    --segment-duration 3.0
```

---

## 🔧 문제 해결 (Troubleshooting)

### 문제 1: 전처리 시 "검사 구간이 이상함"

**증상**: 시각화를 보니 검사 구간이 잘못 감지됨

**해결**:
```python
# preprocess_videos_robust.py 수정
# detect_examination_period 함수 내부

# 임계값 조정
adaptive_threshold = median_score + 0.5 * std_score  # 0.5를 0.3~0.7로 조정

# 또는 다른 감지 방법 시도
python preprocess_videos_robust.py --method threshold
```

### 문제 2: GPU 메모리 부족

**증상**: `CUDA out of memory`

**해결**:
```python
# train.py 수정
config['batch_size'] = 16  # 32 → 16 → 8
config['img_size'] = 128   # 224 → 128

# 또는 더 작은 모델 사용
config['model_name'] = 'resnet18'  # resnet50 대신
```

### 문제 3: None 클래스 성능이 낮음

**증상**: Confusion matrix에서 None 오분류가 많음

**해결**:

1. **더 많은 None 데이터 수집**
```bash
# 별도 None 비디오 추가
mkdir dataset/None
# 삽입/제거 과정만 담은 비디오들 추가
```

2. **frame_interval 조정**
```python
# preprocess_videos_robust.py 수정
# extract_frames 호출 시
frame_interval=2  # 더 작게 (더 많은 None 프레임)
```

3. **Class Weight 사용**
```python
# train.py에 추가
from torch.nn import CrossEntropyLoss

# 클래스 가중치 계산
class_counts = [len_ote, len_velum, len_none]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
criterion = CrossEntropyLoss(weight=weights)
```

### 문제 4: 학습이 수렴하지 않음

**증상**: Loss가 감소하지 않음, accuracy가 낮음

**해결**:
```python
# train.py 수정

# 1. Learning rate 조정
config['learning_rate'] = 5e-5  # 더 작게

# 2. 더 오래 학습
config['epochs'] = 100

# 3. Scheduler 변경
config['scheduler'] = 'cosine'  # 'plateau' 대신
```

---

## 📈 성능 평가 가이드

### 학습 후 확인할 것들

#### 1. Training History
```bash
# checkpoints/training_history.png 확인
```
- Loss가 감소하는가?
- Train/Val gap이 크지 않은가? (과적합 체크)
- Accuracy가 안정적으로 증가하는가?

#### 2. Confusion Matrix
```bash
# checkpoints/confusion_matrix.png 확인
```

**좋은 예시**:
```
           OTE  Velum  None
OTE       [850    50    10]
Velum     [ 40   870    15]
None      [ 10    20   880]

→ 대각선에 집중, 오분류 적음
```

**나쁜 예시**:
```
           OTE  Velum  None
OTE       [600   200   200]  ← OTE를 Velum/None으로 많이 오분류
Velum     [150   700   150]
None      [300   200   500]  ← None 성능이 매우 낮음

→ None 데이터 추가 필요
```

#### 3. Classification Report
```bash
# checkpoints/test_results.json 확인
```

```json
{
  "OTE": {
    "precision": 0.89,
    "recall": 0.91,
    "f1-score": 0.90
  },
  "Velum": {
    "precision": 0.92,
    "recall": 0.88,
    "f1-score": 0.90
  },
  "None": {
    "precision": 0.75,  ← None이 낮음
    "recall": 0.70,
    "f1-score": 0.72
  }
}
```

**목표 성능**:
- Overall Accuracy: 85%+
- 각 클래스 F1-score: 80%+

---

## 💡 Best Practices

### 1. 전처리 단계

✅ **DO**:
- 항상 시각화 확인 (`visualizations` 폴더)
- 이상한 비디오는 수동으로 제외
- 여러 detection method 비교

❌ **DON'T**:
- 시각화 없이 바로 학습
- 모든 비디오를 무조건 포함
- 한 가지 방법만 고집

### 2. 학습 단계

✅ **DO**:
- 작은 모델로 빠르게 시작 (resnet18)
- 과적합 모니터링 (train/val gap)
- 정기적으로 checkpoint 저장

❌ **DON'T**:
- 처음부터 큰 모델 (resnet50)
- Val accuracy만 보고 판단
- 한 번만 학습하고 끝

### 3. 평가 단계

✅ **DO**:
- Confusion matrix 상세 분석
- 클래스별 성능 확인
- 실제 비디오로 테스트

❌ **DON'T**:
- Overall accuracy만 확인
- Test set 결과만 믿음
- 학습 데이터로만 평가

---

## 📞 추가 도움말

### 자주 묻는 질문 (FAQ)

**Q: 비디오가 너무 많아서 전처리가 오래 걸려요**
```bash
# 병렬 처리는 없지만, 시각화를 끄면 빠릅니다
python preprocess_videos_robust.py --no-viz
```

**Q: 특정 비디오만 처리하고 싶어요**
```python
# preprocess_videos_robust.py 수정
video_files = list(velum_path.glob('video1.mp4'))  # 특정 파일만
```

**Q: 학습된 모델을 다른 컴퓨터에서 사용하려면?**
```bash
# 모델 파일만 복사
cp checkpoints/best_model.pth /path/to/destination/

# 새 컴퓨터에서
python inference.py --model best_model.pth --input video.mp4
```

**Q: 클래스 비율이 불균형해요 (OTE:Velum:None = 40:40:20)**
```python
# train.py에 class weight 추가
weights = torch.tensor([1.0, 1.0, 2.0])  # None에 2배 가중치
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 📚 다음 단계

### 고급 주제

1. **Data Augmentation 강화**
   - Mixup, CutMix 적용
   - 의료 영상 특화 augmentation

2. **앙상블 모델**
   - 여러 모델의 예측 결합
   - Test-Time Augmentation (TTA)

3. **Video-level Classification**
   - 프레임 단위 → 비디오 단위
   - Temporal modeling (LSTM, Transformer)

4. **Active Learning**
   - 모델이 불확실한 샘플 재레이블링
   - 점진적 성능 개선

---

## ✅ 체크리스트

전처리 전:
- [ ] dataset/OTE 와 dataset/Velum에 비디오 준비
- [ ] requirements.txt 패키지 설치

전처리 후:
- [ ] processed_dataset/visualizations 그래프 확인
- [ ] 이상한 비디오 없는지 체크
- [ ] annotations.json 파일 생성 확인

학습 후:
- [ ] training_history.png 확인 (과적합 체크)
- [ ] confusion_matrix.png 확인
- [ ] test_results.json 확인 (클래스별 성능)

배포 전:
- [ ] 실제 비디오로 inference 테스트
- [ ] 오분류 패턴 분석
- [ ] 필요시 데이터 추가 수집

---

**이제 시작하세요!** 🚀

```bash
# 1. 전처리
python preprocess_videos_robust.py

# 2. 학습
python train.py

# 3. 추론
python inference.py --model checkpoints/best_model.pth --input test.mp4 --visualize
```
