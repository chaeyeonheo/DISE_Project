# video_analyzer.py를 활용한 데이터셋 전처리 가이드

## 📋 개요

업로드하신 이미지들을 보니 수면 내시경(DISE) 영상의 다양한 상태가 있습니다:
- **명확한 OTE/Velum 프레임** (frame_00024~00043)
- **밝거나 흐릿한 None 프레임** (frame_00044~00051)

`video_analyzer.py`의 `AirwayOcclusionAnalyzer`를 활용하여 자동으로 분류하는 파이프라인을 만들었습니다!

---

## 🎯 작동 원리

### 1단계: video_analyzer로 비디오 분석
```python
analyzer = AirwayOcclusionAnalyzer(
    fps_extract=5,              # 초당 5프레임 추출
    threshold_percent=30,
    exclude_first_seconds=2,    # 앞 2초 제외
    exclude_last_seconds=3      # 뒤 3초 제외
)

# ROI 영역 검출 + 품질 분석
results = analyzer.analyze_video(video_path)
```

### 2단계: 각 프레임 품질 분석
```python
quality_score = (
    0.25 × brightness +      # 밝기
    0.25 × sharpness +       # 선명도
    0.20 × contrast +        # 대비
    0.15 × edge_density +    # 엣지 밀도
    0.15 × color_std         # 색상 분산
)
```

### 3단계: 자동 분류

**None으로 분류되는 경우**:
- ❌ 품질 점수 < 0.4 (밝거나 흐림)
- ❌ ROI 영역이 없음
- ❌ ROI 면적이 최대 대비 30% 미만
- ❌ 평균 밝기 > 200 (과다 노출)

**OTE/Velum으로 분류되는 경우**:
- ✅ 위 조건에 해당하지 않음
- ✅ 원본 비디오 타입 그대로 유지

---

## 🚀 사용 방법

### 필수 파일
```
project/
├── video_analyzer.py          # 기존 코드 (업로드하신 파일)
├── preprocess_with_analyzer.py # 새로 생성된 코드
└── dataset/                   # 원본 비디오
    ├── OTE/
    │   ├── video1.mp4
    │   └── video2.mp4
    └── Velum/
        ├── video1.mp4
        └── video2.mp4
```

### 실행

```bash
# 기본 실행
python preprocess_with_analyzer.py

# 커스텀 경로 지정
python preprocess_with_analyzer.py \
    --dataset my_dataset \
    --output my_processed_dataset
```

### 출력 결과

```
processed_dataset/
├── OTE/
│   ├── video1_OTE_frame_000000.jpg
│   ├── video1_OTE_frame_000001.jpg
│   └── ...
├── Velum/
│   ├── video2_Velum_frame_000000.jpg
│   └── ...
├── None/
│   ├── video1_None_frame_000000.jpg  # 밝거나 흐린 프레임들
│   └── ...
├── annotations.json           # 전체 레이블 정보
└── dataset_stats.json         # 통계
```

---

## 📊 annotations.json 구조

```json
[
  {
    "filename": "video1_OTE_frame_000000.jpg",
    "label": "OTE",
    "video_name": "video1",
    "video_type": "OTE",
    "frame_number": 120,
    "timestamp": 5.2,
    "roi_area": 45230,
    "quality_score": 0.78,
    "confidence": 0.85,
    "metrics": {
      "brightness": 0.65,
      "sharpness": 0.82,
      "contrast": 0.71,
      "edge_density": 0.45,
      "color_std": 0.68,
      "quality_score": 0.78
    },
    "path": "processed_dataset/OTE/video1_OTE_frame_000000.jpg"
  },
  ...
]
```

---

## 🔍 분류 예시

### 업로드하신 이미지 기준

**OTE/Velum으로 분류될 프레임들**:
- frame_00024~00027: 명확한 기도 구조
- frame_00030~00043: ROI 영역 명확, 품질 양호

**None으로 분류될 프레임들**:
- frame_00044~00051: 과다 노출, ROI 없음
- 밝기 > 200, quality_score < 0.4

---

## ⚙️ 설정 조정

### 프레임 추출 빈도 조정

```python
# preprocess_with_analyzer.py 내부 수정

analyzer = AirwayOcclusionAnalyzer(
    fps_extract=10,  # 5 → 10으로 증가 (더 많은 프레임)
    threshold_percent=30,
    exclude_first_seconds=3,  # 2 → 3으로 증가
    exclude_last_seconds=5    # 3 → 5로 증가
)
```

### 분류 임계값 조정

```python
# preprocess_with_analyzer.py의 classify_frame 메서드 수정

# None 분류를 더 엄격하게
if quality_score < 0.5:  # 0.4 → 0.5
    return 'None', 1.0 - quality_score

# ROI 면적 임계값 조정
if area_ratio < 0.4:  # 0.3 → 0.4 (더 많이 None으로)
    return 'None', 0.8
```

---

## 📈 실행 결과 예시

```
====================================================================
DISE Dataset Preprocessing Pipeline
====================================================================

📹 Found 5 Velum videos

====================================================================
Processing: velum_video1.mp4 (Type: Velum)
====================================================================

📹 비디오 정보
  - 파일: velum_video1.mp4
  - FPS: 30.00
  - 총 프레임: 900
  - 실제 처리 구간: [60 .. 810]

Classifying and saving frames...
Frames: 100%|████████████| 150/150 [00:15<00:00, 10.2it/s]

✓ velum_video1.mp4 처리 완료
  - OTE: 0 frames
  - Velum: 120 frames
  - None: 30 frames

====================================================================
=== Final Statistics ===
====================================================================
Total frames: 750

Class distribution:
  OTE: 280 frames (37.3%)
  Velum: 320 frames (42.7%)
  None: 150 frames (20.0%)

Video types:
  OTE: 280 frames
  Velum: 470 frames

====================================================================
✅ Preprocessing completed!
====================================================================

💡 Next steps:
1. Check the frames in processed_dataset/[OTE|Velum|None]/
2. Review dataset_stats.json for class distribution
3. Run: python train.py
```

---

## 🔧 문제 해결

### 문제 1: None이 너무 많이 분류됨

**해결**:
```python
# classify_frame 메서드에서 임계값 낮추기
if quality_score < 0.3:  # 0.4 → 0.3
    return 'None', 1.0 - quality_score
```

### 문제 2: None이 너무 적게 분류됨

**해결**:
```python
# 임계값 높이기
if quality_score < 0.5:  # 0.4 → 0.5
    return 'None', 1.0 - quality_score

# ROI 면적 기준 강화
if area_ratio < 0.5:  # 0.3 → 0.5
    return 'None', 0.8
```

### 문제 3: 특정 비디오 처리 실패

**확인사항**:
1. 비디오 코덱 문제
   ```bash
   ffmpeg -i video.mp4 -c:v libx264 -c:a aac video_fixed.mp4
   ```

2. 비디오가 너무 짧음
   - `exclude_first_seconds`와 `exclude_last_seconds` 조정

---

## 📝 다음 단계

### 1. 결과 확인

```bash
# 프레임 확인
ls processed_dataset/OTE/
ls processed_dataset/Velum/
ls processed_dataset/None/

# 통계 확인
cat processed_dataset/dataset_stats.json
```

### 2. 데이터 검증

```python
# 샘플 확인 스크립트
import json
import random
import cv2
import matplotlib.pyplot as plt

with open('processed_dataset/annotations.json') as f:
    annotations = json.load(f)

# 각 클래스에서 랜덤 샘플
for label in ['OTE', 'Velum', 'None']:
    samples = [a for a in annotations if a['label'] == label]
    sample = random.choice(samples)
    
    img = cv2.imread(sample['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"{label} - Quality: {sample['quality_score']:.2f}")
    plt.axis('off')
    plt.show()
```

### 3. 학습 시작

```bash
python train.py
```

---

## 💡 핵심 장점

1. **video_analyzer.py 활용**: 기존 ROI 검출 로직 재사용
2. **자동 품질 평가**: 6가지 지표 종합 분석
3. **적응형 분류**: 비디오별 최대 ROI 면적 기준
4. **상세 어노테이션**: 품질 점수, 신뢰도 포함

---

## 🎓 추가 개선 아이디어

### 1. 수동 검증 단계 추가

```python
# 낮은 confidence 프레임 수동 확인
low_conf_frames = [
    a for a in annotations 
    if a['confidence'] < 0.6
]

# 이미지 표시하고 수동으로 레이블 조정
```

### 2. 클래스 균형 조정

```python
# None 클래스가 부족하면 augmentation
from albumentations import *

transform = Compose([
    RandomBrightnessContrast(p=0.5),
    GaussianBlur(p=0.3),
    GaussNoise(p=0.3)
])
```

### 3. Active Learning

```python
# 1차 학습 후 모델이 불확실한 프레임 재검토
# Confidence가 낮은 프레임들을 재레이블링
```

---

## 📞 요약

1. **video_analyzer.py**: ROI 검출 + 프레임 전처리
2. **preprocess_with_analyzer.py**: 품질 분석 + 자동 분류
3. **실행**: `python preprocess_with_analyzer.py`
4. **결과**: OTE/Velum/None으로 자동 분류된 프레임들
5. **다음**: `python train.py`로 모델 학습

이제 업로드하신 이미지처럼 명확한 프레임은 OTE/Velum으로, 밝거나 흐린 프레임은 None으로 자동 분류됩니다! 🎉
