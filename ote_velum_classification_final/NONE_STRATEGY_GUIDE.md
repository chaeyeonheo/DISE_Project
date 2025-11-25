# None 레이블 추출 전략 가이드

## 문제점 분석

현재 방식(앞뒤 3초만 사용)의 문제:

1. **데이터 다양성 부족**
   - None 상태의 다양한 변형을 학습하기 어려움
   - 특정 위치/시간대의 특징만 학습할 위험

2. **전환 구간 포함 가능성**
   - 앞뒤 3초에 OTE/Velum으로 전환되는 순간이 포함될 수 있음
   - 레이블 노이즈 발생 가능

3. **클래스 불균형**
   - OTE/Velum 프레임에 비해 None 프레임이 상대적으로 적을 수 있음

---

## 개선된 전략들

### 전략 1: edges_only (기존 방식)

**설명**: 비디오의 앞뒤 3초만 사용

```python
preprocessor.run(none_strategy='edges_only')
```

**장점**:
- ✅ 빠른 처리 속도
- ✅ 구현이 간단
- ✅ 확실히 None인 구간 (검사 시작 전/후)

**단점**:
- ❌ 데이터 양이 적음
- ❌ 다양성 부족
- ❌ 전환 구간 포함 가능

**권장 상황**: 
- 빠른 프로토타이핑
- 데이터가 매우 많을 때

---

### 전략 2: static_regions (정적 구간 분석)

**설명**: 프레임 간 차이를 분석하여 움직임이 적은 구간을 None으로 분류

```python
preprocessor.run(none_strategy='static_regions')
```

**동작 방식**:
```python
# 프레임 간 차이 계산
for each frame:
    diff = |current_frame - previous_frame|
    if diff < threshold:
        mark as static region

# 2초 이상 정적인 구간을 None으로 추출
```

**장점**:
- ✅ 실제로 움직임이 없는 구간 탐지
- ✅ 데이터 품질이 높음
- ✅ OTE/Velum 구간 회피

**단점**:
- ❌ 처리 속도가 느림 (모든 프레임 분석)
- ❌ 정적 구간이 없을 수 있음
- ❌ 매개변수 튜닝 필요 (threshold)

**권장 상황**:
- 고품질 None 데이터가 필요할 때
- 비디오에 정적 구간이 명확히 있을 때

---

### 전략 3: middle_section (중간 구간 샘플링)

**설명**: 비디오의 여러 위치에서 골고루 샘플링

```python
preprocessor.run(none_strategy='middle_section')
```

**Velum 비디오**:
- 앞 2초
- 중간 3초 (duration/2 ± 1.5초)
- 뒤 2초

**OTE 비디오**:
- 앞 2초
- Velum 구간(3~9초) 중 일부 (4~7초)
- 뒤 2초

**장점**:
- ✅ 다양한 시간대에서 샘플링
- ✅ 적당한 처리 속도
- ✅ 데이터 양 확보

**단점**:
- ❌ OTE/Velum 구간과 겹칠 위험
- ❌ 시간적 위치에 의존

**권장 상황**:
- 빠른 처리와 다양성의 균형
- 비디오 길이가 충분할 때

---

### 전략 4: combined (조합 전략) ⭐ 권장

**설명**: 여러 전략을 조합하여 최대한 다양하고 확실한 None 데이터 확보

```python
preprocessor.run(none_strategy='combined')  # 기본값
```

**Velum 비디오 추출**:
```
1. 앞 2초 (frame_interval=3)
2. 뒤 2초 (frame_interval=3)
3. Velum 구간(3~duration-3) 내부에서 N개 지점 샘플링
   - 각 지점에서 1초씩 추출 (frame_interval=5)
   - Velum 레이블 구간과 겹치지 않도록 확인
```

**OTE 비디오 추출**:
```
1. 앞 2초 (frame_interval=3)
2. Velum 구간(4~7초) 일부 (frame_interval=4)
3. 뒤 2초 (frame_interval=3)
4. OTE 구간(9~duration-3) 내부에서 N개 지점 샘플링
   - 각 지점에서 1초씩 추출 (frame_interval=6)
```

**장점**:
- ✅ 최대한 다양한 None 데이터
- ✅ 여러 시간대에서 균등 샘플링
- ✅ 적절한 데이터 양 확보
- ✅ 클래스 불균형 완화
- ✅ frame_interval로 데이터 양 조절

**단점**:
- ❌ 약간 복잡한 로직
- ❌ 처리 시간이 중간 정도

**권장 상황**:
- 대부분의 경우 (기본 선택)
- 균형잡힌 데이터셋 필요
- 프로덕션 환경

---

## 전략 비교표

| 전략 | 처리속도 | 데이터양 | 다양성 | 품질 | 추천도 |
|------|----------|----------|--------|------|--------|
| edges_only | ⚡⚡⚡ 빠름 | ⭐ 적음 | ⭐ 낮음 | ⭐⭐ 보통 | ⭐⭐ |
| static_regions | ⚡ 느림 | ⭐⭐ 보통 | ⭐⭐⭐ 높음 | ⭐⭐⭐ 높음 | ⭐⭐⭐ |
| middle_section | ⚡⚡ 보통 | ⭐⭐ 보통 | ⭐⭐ 보통 | ⭐⭐ 보통 | ⭐⭐⭐ |
| combined | ⚡⚡ 보통 | ⭐⭐⭐ 많음 | ⭐⭐⭐ 높음 | ⭐⭐⭐ 높음 | ⭐⭐⭐⭐⭐ |

---

## 사용 방법

### 명령줄에서

```bash
# Combined 전략 (기본, 권장)
python preprocess_videos_improved.py --strategy combined

# Edges only (빠른 테스트)
python preprocess_videos_improved.py --strategy edges_only

# Static regions (고품질)
python preprocess_videos_improved.py --strategy static_regions

# Middle section (균형)
python preprocess_videos_improved.py --strategy middle_section
```

### Python 코드에서

```python
from preprocess_videos_improved import ImprovedVideoPreprocessor

preprocessor = ImprovedVideoPreprocessor(
    dataset_path='dataset',
    output_path='processed_dataset'
)

# 원하는 전략 선택
annotations = preprocessor.run(none_strategy='combined')

# 결과 확인
print(f"Total annotations: {len(annotations)}")
```

---

## 추가 개선 아이디어

### 1. 수동 레이블링 보조

```python
# 비디오를 재생하면서 None 구간을 수동으로 표시
# 추출한 프레임을 검토하여 잘못된 레이블 제거
```

### 2. 앙상블 전략

```python
# 여러 전략으로 추출한 후 중복 제거
none_frames_1 = process_with_strategy('edges_only')
none_frames_2 = process_with_strategy('middle_section')
none_frames = combine_and_deduplicate(none_frames_1, none_frames_2)
```

### 3. 별도 None 비디오 수집

```python
# 가장 확실한 방법: None 상태만 포함된 비디오를 별도로 촬영/수집
dataset/
  ├── OTE/
  ├── Velum/
  └── None/  # 새로 추가
      └── *.mp4
```

### 4. Active Learning

```python
# 1차 학습 후 모델이 불확실한 프레임들을 재검토
# 모델의 confidence가 낮은 프레임을 수동 레이블링
```

---

## 권장 워크플로우

### 단계 1: 빠른 프로토타입 (edges_only)

```bash
python preprocess_videos_improved.py --strategy edges_only
python train.py  # 빠르게 베이스라인 확인
```

### 단계 2: 개선된 데이터셋 (combined)

```bash
python preprocess_videos_improved.py --strategy combined
python train.py  # 성능 비교
```

### 단계 3: 필요시 추가 분석 (static_regions)

```bash
python preprocess_videos_improved.py --strategy static_regions
# 기존 데이터와 병합하여 사용
```

### 단계 4: 결과 비교 및 선택

각 전략의 결과를 비교하여 최적의 방법 선택:
- Validation accuracy
- Class-wise F1 scores
- Confusion matrix 분석

---

## 결론

**권장 사항**: 
1. 🥇 **처음에는 `combined` 전략 사용** (가장 균형잡힌 선택)
2. 🥈 모델 성능이 부족하면 `static_regions` 추가 고려
3. 🥉 빠른 실험이 필요하면 `edges_only`로 시작

**핵심 원칙**:
- None 데이터는 다양한 시간대에서 수집
- OTE/Velum 레이블 구간과 최대한 겹치지 않도록
- frame_interval로 데이터 양 조절
- 클래스 불균형 모니터링
