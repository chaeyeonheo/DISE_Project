# Smart Video Preprocessing 가이드

## 문제 정의

### 실제 상황
- **None의 의미**: OTE/Velum이 아닌 **모든 영역**
  - 내시경 **삽입 시** 노이즈 (아직 목표 영역 도달 전)
  - 내시경 **제거 시** 노이즈 (목표 영역 벗어난 후)
  - **전환 구간** (OTE↔Velum 사이, 불명확한 순간)
  - 기타 모든 비정상 프레임

### 기존 방식의 문제
```
문제 1: 바로 OTE/Velum으로 시작하는 영상
├─ 앞 3초 = 이미 OTE/Velum 포함
└─ None으로 레이블링 → 잘못된 학습

문제 2: 고정된 시간 기준
├─ 모든 영상이 다른 패턴
├─ 어떤 영상: 긴 삽입 과정 (10초+)
└─ 어떤 영상: 바로 시작 (0초)

문제 3: 전환 구간 무시
├─ OTE→Velum, Velum→OTE 전환 순간
└─ 이런 애매한 프레임들을 None으로 학습해야 함
```

---

## 해결책: Smart Detection

### 핵심 아이디어

**자동으로 "실제 검사 구간"을 감지하여 그 외는 모두 None으로 처리**

```python
# 1. 비디오 품질 분석
품질 점수 = 0.6 × 밝기 + 0.4 × 선명도

# 2. 검사 구간 자동 감지
if 품질 점수 > 임계값:
    → 검사 중 (OTE/Velum)
else:
    → 검사 전/후 (None)

# 3. 레이블링
검사 전 (0 ~ 검사시작) → None (삽입 노이즈)
검사 중 (검사시작 ~ 검사종료) → OTE or Velum
검사 후 (검사종료 ~ 끝) → None (제거 노이즈)
```

### 작동 원리

#### 1단계: 품질 프로파일 분석

```python
for each frame:
    # 밝기 계산
    brightness = mean(grayscale_image)
    
    # 선명도 계산 (Laplacian variance)
    sharpness = variance(laplacian(image))
    
    # 품질 점수
    quality = 0.6 * (brightness/255) + 0.4 * (sharpness/1000)
```

**의료 내시경 특징**:
- 삽입 중: 어둡고 흐릿함 (quality < 0.5)
- 검사 중: 밝고 선명함 (quality > 0.5)
- 제거 중: 다시 어두워짐 (quality < 0.5)

#### 2단계: 검사 구간 감지

```python
# 임계값 이상인 구간 찾기
above_threshold = [t for t, q in quality_scores if q > 0.5]

examination_start = first(above_threshold)
examination_end = last(above_threshold)
```

#### 3단계: 시각화 저장

각 비디오마다 품질 프로파일 그래프 생성:
```
processed_dataset/
└── visualizations/
    ├── video1_quality.png
    ├── video2_quality.png
    └── ...
```

이를 통해 자동 감지가 올바른지 확인 가능!

---

## 사용 방법

### 기본 사용 (자동 감지 + 시각화)

```bash
python preprocess_videos_smart.py
```

### 수동 모드 (기존 방식)

```bash
python preprocess_videos_smart.py --no-smart-detection
```

### 시각화 없이 빠르게

```bash
python preprocess_videos_smart.py --no-visualizations
```

---

## 레이블링 전략

### Velum 비디오

```
┌─────────────────────────────────────────────────┐
│  0초          exam_start        exam_end    끝  │
│   │               │                 │          │ │
│   ▼───None────▼────Velum────▼───None───▼      │
│  삽입         검사 시작      검사 종료    제거    │
└─────────────────────────────────────────────────┘

추출:
1. None (삽입): 0 ~ exam_start
2. Velum: exam_start ~ exam_end
3. None (제거): exam_end ~ 끝
4. None (전환): exam_start 직후 + exam_end 직전
```

### OTE 비디오

```
┌────────────────────────────────────────────────────────┐
│  0초    exam_start  velum_end    ote_end         끝   │
│   │         │          │            │              │   │
│   ▼─None──▼──Velum──▼────OTE────▼────None────▼      │
│  삽입    검사시작  OTE시작    OTE종료         제거      │
└────────────────────────────────────────────────────────┘

추출:
1. None (삽입): 0 ~ exam_start
2. None (Velum 전환): exam_start 직후, velum_end 직전
3. OTE: velum_end ~ ote_end
4. None (제거): ote_end ~ 끝
```

---

## 장점

### ✅ 1. 영상별 적응형 처리

```python
# 영상 A: 긴 삽입 과정
0초────────10초──────────────30초────35초
│   None    │      Velum      │ None │
           ↑ 자동 감지됨

# 영상 B: 바로 시작
0초──────────────────20초────23초
│      Velum         │ None │
↑ 삽입 구간 없음을 자동 인식
```

### ✅ 2. None 데이터의 의미적 정확성

```
기존 방식:
- 앞 3초 = 무조건 None (잘못될 수 있음)

Smart 방식:
- 실제로 검사 전/후인 구간만 None
- 전환 구간도 포함
- 더 정확한 "OTE/Velum 아님" 학습
```

### ✅ 3. 시각적 확인 가능

각 비디오의 품질 프로파일을 확인하여:
- 자동 감지가 올바른지 검증
- 문제 있는 영상 발견
- 임계값 조정 가능

---

## 예상 결과

### None 프레임 분포

```
Total None frames: 1,234
  - Insertion (삽입 노이즈): 456 (37%)
  - Examination (전환 구간): 321 (26%)
  - Removal (제거 노이즈): 457 (37%)
```

### 품질 프로파일 예시

```
Quality Score
1.0 ┤                 ┌────────────┐
    │                 │    Velum   │
0.8 ┤                 │            │
    │                 │            │
0.6 ┤             ┌───┤            ├───┐
0.4 │         ┌───┘   │            │   └───┐
    │     ┌───┘       │            │       └───┐
0.2 │ ┌───┘           │            │           └───┐
    │ │ Insertion     │            │    Removal    │
0.0 └─┴───────────────┴────────────┴───────────────┴──→ Time
    0s              10s           30s             35s
         │                                 │
    exam_start                        exam_end
```

---

## 추가 개선 방안

### 1. 수동 검증 후 재학습

```python
# 1단계: 자동 감지로 1차 전처리
preprocessor.run(use_smart_detection=True)

# 2단계: visualizations 폴더의 그래프 확인
# 잘못 감지된 영상이 있으면 수동으로 보정

# 3단계: 보정된 정보로 재전처리
```

### 2. 임계값 조정

```python
# preprocess_videos_smart.py 내부 수정
def analyze_brightness_profile(...):
    threshold = 0.5  # 기본값
    # 너무 많이 None으로 분류되면 → 낮춤 (0.4)
    # 너무 적게 None으로 분류되면 → 높임 (0.6)
```

### 3. 앙상블 검증

```python
# 여러 방법으로 추출 후 교집합 사용
none_auto = extract_with_smart_detection()
none_manual = extract_with_fixed_time()
none_final = validate_both(none_auto, none_manual)
```

### 4. 별도 None 영상 수집 (최선)

```python
dataset/
├── OTE/
├── Velum/
└── None/  # 새로 추가
    ├── insertion_noise_1.mp4
    ├── removal_noise_2.mp4
    └── transition_3.mp4
```

**가장 확실한 방법**: 
- 내시경 삽입/제거 시 영상만 별도 촬영
- 전환 구간만 수동으로 편집
- 100% 정확한 None 데이터 확보

---

## 권장 워크플로우

### Phase 1: 자동 감지 테스트

```bash
# 모든 영상에 대해 자동 감지 실행
python preprocess_videos_smart.py

# visualizations 폴더의 그래프들 확인
# 이상한 패턴이 있는지 체크
```

### Phase 2: 수동 검증

```python
# 각 비디오의 품질 프로파일 확인
for graph in visualizations/*.png:
    if looks_wrong:
        mark_for_manual_adjustment()
```

### Phase 3: 학습 및 평가

```bash
# 학습
python train.py

# 결과 분석
# - None 클래스의 precision/recall 확인
# - Confusion matrix에서 None 오분류 패턴 확인
```

### Phase 4: 반복 개선

```python
if None class performance is low:
    # 방법 1: 임계값 조정
    adjust_quality_threshold()
    
    # 방법 2: None 데이터 추가 수집
    collect_more_none_videos()
    
    # 방법 3: Data augmentation
    augment_none_frames()
```

---

## 결론

### 핵심 개선 사항

1. **적응형 처리**: 영상마다 다른 시작/종료 시점 자동 감지
2. **의미적 정확성**: 실제로 OTE/Velum이 아닌 구간만 None으로
3. **시각적 검증**: 품질 프로파일로 자동 감지 결과 확인 가능
4. **다양한 None 타입**: 삽입/전환/제거 노이즈 모두 포함

### 기대 효과

- ✅ None 클래스의 정확도 향상
- ✅ OTE/Velum 오분류 감소
- ✅ 더 robust한 모델 학습
- ✅ 실제 임상 환경에서도 잘 동작

**가장 중요한 것**: None = "OTE/Velum이 아닌 모든 것"을 정확히 학습!
