# DISE 통합 분석 시스템 가이드

## 🎯 전체 파이프라인

```
비디오 입력 (.mp4)
    ↓
[1] OTE/Velum/None 프레임별 분류
    ↓
[2] OTE/Velum 구간에서만 ROI 면적 분석
    ↓
[3] 연속 구간 감지 (몇 초~몇 초)
    ↓
[4] 폐색 이벤트 감지 (30% 이상 감소)
    ↓
[5] 위험 구간 비디오 클립 생성
    ↓
[6] 보고서 생성 & 웹 표시
```

---

## 📁 파일 구조

```
real_dise_cy/
├── video_analyzer.py              # 기존 (폐색 분석)
├── app.py                         # 기존 웹앱
├── report_generator.py            # 기존 보고서
│
├── integrated_analyzer.py         # ⭐ 새로운 통합 분석기
├── integrated_app.py              # ⭐ 새로운 통합 웹앱
├── integrated_report_generator.py # ⭐ 새로운 통합 보고서
│
└── ote_velum_classification_final/
    ├── preprocess_with_analyzer.py
    ├── train.py
    ├── model.py
    ├── dataset.py
    └── checkpoints/
        └── best_model.pth         # 학습된 분류 모델
```

---

## 🚀 사용 방법

### 1단계: Classification 모델 학습

```bash
cd ote_velum_classification_final

# 전처리 (원하는 방법 선택)
python preprocess_with_analyzer.py

# 학습
python train.py

# 결과: checkpoints/best_model.pth 생성됨
```

### 2단계: 통합 분석 실행

#### 방법 A: Python 스크립트로 직접 실행

```python
from integrated_analyzer import IntegratedDISEAnalyzer

# 분석기 생성
analyzer = IntegratedDISEAnalyzer(
    model_path='ote_velum_classification_final/checkpoints/best_model.pth',
    fps_extract=5,              # 초당 5프레임 분석
    threshold_percent=30,        # 30% 이상 감소 시 이벤트
    min_event_duration=1.0       # 최소 1초 이상 지속
)

# 비디오 분석
results = analyzer.analyze_video(
    video_path='test_video.mp4',
    output_dir='output'
)

# 결과 확인
print(f"총 구간: {results['summary']['total_segments']}")
print(f"폐색 이벤트: {results['summary']['total_events']}")
```

#### 방법 B: 웹 인터페이스 사용

```bash
# 웹 서버 시작
python integrated_app.py

# 브라우저에서 http://localhost:5000 접속
# 비디오 업로드 → 분석 → 보고서 확인
```

---

## 📊 출력 결과

### analysis_results.json 구조

```json
{
  "video_info": {
    "filename": "test_video.mp4",
    "fps": 30.0,
    "duration": 20.0,
    "total_frames": 600
  },
  
  "frame_classifications": [
    {
      "frame_number": 120,
      "timestamp": 4.0,
      "label": "Velum",
      "confidence": 0.95,
      "roi_area": 45230
    }
  ],
  
  "segments": [
    {
      "label": "Velum",
      "start_frame": 120,
      "end_frame": 450,
      "start_time": 4.0,
      "end_time": 15.0,
      "duration": 11.0
    }
  ],
  
  "occlusion_events": [
    {
      "segment_label": "Velum",
      "severity": "Severe",
      "start_time": 7.2,
      "end_time": 9.5,
      "duration": 2.3,
      "max_reduction": 65.3,
      "clip_path": "output/event_clips/event_001_Velum_Severe.mp4"
    }
  ],
  
  "summary": {
    "total_segments": 3,
    "ote_segments": 1,
    "velum_segments": 2,
    "total_events": 2,
    "events_by_severity": {
      "Critical": 0,
      "Severe": 1,
      "Moderate": 1,
      "Mild": 0
    }
  }
}
```

### 생성되는 파일들

```
output/
├── analysis_results.json         # 전체 분석 결과
├── report.html                   # HTML 보고서
├── timeline.png                  # 타임라인 차트
├── severity_chart.png            # 심각도 차트
└── event_clips/                  # 위험 구간 비디오 클립
    ├── event_001_Velum_Severe.mp4
    ├── event_002_OTE_Moderate.mp4
    └── ...
```

---

## 🔑 핵심 기능

### 1. 프레임별 분류
- **입력**: 전체 프레임 (검정 배경 제거됨)
- **출력**: OTE / Velum / None
- **방법**: Classification 모델 추론

### 2. 연속 구간 감지
```python
# 예시
프레임 1-50: None
프레임 51-200: Velum    ← 구간 1 (5.0초 지속)
프레임 201-250: None
프레임 251-400: OTE     ← 구간 2 (5.0초 지속)
```

### 3. 폐색 분석 (OTE/Velum 구간만)
```python
# Velum 구간 (프레임 51-200)
최대 ROI 면적: 50,000 px²

프레임 120: ROI = 35,000 px² (30% 감소)  ← 이벤트 시작
프레임 130: ROI = 25,000 px² (50% 감소)
프레임 140: ROI = 15,000 px² (70% 감소)  ← Critical!
프레임 150: ROI = 40,000 px² (20% 감소)  ← 이벤트 종료

→ 이벤트: 4.0초 ~ 5.0초 (지속 1.0초, 최대 70% 감소, Critical)
```

### 4. 비디오 클립 생성
- 각 이벤트마다 독립된 mp4 클립 생성
- 전후 0.5초 여유 포함
- 빨간 테두리 + 텍스트 오버레이

### 5. 웹 인터페이스
- 비디오 업로드
- 실시간 분석 진행 상황
- 인터랙티브 보고서
- 이벤트 클립 재생

---

## ⚙️ 설정 조정

### 분석 민감도 조정

```python
analyzer = IntegratedDISEAnalyzer(
    model_path='checkpoints/best_model.pth',
    
    fps_extract=10,              # 5 → 10: 더 세밀한 분석
    
    threshold_percent=20,         # 30 → 20: 더 민감하게 (더 많은 이벤트)
    
    min_event_duration=0.5,       # 1.0 → 0.5: 짧은 이벤트도 포착
    
    exclude_first_seconds=3,      # 2 → 3: 앞부분 더 많이 제외
    exclude_last_seconds=5        # 3 → 5: 뒷부분 더 많이 제외
)
```

### Classification 신뢰도 조정

현재는 모든 프레임을 분류하지만, 낮은 신뢰도 프레임을 None으로 처리하려면:

```python
# integrated_analyzer.py의 classify_frame 메서드 수정

def classify_frame(self, frame):
    # ... 기존 코드 ...
    
    # 신뢰도 임계값 추가
    if confidence < 0.7:  # 70% 미만은 None으로
        return 'None', confidence
    
    return self.class_names[pred_class], confidence
```

---

## 🐛 문제 해결

### 문제 1: 모든 프레임이 None으로 분류

**원인**: Classification 모델 경로 오류

**해결**:
```bash
# 모델 경로 확인
ls ote_velum_classification_final/checkpoints/best_model.pth

# 경로 수정
analyzer = IntegratedDISEAnalyzer(
    model_path='정확한/경로/best_model.pth'
)
```

### 문제 2: 이벤트가 너무 많이 감지됨

**원인**: 임계값이 너무 낮음

**해결**:
```python
threshold_percent=40,        # 30 → 40
min_event_duration=2.0       # 1.0 → 2.0
```

### 문제 3: 비디오 클립 생성 실패

**원인**: OpenCV 코덱 문제

**해결**:
```bash
# ffmpeg 설치
conda install ffmpeg

# 또는
apt-get install ffmpeg
```

### 문제 4: GPU 메모리 부족

**해결**:
```python
# CPU 사용
device = torch.device('cpu')

# 또는 배치 처리 수 줄이기
fps_extract=3  # 5 → 3
```

---

## 📈 성능 최적화

### 빠른 분석

```python
analyzer = IntegratedDISEAnalyzer(
    fps_extract=3,               # 적게 추출
    min_event_duration=2.0,      # 긴 이벤트만
    threshold_percent=40         # 덜 민감하게
)
```

### 정밀 분석

```python
analyzer = IntegratedDISEAnalyzer(
    fps_extract=10,              # 많이 추출
    min_event_duration=0.5,      # 짧은 이벤트도
    threshold_percent=20         # 민감하게
)
```

---

## 🎓 고급 기능

### 1. 커스텀 ROI 검출

`integrated_analyzer.py`의 `extract_roi_area` 메서드 수정:

```python
def extract_roi_area(self, frame):
    # 색상 범위 조정
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])  # 80 → 100
    
    # ... 나머지 코드
```

### 2. 심각도 기준 변경

`_classify_severity` 메서드 수정:

```python
def _classify_severity(self, reduction_percent):
    if reduction_percent >= 80:      # 70 → 80
        return 'Critical'
    elif reduction_percent >= 60:    # 50 → 60
        return 'Severe'
    # ...
```

### 3. 다중 비디오 배치 처리

```python
video_files = ['video1.mp4', 'video2.mp4', 'video3.mp4']

for video_file in video_files:
    print(f"\n처리 중: {video_file}")
    
    results = analyzer.analyze_video(
        video_path=video_file,
        output_dir=f'output/{Path(video_file).stem}'
    )
    
    print(f"완료: {results['summary']['total_events']}개 이벤트")
```

---

## 📝 보고서 커스터마이징

### HTML 템플릿 수정

`integrated_report_generator.py`의 `generate_html_report` 메서드에서:

- CSS 스타일 변경
- 차트 종류 추가
- 추가 정보 표시

### 차트 커스터마이징

```python
# timeline 차트 색상 변경
colors = {
    'OTE': '#ff6b6b',      # 빨강
    'Velum': '#4ecdc4',    # 청록색
    'None': '#95a5a6'      # 회색
}
```

---

## ✅ 체크리스트

### 초기 설정
- [ ] Classification 모델 학습 완료
- [ ] best_model.pth 생성 확인
- [ ] 필요 패키지 설치 (`torch`, `cv2`, `matplotlib` 등)

### 분석 전
- [ ] 비디오 파일 준비
- [ ] 출력 폴더 권한 확인
- [ ] 모델 경로 확인

### 분석 후
- [ ] analysis_results.json 생성 확인
- [ ] event_clips/ 폴더 확인
- [ ] report.html 열어서 확인
- [ ] 비디오 클립 재생 테스트

---

## 🆘 지원

문제가 발생하면:
1. 로그 메시지 확인
2. analysis_results.json 내용 확인
3. 모델 경로 및 권한 확인
4. GPU/CPU 메모리 확인

---

**🎉 이제 통합 분석 시스템을 사용할 수 있습니다!**

```bash
python integrated_app.py
# → http://localhost:5000 접속
```
