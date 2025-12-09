# Render 배포 문제 해결: Matplotlib 폰트 캐시 빌드

## 문제: 앱 시작 시 Matplotlib 폰트 캐시 빌드로 인한 지연

### 증상
- 배포 후 "Matplotlib is building the font cache" 메시지 후 멈춤
- 5분 이상 기다려도 진행되지 않음
- 앱이 시작되지 않음

### 원인
1. Matplotlib가 앱 시작 시 폰트 캐시를 빌드
2. Render 환경에서 폰트 시스템 접근이 느림
3. 첫 배포 시 시간이 오래 걸림

### 해결 방법

#### 방법 1: Matplotlib 지연 로딩 (적용됨)

`integrated_report_generator.py`에서 Matplotlib를 필요할 때만 import:

```python
# 앱 시작 시 import하지 않음
def _import_matplotlib():
    """Matplotlib를 필요할 때만 import"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    return plt, ticker

# 차트 생성 함수에서만 사용
def generate_timeline_chart(self, output_dir):
    plt, _ = _import_matplotlib()  # 여기서만 import
    # ... 차트 생성 코드
```

#### 방법 2: 환경 변수 설정

Render 대시보드에서 환경 변수 추가:

| Key | Value |
|-----|-------|
| `MPLCONFIGDIR` | `/tmp/matplotlib` |

또는 코드에서 설정:

```python
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
```

#### 방법 3: 빌드 시 폰트 캐시 미리 생성

`render.yaml` 또는 Build Command에 추가:

```yaml
buildCommand: |
  pip install -r requirements.txt &&
  python download_model_from_drive.py &&
  python -c "import matplotlib; matplotlib.font_manager._rebuild()"
```

### 확인 방법

배포 후 로그에서 다음을 확인:

✅ **성공:**
```
==> Running 'gunicorn integrated_app:app'
[INFO] Starting gunicorn 21.2.0
[INFO] Listening at: http://0.0.0.0:10000
```

❌ **실패 (여전히 멈춤):**
```
Matplotlib is building the font cache; this may take a moment.
(5분 이상 진행 없음)
```

### 추가 팁

1. **첫 배포 후**: 폰트 캐시가 생성되면 이후 배포는 빠름
2. **타임아웃 설정**: Render에서 타임아웃을 늘릴 수 있음 (무료 플랜 제한 있음)
3. **로그 모니터링**: 실시간 로그를 확인하여 진행 상황 파악

---

## 적용된 수정 사항

1. ✅ `integrated_report_generator.py`: Matplotlib 지연 로딩 구현
2. ✅ `integrated_app.py`: MPLCONFIGDIR 환경 변수 설정

이제 앱 시작 시 Matplotlib 폰트 캐시 빌드가 발생하지 않습니다!

