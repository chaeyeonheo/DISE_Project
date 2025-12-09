# Gunicorn 설정 가이드: 타임아웃 및 메모리 최적화

## 문제: WORKER TIMEOUT 및 메모리 부족

### 증상
- `WORKER TIMEOUT (pid:66)`
- `Worker was sent SIGKILL! Perhaps out of memory?`
- 502 Bad Gateway 에러

### 원인
1. **타임아웃**: Gunicorn 기본 타임아웃은 30초인데, 비디오 분석은 수분~수십 분 걸림
2. **메모리**: 모델 로드 및 비디오 처리로 인한 메모리 사용량 증가
3. **워커 수**: 너무 많은 워커가 메모리를 과도하게 사용

## 해결 방법

### 1. Gunicorn 타임아웃 증가

**render.yaml 수정:**
```yaml
startCommand: gunicorn -w 2 -b 0.0.0.0:10000 --timeout 1200 --max-requests 1000 integrated_app:app
```

**옵션 설명:**
- `-w 2`: 워커 수 2개 (메모리 절약)
- `--timeout 1200`: 타임아웃 20분 (1200초)
- `--max-requests 1000`: 워커당 최대 요청 수

### 2. Render 대시보드에서 직접 설정

Render 대시보드 → 서비스 → Settings → Start Command:
```
gunicorn -w 2 -b 0.0.0.0:10000 --timeout 1200 --max-requests 1000 integrated_app:app
```

### 3. 메모리 최적화

**워커 수 조정:**
- 무료 플랜: `-w 1` (단일 워커)
- 유료 플랜: `-w 2` (2개 워커)

**모델 로드 최적화:**
- 모델을 앱 시작 시 한 번만 로드 (현재는 요청마다 로드)
- 또는 모델 캐싱 사용

## 권장 설정

### 무료 플랜
```bash
gunicorn -w 1 -b 0.0.0.0:10000 --timeout 1200 integrated_app:app
```

### 유료 플랜 (Starter 이상)
```bash
gunicorn -w 2 -b 0.0.0.0:10000 --timeout 1200 --max-requests 1000 integrated_app:app
```

## 확인 방법

배포 후 로그에서 확인:
```
[INFO] Starting gunicorn 21.2.0
[INFO] Listening at: http://0.0.0.0:10000
[INFO] Using worker: sync
[INFO] Booting worker with pid: XX
```

워커 타임아웃이 발생하지 않으면 성공입니다!

