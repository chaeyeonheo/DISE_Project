# 웹 배포 가이드 (Render / PythonAnywhere)

Flask 애플리케이션을 무료 웹 호스팅 서비스에 배포하는 방법을 설명합니다.

---

## 📋 목차

1. [배포 전 준비사항](#1-배포-전-준비사항)
2. [Render 배포 (추천)](#2-render-배포-추천)
3. [PythonAnywhere 배포](#3-pythonanywhere-배포)
4. [배포 후 확인사항](#4-배포-후-확인사항)
5. [문제 해결](#5-문제-해결)

---

## 1. 배포 전 준비사항

### 1.1 필수 체크리스트

✅ **코드 준비**
- [ ] `requirements.txt`에 `gunicorn` 포함 확인
- [ ] `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
- [ ] API 키가 코드에 하드코딩되어 있지 않은지 확인
- [ ] GitHub 저장소에 코드 업로드 완료

✅ **파일 확인**
- [ ] `integrated_app.py`가 메인 애플리케이션 파일
- [ ] `ote_velum_classification_final/checkpoints/best_model.pth` 모델 파일 존재
- [ ] `templates/index.html` 템플릿 파일 존재

### 1.2 GitHub에 코드 업로드

```bash
# Git 초기화 (아직 안 했다면)
git init

# .gitignore 확인 (중요!)
# .env, outputs/, uploads/ 등이 무시되는지 확인

# 파일 추가
git add .

# 커밋
git commit -m "Initial commit for deployment"

# GitHub에 푸시
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

> ⚠️ **중요**: `.env` 파일은 절대 GitHub에 올리지 마세요! API 키가 노출되면 해킹봇이 즉시 사용합니다.

---

## 2. Render 배포 (추천)

Render는 Heroku의 정신적 후속작으로, Flask 배포가 가장 쉽고 직관적입니다.

### 2.1 Render 가입 및 설정

1. **Render 가입**
   - [https://render.com](https://render.com) 접속
   - GitHub 계정으로 가입 (권장)

2. **새 Web Service 생성**
   - 대시보드에서 "New +" 클릭
   - "Web Service" 선택
   - GitHub 저장소 연결 및 선택

### 2.2 배포 설정

**기본 설정:**
- **Name**: `dise-analyzer` (원하는 이름)
- **Environment**: `Python 3`
- **Region**: `Singapore` (또는 가장 가까운 지역)
- **Branch**: `main` (또는 기본 브랜치)

**빌드 및 시작 명령어:**

```bash
# Build Command
pip install -r requirements.txt

# Start Command (중요!)
gunicorn integrated_app:app
```

> 💡 **설명**: `integrated_app:app`은 `integrated_app.py` 파일의 `app` 객체를 의미합니다.

### 2.3 환경 변수 설정

Render 대시보드에서 "Environment" 섹션으로 이동하여 다음 환경 변수를 추가:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | `your_gemini_api_key_here` |
| `PYTHONUNBUFFERED` | `1` |

> 🔒 **보안**: 환경 변수는 대시보드에서만 설정하고, 코드에는 절대 넣지 마세요.

### 2.4 고급 설정 (선택사항)

**리소스 설정:**
- **Free Tier**: 512MB RAM, 0.5 CPU
- **Starter ($7/월)**: 512MB RAM, 0.5 CPU (항상 켜져있음)
- **Standard ($25/월)**: 2GB RAM, 1 CPU

> ⚠️ **무료 플랜 제한**: 15분간 접속이 없으면 서비스가 잠듭니다 (Spin-down). 다시 깨어날 때 30초~1분 정도 걸립니다.

**타임아웃 설정:**
- 비디오 분석은 시간이 오래 걸릴 수 있으므로, "Advanced" 섹션에서:
  - **Health Check Path**: `/` (또는 비워두기)
  - **Auto-Deploy**: `Yes` (GitHub 푸시 시 자동 배포)

### 2.5 배포 완료

1. "Create Web Service" 클릭
2. 배포 로그 확인 (빌드 및 시작 과정 모니터링)
3. 배포 완료 후 제공되는 URL로 접속 (예: `https://dise-analyzer.onrender.com`)

---

## 3. PythonAnywhere 배포

PythonAnywhere는 파이썬 전용 호스팅 서비스로, 서버가 잠들지 않습니다.

### 3.1 PythonAnywhere 가입

1. [https://www.pythonanywhere.com](https://www.pythonanywhere.com) 접속
2. "Beginner" 무료 계정 가입

### 3.2 코드 업로드

**방법 1: GitHub에서 클론 (권장)**

```bash
# PythonAnywhere의 Bash 콘솔에서
cd ~
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

**방법 2: 파일 업로드**
- Files 탭에서 직접 업로드

### 3.3 가상 환경 설정

```bash
# PythonAnywhere Bash 콘솔에서
cd ~/your-repo
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.4 Web App 설정

1. **Web 탭으로 이동**
2. "Add a new web app" 클릭
3. **Python 3.10** 선택
4. **Manual configuration** 선택
5. **WSGI configuration file** 편집:

```python
# /var/www/yourusername_pythonanywhere_com_wsgi.py

import sys
import os

# 프로젝트 경로 추가
path = '/home/yourusername/your-repo'
if path not in sys.path:
    sys.path.insert(0, path)

# 가상 환경 활성화
activate_this = '/home/yourusername/your-repo/venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

# Flask 앱 import
from integrated_app import app as application

# 환경 변수 설정
os.environ['GEMINI_API_KEY'] = 'your_gemini_api_key_here'
```

> ⚠️ **주의**: PythonAnywhere 무료 플랜은 **외부 API 호출에 제한**이 있을 수 있습니다. Gemini API는 일반적으로 작동하지만, 테스트가 필요합니다.

### 3.5 정적 파일 설정

**Static files** 섹션:
- URL: `/static/`
- Directory: `/home/yourusername/your-repo/static/`

**Static files** (outputs):
- URL: `/outputs/`
- Directory: `/home/yourusername/your-repo/outputs/`

### 3.6 디렉토리 생성

```bash
# Bash 콘솔에서
cd ~/your-repo
mkdir -p uploads outputs
chmod 755 uploads outputs
```

### 3.7 Web App 재로드

Web 탭에서 "Reload" 버튼 클릭

---

## 4. 배포 후 확인사항

### 4.1 기본 동작 확인

1. **홈페이지 접속**: 메인 페이지가 정상적으로 로드되는지 확인
2. **API 엔드포인트 확인**: `/api/analyze` 엔드포인트 존재 확인
3. **환경 변수 확인**: 로그에서 `GEMINI_API_KEY` 로드 여부 확인

### 4.2 파일 업로드 테스트

1. 작은 테스트 비디오 파일 업로드
2. 분석이 정상적으로 시작되는지 확인
3. 에러 로그 확인

### 4.3 로그 확인

**Render:**
- 대시보드의 "Logs" 탭에서 실시간 로그 확인

**PythonAnywhere:**
- Web 탭의 "Error log" 및 "Server log" 확인

---

## 5. 문제 해결

### 5.1 "Module not found" 오류

**원인**: 의존성 설치 실패

**해결**:
```bash
# requirements.txt 확인
# 모든 패키지가 올바른 버전으로 지정되어 있는지 확인

# 로컬에서 테스트
pip install -r requirements.txt
```

### 5.2 "GEMINI_API_KEY not found" 오류

**원인**: 환경 변수 미설정

**해결**:
- Render: Environment Variables 섹션에서 확인
- PythonAnywhere: WSGI 파일에서 `os.environ` 설정 확인

### 5.3 "Port already in use" 오류

**원인**: 포트 충돌 (일반적으로 발생하지 않음)

**해결**:
- Render/PythonAnywhere는 자동으로 포트를 할당하므로 문제 없음
- 로컬 테스트 시에만 발생 가능

### 5.4 "Request timeout" 오류

**원인**: 비디오 분석 시간이 너무 김

**해결**:
- Render: 무료 플랜은 30초 타임아웃이 있음 → 유료 플랜 고려
- PythonAnywhere: 타임아웃 설정 확인
- 대안: 비동기 처리 구현 (Celery 등)

### 5.5 모델 파일 누락 오류

**원인**: `best_model.pth` 파일이 GitHub에 없음

**해결**:
1. **방법 1**: 모델 파일을 GitHub에 추가 (용량이 크면 Git LFS 사용)
2. **방법 2**: 배포 시 모델 파일 다운로드 스크립트 추가

```python
# 배포 시 자동으로 모델 다운로드하는 스크립트 추가
# ote_velum_classification_final/download_model.py 실행
```

### 5.6 디스크 공간 부족

**원인**: `outputs/` 폴더에 결과가 계속 쌓임

**해결**:
- 정기적으로 오래된 결과 삭제하는 스크립트 추가
- 또는 외부 스토리지 (S3 등) 사용

### 5.7 외부 API 호출 제한 (PythonAnywhere)

**증상**: Gemini API 호출 실패

**해결**:
- PythonAnywhere 지원팀에 문의하여 특정 도메인 화이트리스트 요청
- 또는 Render 사용 (외부 API 제한 없음)

---

## 6. 배포 후 최적화

### 6.1 성능 최적화

- **이미지 최적화**: 업로드된 비디오 해상도 자동 조정
- **캐싱**: 정적 파일 캐싱 설정
- **CDN 사용**: 정적 파일을 CDN으로 서빙 (선택사항)

### 6.2 모니터링 설정

- **에러 추적**: Sentry 등 에러 추적 서비스 연동
- **성능 모니터링**: 응답 시간, 메모리 사용량 모니터링

### 6.3 백업

- **정기 백업**: 중요한 분석 결과는 외부 스토리지에 백업
- **코드 백업**: GitHub에 정기적으로 푸시

---

## 7. 무료 vs 유료 플랜 비교

### Render

| 기능 | Free | Starter ($7/월) | Standard ($25/월) |
|------|------|-----------------|-------------------|
| RAM | 512MB | 512MB | 2GB |
| CPU | 0.5 | 0.5 | 1 |
| 항상 켜져있음 | ❌ | ✅ | ✅ |
| 타임아웃 | 30초 | 30초 | 600초 |
| 외부 API | ✅ | ✅ | ✅ |

### PythonAnywhere

| 기능 | Beginner (Free) | Hacker ($5/월) | Web Dev ($12/월) |
|------|----------------|----------------|------------------|
| 항상 켜져있음 | ✅ | ✅ | ✅ |
| 외부 API | ⚠️ 제한적 | ✅ | ✅ |
| 디스크 | 512MB | 3GB | 5GB |
| CPU 시간 | 제한적 | 무제한 | 무제한 |

---

## 8. 추천 배포 전략

### 개인 프로젝트 / 포트폴리오
→ **Render Free Tier** (간단하고 빠름)

### 소규모 프로덕션
→ **Render Starter** ($7/월) 또는 **PythonAnywhere Hacker** ($5/월)

### 중규모 프로덕션
→ **Render Standard** ($25/월) 또는 **AWS/Azure/GCP**

---

## 📞 추가 지원

배포 관련 문제가 발생하면:

1. **로그 확인**: 에러 메시지 자세히 읽기
2. **로컬 테스트**: 로컬에서 `gunicorn integrated_app:app` 실행하여 테스트
3. **문서 참고**: Render/PythonAnywhere 공식 문서 확인
4. **커뮤니티**: Stack Overflow, Reddit 등에서 유사한 문제 검색

---

**마지막 업데이트**: 2025-01-XX

**추천**: 처음 배포하시는 분은 **Render**를 강력 추천합니다! 🚀

