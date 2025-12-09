# 배포 가이드 (Deployment Guide)

이 문서는 수면 내시경 기도 폐색 분석 시스템의 배포 절차를 설명합니다.

---

## 📋 목차

1. [배포 전 준비사항](#1-배포-전-준비사항)
2. [로컬 배포](#2-로컬-배포)
3. [프로덕션 서버 배포](#3-프로덕션-서버-배포)
4. [Docker를 이용한 배포](#4-docker를-이용한-배포)
5. [클라우드 배포](#5-클라우드-배포)
6. [보안 고려사항](#6-보안-고려사항)
7. [문제 해결](#7-문제-해결)

---

## 1. 배포 전 준비사항

### 1.1 필수 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows, Linux, macOS
- **메모리**: 최소 8GB RAM (권장: 16GB 이상)
- **디스크 공간**: 최소 10GB (모델 파일 및 출력 결과 저장용)
- **GPU**: 선택사항 (CPU만으로도 동작 가능하나, GPU 사용 시 처리 속도 향상)

### 1.2 필수 파일 확인

배포 전 다음 파일들이 존재하는지 확인하세요:

```
✓ ote_velum_classification_final/checkpoints/best_model.pth  # 모델 파일
✓ requirements.txt                                           # 의존성 목록
✓ integrated_app.py                                          # 메인 애플리케이션
✓ integrated_analyzer.py                                     # 분석 엔진
✓ integrated_report_generator.py                             # 보고서 생성기
✓ templates/index.html                                       # 웹 UI
```

### 1.3 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 다음 내용을 추가하세요:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Gemini API 키 발급 방법:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. API 키 생성
3. `.env` 파일에 추가

> ⚠️ **보안 주의**: `.env` 파일은 절대 Git에 커밋하지 마세요. `.gitignore`에 추가되어 있는지 확인하세요.

---

## 2. 로컬 배포

### 2.1 개발 환경 설정

```bash
# 1. 프로젝트 디렉토리로 이동
cd real_dise_cy

# 2. 가상 환경 생성 (권장)
python -m venv venv

# 3. 가상 환경 활성화
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. 의존성 설치
pip install -r requirements.txt

# 5. .env 파일 생성 및 API 키 설정
# (위의 1.3 절차 참고)

# 6. 서버 실행
python integrated_app.py
```

### 2.2 접속 확인

브라우저에서 `http://localhost:5000` 접속하여 웹 인터페이스가 정상적으로 표시되는지 확인하세요.

---

## 3. 프로덕션 서버 배포

### 3.1 Gunicorn을 이용한 배포 (Linux/macOS)

Flask 개발 서버는 프로덕션 환경에 적합하지 않습니다. Gunicorn을 사용하세요.

#### 3.1.1 Gunicorn 설치

```bash
pip install gunicorn
```

#### 3.1.2 Gunicorn 실행

```bash
# 기본 실행
gunicorn -w 4 -b 0.0.0.0:5000 integrated_app:app

# 더 많은 옵션 (권장)
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 1200 --max-requests 1000 integrated_app:app
```

**옵션 설명:**
- `-w 4`: 워커 프로세스 수 (CPU 코어 수에 맞게 조정)
- `-b 0.0.0.0:5000`: 바인딩 주소 및 포트
- `--timeout 1200`: 요청 타임아웃 (초) - 비디오 분석은 시간이 오래 걸릴 수 있음
- `--max-requests 1000`: 워커당 최대 요청 수

#### 3.1.3 systemd 서비스로 등록 (Linux)

`/etc/systemd/system/dise-analyzer.service` 파일 생성:

```ini
[Unit]
Description=DISE Analyzer Web Application
After=network.target

[Service]
User=your_username
Group=your_group
WorkingDirectory=/path/to/real_dise_cy
Environment="PATH=/path/to/real_dise_cy/venv/bin"
ExecStart=/path/to/real_dise_cy/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 1200 integrated_app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

서비스 시작:

```bash
sudo systemctl daemon-reload
sudo systemctl enable dise-analyzer
sudo systemctl start dise-analyzer
sudo systemctl status dise-analyzer
```

### 3.2 Nginx 리버스 프록시 설정

Nginx를 앞단에 두어 정적 파일 제공 및 로드 밸런싱을 수행합니다.

`/etc/nginx/sites-available/dise-analyzer` 파일 생성:

```nginx
server {
    listen 80;
    server_name your_domain.com;

    client_max_body_size 500M;  # 비디오 업로드용

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 1200s;  # 비디오 분석 시간 고려
    }

    location /outputs {
        alias /path/to/real_dise_cy/outputs;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /static {
        alias /path/to/real_dise_cy/static;
        expires 30d;
    }
}
```

설정 활성화:

```bash
sudo ln -s /etc/nginx/sites-available/dise-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 3.3 SSL/TLS 인증서 설정 (Let's Encrypt)

HTTPS를 위해 Let's Encrypt를 사용하세요:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

---

## 4. Docker를 이용한 배포

### 4.1 Dockerfile 생성

프로젝트 루트에 `Dockerfile` 생성:

```dockerfile
FROM python:3.10-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 디렉토리 생성
RUN mkdir -p uploads outputs

# 포트 노출
EXPOSE 5000

# 환경 변수 설정
ENV FLASK_APP=integrated_app.py
ENV PYTHONUNBUFFERED=1

# Gunicorn 실행
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "1200", "integrated_app:app"]
```

### 4.2 .dockerignore 생성

```dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.env
.venv
outputs/
uploads/
*.log
.git
.gitignore
README.md
docs/
```

### 4.3 Docker 이미지 빌드 및 실행

```bash
# 이미지 빌드
docker build -t dise-analyzer:latest .

# 컨테이너 실행
docker run -d \
  --name dise-analyzer \
  -p 5000:5000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/.env:/app/.env \
  --restart unless-stopped \
  dise-analyzer:latest

# 로그 확인
docker logs -f dise-analyzer
```

### 4.4 Docker Compose 사용 (권장)

`docker-compose.yml` 파일 생성:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./outputs:/app/outputs
      - ./uploads:/app/uploads
      - ./.env:/app/.env
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

실행:

```bash
docker-compose up -d
```

---

## 5. 클라우드 배포

### 5.1 AWS 배포

#### EC2 인스턴스

1. **EC2 인스턴스 생성**
   - 인스턴스 타입: t3.large 이상 (메모리 8GB 이상)
   - OS: Ubuntu 22.04 LTS
   - 보안 그룹: 포트 5000 (또는 80/443) 열기

2. **인스턴스 접속 및 설정**

```bash
# SSH 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 및 필수 패키지 설치
sudo apt install python3-pip python3-venv nginx git -y

# 프로젝트 클론 또는 업로드
git clone your-repo-url
# 또는 scp로 파일 전송

# 가상 환경 설정 및 의존성 설치
cd real_dise_cy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt gunicorn

# .env 파일 생성
nano .env
# GEMINI_API_KEY=your_key 입력

# Gunicorn 및 systemd 설정 (3.1.3 참고)
# Nginx 설정 (3.2 참고)
```

#### Elastic Beanstalk (더 간단한 방법)

1. EB CLI 설치

```bash
pip install awsebcli
```

2. EB 초기화

```bash
eb init -p python-3.10 dise-analyzer
eb create dise-analyzer-env
```

3. 환경 변수 설정

```bash
eb setenv GEMINI_API_KEY=your_key
```

### 5.2 Azure 배포

#### App Service

1. Azure CLI 설치 및 로그인

```bash
az login
```

2. 리소스 그룹 및 App Service 생성

```bash
az group create --name dise-analyzer-rg --location eastus
az appservice plan create --name dise-analyzer-plan --resource-group dise-analyzer-rg --sku B2
az webapp create --resource-group dise-analyzer-rg --plan dise-analyzer-plan --name dise-analyzer-app --runtime "PYTHON:3.10"
```

3. 환경 변수 설정

```bash
az webapp config appsettings set --resource-group dise-analyzer-rg --name dise-analyzer-app --settings GEMINI_API_KEY=your_key
```

4. 배포

```bash
az webapp deployment source config-local-git --name dise-analyzer-app --resource-group dise-analyzer-rg
git remote add azure <deployment-url>
git push azure main
```

### 5.3 Google Cloud Platform (GCP) 배포

#### Cloud Run

1. Dockerfile 준비 (4.1 참고)

2. 이미지 빌드 및 푸시

```bash
gcloud builds submit --tag gcr.io/your-project-id/dise-analyzer
```

3. Cloud Run에 배포

```bash
gcloud run deploy dise-analyzer \
  --image gcr.io/your-project-id/dise-analyzer \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 8Gi \
  --timeout 1200 \
  --set-env-vars GEMINI_API_KEY=your_key
```

---

## 6. 보안 고려사항

### 6.1 환경 변수 보호

- `.env` 파일을 절대 Git에 커밋하지 마세요
- 프로덕션 환경에서는 환경 변수 관리 도구 사용 (AWS Secrets Manager, Azure Key Vault 등)

### 6.2 파일 업로드 보안

- 파일 크기 제한: 현재 500MB (필요시 조정)
- 파일 타입 검증: 서버 측에서도 검증 수행
- 업로드 파일 스캔: 악성 코드 검사 고려

### 6.3 API 키 보호

- Gemini API 키는 환경 변수로만 관리
- API 키 사용량 모니터링 설정
- 필요시 API 키 로테이션

### 6.4 방화벽 설정

- 필요한 포트만 열기
- SSH 접근은 키 기반 인증만 허용
- 불필요한 서비스 비활성화

### 6.5 HTTPS 사용

- 모든 프로덕션 배포에서 HTTPS 필수
- Let's Encrypt 또는 상용 인증서 사용

---

## 7. 문제 해결

### 7.1 모델 파일 누락 오류

**증상**: `FileNotFoundError: best_model.pth`

**해결**:
```bash
# 모델 파일 경로 확인
ls -la ote_velum_classification_final/checkpoints/best_model.pth

# 모델 파일이 없다면 다운로드 스크립트 실행
cd ote_velum_classification_final
python download_model.py
```

### 7.2 메모리 부족 오류

**증상**: `MemoryError` 또는 프로세스 강제 종료

**해결**:
- 더 큰 인스턴스 타입 사용
- Gunicorn 워커 수 감소 (`-w 2`)
- 비디오 해상도 축소 또는 프레임 추출 FPS 감소

### 7.3 API 키 오류

**증상**: `GEMINI_API_KEY가 환경 변수에서 로드되지 않았습니다`

**해결**:
```bash
# .env 파일 확인
cat .env

# 환경 변수 직접 설정 (임시)
export GEMINI_API_KEY=your_key

# 또는 Docker 환경 변수로 전달
docker run -e GEMINI_API_KEY=your_key ...
```

### 7.4 포트 충돌

**증상**: `Address already in use`

**해결**:
```bash
# 포트 사용 중인 프로세스 확인
# Linux/macOS:
lsof -i :5000
# Windows:
netstat -ano | findstr :5000

# 프로세스 종료 또는 다른 포트 사용
gunicorn -b 0.0.0.0:8000 integrated_app:app
```

### 7.5 업로드 파일 크기 제한

**증상**: `413 Request Entity Too Large`

**해결**:
- Nginx 설정에서 `client_max_body_size 500M;` 추가
- Flask 설정에서 `MAX_CONTENT_LENGTH` 확인

---

## 8. 모니터링 및 로깅

### 8.1 로그 확인

```bash
# Gunicorn 로그
tail -f /var/log/gunicorn/error.log

# systemd 서비스 로그
sudo journalctl -u dise-analyzer -f

# Docker 로그
docker logs -f dise-analyzer
```

### 8.2 성능 모니터링

- CPU/메모리 사용량 모니터링
- 디스크 공간 확인 (outputs 폴더)
- API 응답 시간 추적

---

## 9. 백업 및 복구

### 9.1 중요 데이터 백업

```bash
# outputs 폴더 백업
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/

# 모델 파일 백업
cp ote_velum_classification_final/checkpoints/best_model.pth backups/
```

### 9.2 자동 백업 스크립트

`backup.sh` 예시:

```bash
#!/bin/bash
BACKUP_DIR="/backups/dise-analyzer"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/outputs_$DATE.tar.gz outputs/
tar -czf $BACKUP_DIR/model_$DATE.tar.gz ote_velum_classification_final/checkpoints/

# 30일 이상 된 백업 삭제
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

crontab에 추가:

```bash
0 2 * * * /path/to/backup.sh
```

---

## 10. 업데이트 절차

### 10.1 코드 업데이트

```bash
# Git에서 최신 코드 가져오기
git pull origin main

# 의존성 업데이트
pip install -r requirements.txt --upgrade

# 서비스 재시작
sudo systemctl restart dise-analyzer
# 또는
docker-compose restart
```

### 10.2 무중단 배포 (Blue-Green)

1. 새 버전을 별도 포트에서 실행
2. 헬스 체크 확인
3. Nginx 설정 변경하여 트래픽 전환
4. 이전 버전 종료

---

## 📞 추가 지원

배포 관련 문제가 발생하면:
1. 로그 파일 확인
2. 환경 변수 및 설정 파일 점검
3. 시스템 리소스 (메모리, 디스크) 확인
4. 이슈 트래커에 문제 보고

---

**마지막 업데이트**: 2025-01-XX

