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
RUN pip install --no-cache-dir -r requirements.txt gunicorn

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
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "1200", "--max-requests", "1000", "integrated_app:app"]




