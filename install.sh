#!/bin/bash
# 수면 내시경 분석 시스템 설치 스크립트

echo "======================================"
echo "  수면 내시경 분석 시스템 설치"
echo "======================================"

# Python 버전 체크
echo ""
echo "[1/5] Python 버전 확인..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3가 설치되지 않았습니다."
    echo "Python 3.8 이상을 설치해주세요."
    exit 1
fi

# 필수 패키지 설치
echo ""
echo "[2/5] 필수 패키지 설치 중..."
pip3 install --user opencv-python flask numpy matplotlib tqdm

if [ $? -ne 0 ]; then
    echo "❌ 패키지 설치에 실패했습니다."
    exit 1
fi

# 디렉토리 생성
echo ""
echo "[3/5] 디렉토리 구조 생성..."
mkdir -p uploads outputs templates static/css static/js

# 권한 설정
echo ""
echo "[4/5] 권한 설정..."
chmod +x app.py

echo ""
echo "[5/5] 설치 완료!"
echo ""
echo "======================================"
echo "  설치가 완료되었습니다!"
echo "======================================"
echo ""
echo "실행 방법:"
echo "  python3 app.py"
echo ""
echo "웹 브라우저에서 다음 주소로 접속:"
echo "  http://localhost:5000"
echo ""
echo "======================================"
