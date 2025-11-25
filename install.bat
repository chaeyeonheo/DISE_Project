@echo off
REM 수면 내시경 분석 시스템 설치 스크립트 (Windows)

echo ======================================
echo   수면 내시경 분석 시스템 설치
echo ======================================

REM Python 버전 체크
echo.
echo [1/5] Python 버전 확인...
python --version

if errorlevel 1 (
    echo ❌ Python이 설치되지 않았습니다.
    echo Python 3.8 이상을 설치해주세요.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 필수 패키지 설치
echo.
echo [2/5] 필수 패키지 설치 중...
pip install opencv-python flask numpy matplotlib tqdm

if errorlevel 1 (
    echo ❌ 패키지 설치에 실패했습니다.
    pause
    exit /b 1
)

REM 디렉토리 생성
echo.
echo [3/5] 디렉토리 구조 생성...
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs
if not exist templates mkdir templates
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js

echo.
echo [4/5] 설정 완료...

echo.
echo [5/5] 설치 완료!
echo.
echo ======================================
echo   설치가 완료되었습니다!
echo ======================================
echo.
echo 실행 방법:
echo   python app.py
echo.
echo 웹 브라우저에서 다음 주소로 접속:
echo   http://localhost:5000
echo.
echo ======================================
pause
