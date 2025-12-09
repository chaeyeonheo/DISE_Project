"""
Google Drive에서 모델 파일을 다운로드하는 스크립트
배포 시 자동으로 실행됩니다.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Google Drive 파일 ID
GOOGLE_DRIVE_FILE_ID = "161GXpszELcLSc6ACP1Uzdpz26a8jXYDK"
MODEL_DIR = Path("ote_velum_classification_final/checkpoints")
MODEL_PATH = MODEL_DIR / "best_model.pth"


def download_file_from_google_drive(file_id, destination):
    """
    Google Drive에서 대용량 파일 다운로드
    
    Args:
        file_id: Google Drive 파일 ID
        destination: 저장 경로
    """
    def get_confirm_token(response):
        """다운로드 확인 토큰 추출"""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        """응답 내용을 파일로 저장 (진행률 표시)"""
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, "wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc='Downloading model', initial=0) as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    print(f"📥 Google Drive에서 모델 다운로드 시작...")
    print(f"   File ID: {file_id}")
    print(f"   Destination: {destination}")
    
    # 첫 번째 요청 (확인 토큰 받기)
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        # 확인 토큰이 있으면 다시 요청
        print("   확인 토큰 받음, 다운로드 진행...")
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    else:
        # 토큰이 없으면 바로 다운로드 (작은 파일의 경우)
        print("   직접 다운로드 진행...")
    
    # 파일 저장
    save_response_content(response, destination)
    print(f"✅ 모델 다운로드 완료: {destination}")


def main():
    """메인 함수"""
    # 모델 디렉토리 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 이미 모델 파일이 있으면 스킵
    if MODEL_PATH.exists():
        file_size = MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 모델 파일이 이미 존재합니다: {MODEL_PATH}")
        print(f"   크기: {file_size:.2f} MB")
        print("   다운로드를 건너뜁니다.")
        return
    
    try:
        # Google Drive에서 다운로드
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        
        # 다운로드 확인
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
            print(f"\n✅ 모델 다운로드 성공!")
            print(f"   경로: {MODEL_PATH}")
            print(f"   크기: {file_size:.2f} MB")
        else:
            print("❌ 다운로드 실패: 파일이 생성되지 않았습니다.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

