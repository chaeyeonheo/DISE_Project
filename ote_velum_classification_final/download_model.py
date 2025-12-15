"""
가중치 다운로드 스크립트 (Google Drive 단일 파일)
- 루트 `download_model_from_drive.py`와 동일한 파일 ID 사용
- 목적지: `checkpoints/best_model.pth`
"""

from pathlib import Path
import requests
from tqdm import tqdm
import re

GOOGLE_DRIVE_FILE_ID = "161GXpszELcLSc6ACP1Uzdpz26a8jXYDK"
DEST_PATH = Path("checkpoints/best_model.pth")


def download_file_from_google_drive(file_id: str, destination: Path):
    """Google Drive 대용량 파일 다운로드 (confirm 토큰 처리 포함)"""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"📥 Google Drive download (ID: {file_id})")
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    if "text/html" in response.headers.get("Content-Type", ""):
        content = response.text
        token_match = re.search(r"confirm=([^&]+)", content)
        token = token_match.group(1) if token_match else get_confirm_token(response)
    else:
        token = get_confirm_token(response)

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    save_response_content(response, destination)


def main():
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEST_PATH.exists():
        size_mb = DEST_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Already exists: {DEST_PATH} ({size_mb:.2f} MB)")
        return

    try:
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, DEST_PATH)
        size_mb = DEST_PATH.stat().st_size / (1024 * 1024)
        if size_mb < 1.0:
            raise ValueError(f"Downloaded file too small: {size_mb:.2f} MB")
        print(f"✅ Downloaded: {DEST_PATH} ({size_mb:.2f} MB)")
    except Exception as e:
        if DEST_PATH.exists():
            DEST_PATH.unlink()
        print(f"❌ Download failed: {e}")


if __name__ == "__main__":
    main()

