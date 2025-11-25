import torch
import os
from pathlib import Path
import hashlib
import json


class ModelUploader:
    """
    학습된 모델을 업로드하는 클래스
    """
    
    def __init__(self, model_path):
        """
        Args:
            model_path: 업로드할 모델 파일 경로
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
    
    def compute_md5(self):
        """파일의 MD5 체크섬 계산"""
        hash_md5 = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_model_info(self):
        """모델 정보 추출"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            info = {
                'filename': self.model_path.name,
                'size_mb': round(self.file_size_mb, 2),
                'md5': self.compute_md5(),
                'config': checkpoint.get('config', {}),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_acc': checkpoint.get('val_acc', 'N/A'),
            }
            
            return info
        except Exception as e:
            print(f"Warning: Could not load model info: {e}")
            return {
                'filename': self.model_path.name,
                'size_mb': round(self.file_size_mb, 2),
                'md5': self.compute_md5(),
            }
    
    def save_model_card(self, output_path='MODEL_CARD.json'):
        """모델 정보를 JSON 파일로 저장"""
        info = self.get_model_info()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Model card saved to {output_path}")
        return info
    
    def upload_to_google_drive(self, credentials_path=None):
        """
        Google Drive에 업로드
        
        Args:
            credentials_path: Google Drive API 인증 정보 경로
        
        Returns:
            file_id: 업로드된 파일의 Google Drive ID
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            import pickle
            
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            
            creds = None
            token_path = 'token.pickle'
            
            # 토큰이 있으면 로드
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            # 유효하지 않으면 로그인
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not credentials_path:
                        credentials_path = 'credentials.json'
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # 토큰 저장
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            # Drive API 서비스 생성
            service = build('drive', 'v3', credentials=creds)
            
            # 파일 메타데이터
            file_metadata = {
                'name': self.model_path.name,
                'description': f'OTE/Velum/None Classification Model - {self.file_size_mb:.1f}MB'
            }
            
            # 파일 업로드
            media = MediaFileUpload(
                str(self.model_path),
                resumable=True,
                chunksize=1024*1024  # 1MB chunks
            )
            
            print(f"\nUploading to Google Drive...")
            print(f"File: {self.model_path.name}")
            print(f"Size: {self.file_size_mb:.1f} MB")
            
            request = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Upload progress: {int(status.progress() * 100)}%")
            
            file_id = response.get('id')
            
            # 공유 링크 생성 (선택사항)
            service.permissions().create(
                fileId=file_id,
                body={'type': 'anyone', 'role': 'reader'}
            ).execute()
            
            share_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            print(f"\n✓ Upload completed!")
            print(f"File ID: {file_id}")
            print(f"Share link: {share_link}")
            print(f"Direct download: {download_link}")
            
            # 다운로드 명령어 출력
            print(f"\nTo download this model, use:")
            print(f"python download_model.py --gdrive-id {file_id} --output {self.model_path.name}")
            
            return file_id
            
        except ImportError:
            print("Error: Google API libraries not installed.")
            print("Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return None
        except Exception as e:
            print(f"Error uploading to Google Drive: {e}")
            return None
    
    def upload_to_huggingface(self, repo_id, token=None, commit_message=None):
        """
        Hugging Face Hub에 업로드
        
        Args:
            repo_id: Hugging Face 저장소 ID (예: "username/model-name")
            token: Hugging Face API 토큰
            commit_message: 커밋 메시지
        
        Returns:
            url: 업로드된 파일 URL
        """
        try:
            from huggingface_hub import HfApi, login
            
            # 로그인
            if token:
                login(token=token)
            else:
                print("Please enter your Hugging Face token:")
                login()
            
            api = HfApi()
            
            # 모델 정보 생성
            info = self.get_model_info()
            
            if commit_message is None:
                commit_message = f"Upload {self.model_path.name} (Val Acc: {info.get('val_acc', 'N/A')})"
            
            print(f"\nUploading to Hugging Face Hub...")
            print(f"Repository: {repo_id}")
            print(f"File: {self.model_path.name}")
            print(f"Size: {self.file_size_mb:.1f} MB")
            
            # 파일 업로드
            url = api.upload_file(
                path_or_fileobj=str(self.model_path),
                path_in_repo=self.model_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )
            
            print(f"\n✓ Upload completed!")
            print(f"URL: {url}")
            
            # 다운로드 명령어 출력
            print(f"\nTo download this model, use:")
            print(f"python download_model.py --hf-repo {repo_id} --hf-filename {self.model_path.name} --output {self.model_path.name}")
            
            return url
            
        except ImportError:
            print("Error: huggingface_hub not installed.")
            print("Install with: pip install huggingface-hub")
            return None
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            return None
    
    def create_download_links(self, gdrive_id=None, hf_repo=None):
        """
        다운로드 링크 정보 생성
        
        Args:
            gdrive_id: Google Drive 파일 ID
            hf_repo: Hugging Face 저장소 ID
        """
        info = self.get_model_info()
        
        links_info = {
            'model_name': self.model_path.stem,
            'filename': self.model_path.name,
            'size_mb': info['size_mb'],
            'md5': info['md5'],
            'download_links': {}
        }
        
        if gdrive_id:
            links_info['download_links']['google_drive'] = {
                'file_id': gdrive_id,
                'direct_url': f"https://drive.google.com/uc?export=download&id={gdrive_id}",
                'share_url': f"https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing",
                'command': f"python download_model.py --gdrive-id {gdrive_id} --output {self.model_path.name}"
            }
        
        if hf_repo:
            links_info['download_links']['huggingface'] = {
                'repo_id': hf_repo,
                'url': f"https://huggingface.co/{hf_repo}/resolve/main/{self.model_path.name}",
                'command': f"python download_model.py --hf-repo {hf_repo} --hf-filename {self.model_path.name} --output {self.model_path.name}"
            }
        
        # JSON 파일로 저장
        output_path = f"{self.model_path.stem}_download_info.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(links_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Download info saved to {output_path}")
        return links_info


def main():
    """메인 함수 - CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Upload trained models to cloud storage'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--platform', type=str, choices=['gdrive', 'huggingface', 'both'],
                       default='both',
                       help='Upload platform')
    
    # Google Drive 옵션
    parser.add_argument('--gdrive-creds', type=str, default='credentials.json',
                       help='Path to Google Drive credentials file')
    
    # Hugging Face 옵션
    parser.add_argument('--hf-repo', type=str,
                       help='Hugging Face repository ID (e.g., "username/model-name")')
    parser.add_argument('--hf-token', type=str,
                       help='Hugging Face API token')
    
    # 기타
    parser.add_argument('--save-info', action='store_true',
                       help='Save model information to JSON')
    
    args = parser.parse_args()
    
    # Uploader 생성
    try:
        uploader = ModelUploader(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 모델 정보 저장
    if args.save_info:
        uploader.save_model_card()
    
    gdrive_id = None
    hf_repo = None
    
    # Google Drive 업로드
    if args.platform in ['gdrive', 'both']:
        print("\n" + "="*70)
        print("Uploading to Google Drive")
        print("="*70)
        gdrive_id = uploader.upload_to_google_drive(args.gdrive_creds)
    
    # Hugging Face 업로드
    if args.platform in ['huggingface', 'both']:
        if not args.hf_repo:
            print("\nError: --hf-repo is required for Hugging Face upload")
        else:
            print("\n" + "="*70)
            print("Uploading to Hugging Face Hub")
            print("="*70)
            uploader.upload_to_huggingface(args.hf_repo, args.hf_token)
            hf_repo = args.hf_repo
    
    # 다운로드 정보 생성
    if gdrive_id or hf_repo:
        uploader.create_download_links(gdrive_id, hf_repo)


if __name__ == '__main__':
    main()
