import torch
import os
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib


class ModelDownloader:
    """
    학습된 모델을 다운로드하는 클래스
    """
    
    # 모델 저장소 설정 (예시)
    MODEL_REGISTRY = {
        'resnet50_ote_velum_v1': {
            'url': 'https://your-storage.com/models/resnet50_ote_velum_v1.pth',
            'filename': 'resnet50_ote_velum_v1.pth',
            'description': 'ResNet-50 model trained on OTE/Velum/None classification',
            'md5': 'abc123def456...',  # MD5 checksum for verification
            'size_mb': 95.4
        },
        'resnet18_ote_velum_v1': {
            'url': 'https://your-storage.com/models/resnet18_ote_velum_v1.pth',
            'filename': 'resnet18_ote_velum_v1.pth',
            'description': 'ResNet-18 lightweight model',
            'md5': 'def789ghi012...',
            'size_mb': 44.6
        },
        'efficientnet_b0_ote_velum_v1': {
            'url': 'https://your-storage.com/models/efficientnet_b0_ote_velum_v1.pth',
            'filename': 'efficientnet_b0_ote_velum_v1.pth',
            'description': 'EfficientNet-B0 model',
            'md5': 'ghi345jkl678...',
            'size_mb': 20.3
        }
    }
    
    def __init__(self, save_dir='pretrained_models'):
        """
        Args:
            save_dir: 모델을 저장할 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
    
    def list_available_models(self):
        """사용 가능한 모델 목록 출력"""
        print("\n" + "="*70)
        print("Available Pretrained Models")
        print("="*70)
        
        for idx, (model_name, info) in enumerate(self.MODEL_REGISTRY.items(), 1):
            print(f"\n{idx}. {model_name}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Filename: {info['filename']}")
        
        print("\n" + "="*70)
    
    def download_file(self, url, dest_path, desc=None):
        """
        파일 다운로드 (진행률 표시)
        
        Args:
            url: 다운로드 URL
            dest_path: 저장 경로
            desc: 진행률 바 설명
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=desc or 'Downloading') as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def compute_md5(self, file_path):
        """파일의 MD5 체크섬 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def verify_checksum(self, file_path, expected_md5):
        """체크섬 검증"""
        if expected_md5 is None:
            print("⚠️  No checksum provided, skipping verification")
            return True
        
        print("Verifying file integrity...")
        actual_md5 = self.compute_md5(file_path)
        
        if actual_md5 == expected_md5:
            print("✓ Checksum verified successfully")
            return True
        else:
            print(f"✗ Checksum mismatch!")
            print(f"  Expected: {expected_md5}")
            print(f"  Got: {actual_md5}")
            return False
    
    def download_model(self, model_name, force_download=False):
        """
        모델 다운로드
        
        Args:
            model_name: 다운로드할 모델 이름
            force_download: 이미 존재해도 다시 다운로드
        
        Returns:
            model_path: 다운로드된 모델 경로
        """
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry. "
                           f"Available models: {list(self.MODEL_REGISTRY.keys())}")
        
        model_info = self.MODEL_REGISTRY[model_name]
        model_path = self.save_dir / model_info['filename']
        
        # 이미 존재하는 경우
        if model_path.exists() and not force_download:
            print(f"\n✓ Model already exists at: {model_path}")
            
            # 체크섬 검증
            if model_info.get('md5'):
                if self.verify_checksum(model_path, model_info['md5']):
                    return model_path
                else:
                    print("Re-downloading due to checksum mismatch...")
            else:
                return model_path
        
        # 다운로드
        print(f"\nDownloading {model_name}...")
        print(f"Size: {model_info['size_mb']:.1f} MB")
        print(f"Destination: {model_path}")
        
        try:
            self.download_file(
                model_info['url'],
                model_path,
                desc=f"Downloading {model_info['filename']}"
            )
            
            # 체크섬 검증
            if model_info.get('md5'):
                if not self.verify_checksum(model_path, model_info['md5']):
                    os.remove(model_path)
                    raise ValueError("Downloaded file failed checksum verification")
            
            print(f"\n✓ Successfully downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            if model_path.exists():
                os.remove(model_path)
            raise Exception(f"Failed to download model: {str(e)}")
    
    def load_model(self, model_name, device='cuda'):
        """
        모델 다운로드 및 로드
        
        Args:
            model_name: 모델 이름
            device: 'cuda' or 'cpu'
        
        Returns:
            model, checkpoint: 로드된 모델과 체크포인트
        """
        from model import create_model
        
        # 모델 다운로드
        model_path = self.download_model(model_name)
        
        # 체크포인트 로드
        print(f"\nLoading model from {model_path}...")
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 생성
        config = checkpoint.get('config', {})
        model = create_model(
            model_name=config.get('model_name', 'resnet50'),
            num_classes=config.get('num_classes', 3),
            pretrained=False
        )
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"  - Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        
        return model, checkpoint


def download_from_google_drive(file_id, destination):
    """
    Google Drive에서 파일 다운로드
    
    Args:
        file_id: Google Drive 파일 ID
        destination: 저장 경로
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        
        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, desc='Downloading') as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)


def download_from_huggingface(repo_id, filename, destination):
    """
    Hugging Face Hub에서 파일 다운로드
    
    Args:
        repo_id: Hugging Face 저장소 ID (예: "username/model-name")
        filename: 파일명
        destination: 저장 경로
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"Downloading from Hugging Face: {repo_id}/{filename}")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None
        )
        
        # 목적지로 복사
        import shutil
        shutil.copy(downloaded_path, destination)
        print(f"✓ Downloaded to {destination}")
        
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface-hub")
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")


def main():
    """메인 함수 - CLI 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download pretrained models for OTE/Velum/None classification'
    )
    parser.add_argument('--list', action='store_true',
                       help='List all available models')
    parser.add_argument('--model', type=str,
                       help='Model name to download')
    parser.add_argument('--save-dir', type=str, default='pretrained_models',
                       help='Directory to save models')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if file exists')
    parser.add_argument('--load', action='store_true',
                       help='Load the model after downloading')
    
    # Google Drive 옵션
    parser.add_argument('--gdrive-id', type=str,
                       help='Download from Google Drive using file ID')
    parser.add_argument('--output', type=str,
                       help='Output filename for Google Drive download')
    
    # Hugging Face 옵션
    parser.add_argument('--hf-repo', type=str,
                       help='Hugging Face repository ID')
    parser.add_argument('--hf-filename', type=str,
                       help='Filename in Hugging Face repository')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(save_dir=args.save_dir)
    
    # 모델 목록 출력
    if args.list:
        downloader.list_available_models()
        return
    
    # Google Drive에서 다운로드
    if args.gdrive_id:
        if not args.output:
            print("Error: --output is required for Google Drive download")
            return
        
        dest_path = Path(args.save_dir) / args.output
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        
        print(f"\nDownloading from Google Drive...")
        print(f"File ID: {args.gdrive_id}")
        print(f"Destination: {dest_path}")
        
        download_from_google_drive(args.gdrive_id, dest_path)
        print(f"\n✓ Download completed: {dest_path}")
        return
    
    # Hugging Face에서 다운로드
    if args.hf_repo and args.hf_filename:
        if not args.output:
            args.output = args.hf_filename
        
        dest_path = Path(args.save_dir) / args.output
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        
        download_from_huggingface(args.hf_repo, args.hf_filename, dest_path)
        return
    
    # 모델 레지스트리에서 다운로드
    if args.model:
        try:
            if args.load:
                model, checkpoint = downloader.load_model(args.model)
                print("\n✓ Model ready for inference!")
            else:
                model_path = downloader.download_model(args.model, args.force)
                print(f"\n✓ Model saved at: {model_path}")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return
    else:
        print("Please specify --model or --list to see available models")
        downloader.list_available_models()


if __name__ == '__main__':
    main()
