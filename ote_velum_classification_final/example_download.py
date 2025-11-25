"""
모델 다운로드 사용 예제
"""

from download_model import ModelDownloader, download_from_google_drive, download_from_huggingface
from pathlib import Path


def example_1_list_models():
    """예제 1: 사용 가능한 모델 목록 확인"""
    print("\n" + "="*70)
    print("Example 1: List Available Models")
    print("="*70)
    
    downloader = ModelDownloader()
    downloader.list_available_models()


def example_2_download_model():
    """예제 2: 모델 다운로드"""
    print("\n" + "="*70)
    print("Example 2: Download Model from Registry")
    print("="*70)
    
    downloader = ModelDownloader(save_dir='pretrained_models')
    
    # 모델 다운로드 (실제로는 URL이 유효해야 함)
    try:
        model_path = downloader.download_model('resnet50_ote_velum_v1')
        print(f"Model downloaded to: {model_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Note: This is expected if the URL in MODEL_REGISTRY is not configured")


def example_3_load_model():
    """예제 3: 모델 다운로드 및 로드"""
    print("\n" + "="*70)
    print("Example 3: Download and Load Model")
    print("="*70)
    
    downloader = ModelDownloader(save_dir='pretrained_models')
    
    try:
        model, checkpoint = downloader.load_model('resnet50_ote_velum_v1', device='cpu')
        print("Model loaded successfully!")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
    except Exception as e:
        print(f"Load failed: {e}")
        print("Note: This is expected if the model file doesn't exist")


def example_4_google_drive():
    """예제 4: Google Drive에서 다운로드"""
    print("\n" + "="*70)
    print("Example 4: Download from Google Drive")
    print("="*70)
    
    # Google Drive 파일 ID (예시)
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
    output_path = Path("pretrained_models/model_from_gdrive.pth")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"File ID: {file_id}")
    print(f"Output: {output_path}")
    print("\nNote: Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with actual file ID")
    
    # 실제 다운로드 (주석 처리)
    # download_from_google_drive(file_id, output_path)


def example_5_huggingface():
    """예제 5: Hugging Face에서 다운로드"""
    print("\n" + "="*70)
    print("Example 5: Download from Hugging Face")
    print("="*70)
    
    repo_id = "username/ote-velum-classifier"
    filename = "best_model.pth"
    output_path = Path("pretrained_models/model_from_hf.pth")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"Repository: {repo_id}")
    print(f"Filename: {filename}")
    print(f"Output: {output_path}")
    print("\nNote: Replace with actual Hugging Face repository ID")
    
    # 실제 다운로드 (주석 처리)
    # download_from_huggingface(repo_id, filename, output_path)


def example_6_use_in_inference():
    """예제 6: 다운로드한 모델로 추론"""
    print("\n" + "="*70)
    print("Example 6: Use Downloaded Model for Inference")
    print("="*70)
    
    code_example = """
from download_model import ModelDownloader
from inference import FrameClassifierInference

# 1. 모델 다운로드
downloader = ModelDownloader()
model_path = downloader.download_model('resnet50_ote_velum_v1')

# 2. 추론기 생성
inferencer = FrameClassifierInference(
    model_path=model_path,
    device='cuda'
)

# 3. 비디오 분석
results = inferencer.predict_video(
    'test_video.mp4',
    output_path='results.json',
    visualize=True
)

# 4. 결과 확인
print(f"Total frames: {results['statistics']['total_predicted_frames']}")
for class_name, count in results['statistics']['class_counts'].items():
    print(f"{class_name}: {count} frames")
"""
    
    print("Python code example:")
    print(code_example)


def main():
    """모든 예제 실행"""
    examples = [
        ("List Models", example_1_list_models),
        ("Download Model", example_2_download_model),
        ("Load Model", example_3_load_model),
        ("Google Drive", example_4_google_drive),
        ("Hugging Face", example_5_huggingface),
        ("Inference", example_6_use_in_inference),
    ]
    
    print("\n" + "="*70)
    print("Model Download Examples")
    print("="*70)
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
    
    print("\n" + "="*70)
    choice = input("\nSelect example to run (1-6, or 'all'): ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            func()
            input("\nPress Enter to continue...")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        examples[int(choice)-1][1]()
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
