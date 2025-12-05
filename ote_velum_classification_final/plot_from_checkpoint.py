"""
best_model.pth 체크포인트에서 정보를 추출하여 간단한 그래프 생성
(전체 히스토리가 없어도 최종 결과를 시각화)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_from_checkpoint(checkpoint_path='checkpoints/best_model.pth',
                        save_path='checkpoints/model_summary.png'):
    """
    체크포인트에서 정보를 추출하여 요약 그래프 생성
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: {checkpoint_path} 파일을 찾을 수 없습니다.")
        return
    
    # 체크포인트 로드
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error: 체크포인트 로드 실패: {e}")
        return
    
    epoch = checkpoint.get('epoch', 'N/A')
    val_acc = checkpoint.get('val_acc', 'N/A')
    config = checkpoint.get('config', {})
    total_epochs = config.get('epochs', 'N/A')
    
    if val_acc == 'N/A' or epoch == 'N/A':
        print(f"❌ Error: 체크포인트에 필요한 정보가 없습니다.")
        print(f"   Available keys: {list(checkpoint.keys())}")
        return
    
    print(f"📊 체크포인트 정보:")
    print(f"   Epoch: {epoch}/{total_epochs}")
    print(f"   Validation Accuracy: {val_acc:.2f}%")
    
    # 간단한 요약 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 최종 성능 표시 (Bar Chart)
    axes[0].barh(['Validation\nAccuracy'], [val_acc], color='#10b981', height=0.5)
    axes[0].set_xlim([0, 100])
    axes[0].set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Final Model Performance', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 값 표시
    axes[0].text(val_acc, 0, f' {val_acc:.2f}%', 
                va='center', fontsize=14, fontweight='bold', color='#10b981')
    
    # 2. 학습 정보 표시 (Text Box)
    axes[1].axis('off')
    info_text = f"""
    Model Training Summary
    
    Model: {config.get('model_name', 'N/A')}
    Total Epochs: {total_epochs}
    Best Epoch: {epoch + 1}
    
    Best Validation Accuracy: {val_acc:.2f}%
    
    Learning Rate: {config.get('learning_rate', 'N/A')}
    Batch Size: {config.get('batch_size', 'N/A')}
    Image Size: {config.get('img_size', 'N/A')}
    """
    
    axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
    
    plt.suptitle('Model Checkpoint Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 저장
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 요약 그래프가 저장되었습니다: {save_path}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='체크포인트에서 정보 추출하여 그래프 생성')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='체크포인트 파일 경로')
    parser.add_argument('--output', type=str, default='checkpoints/model_summary.png',
                       help='저장할 그래프 파일 경로')
    
    args = parser.parse_args()
    
    plot_from_checkpoint(args.checkpoint, args.output)





