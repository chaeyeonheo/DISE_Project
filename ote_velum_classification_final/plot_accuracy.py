"""
이미 학습 완료된 모델의 accuracy 그래프를 그리는 스크립트
training_history.json 파일을 읽어서 그래프 생성
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_accuracy_from_history(history_path='checkpoints/training_history.json', 
                               save_path='checkpoints/accuracy_plot.png'):
    """
    학습 히스토리 JSON 파일을 읽어서 accuracy 그래프 생성
    
    Args:
        history_path: training_history.json 파일 경로
        save_path: 저장할 그래프 파일 경로
    """
    history_path = Path(history_path)
    
    if not history_path.exists():
        print(f"❌ Error: {history_path} 파일을 찾을 수 없습니다.")
        print(f"   학습을 먼저 실행하거나, training_history.json 파일이 있는지 확인해주세요.")
        return
    
    # 히스토리 로드
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if 'train_acc' not in history or 'val_acc' not in history:
        print(f"❌ Error: 히스토리 파일에 accuracy 데이터가 없습니다.")
        return
    
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    if len(train_acc) == 0 or len(val_acc) == 0:
        print(f"❌ Error: accuracy 데이터가 비어있습니다.")
        return
    
    # 그래프 생성
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_acc) + 1)
    
    # Train/Val Accuracy 플롯
    plt.plot(epochs, train_acc, 'o-', label='Train Accuracy', 
            linewidth=2.5, markersize=6, color='#3b82f6')
    plt.plot(epochs, val_acc, 's-', label='Validation Accuracy', 
            linewidth=2.5, markersize=6, color='#10b981')
    
    # Best validation accuracy 표시
    best_epoch = np.argmax(val_acc) + 1
    best_acc = val_acc[best_epoch - 1]
    plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.plot(best_epoch, best_acc, 'ro', markersize=12, label=f'Best Val Acc: {best_acc:.2f}%')
    
    # 최종 accuracy 표시
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    plt.plot(len(epochs), final_train_acc, 'bo', markersize=10, alpha=0.7)
    plt.plot(len(epochs), final_val_acc, 'go', markersize=10, alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Model Accuracy Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 100])
    
    # 통계 정보 텍스트 박스
    stats_text = f'Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})\n'
    stats_text += f'Final Train Acc: {final_train_acc:.2f}%\n'
    stats_text += f'Final Val Acc: {final_val_acc:.2f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Accuracy 그래프가 저장되었습니다: {save_path}")
    print(f"   Best Val Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"   Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"   Final Val Accuracy: {final_val_acc:.2f}%")
    
    plt.close()


def plot_full_history(history_path='checkpoints/training_history.json',
                     save_path='checkpoints/training_history.png'):
    """
    전체 학습 히스토리 그래프 (Loss, Accuracy, Learning Rate)
    """
    history_path = Path(history_path)
    
    if not history_path.exists():
        print(f"❌ Error: {history_path} 파일을 찾을 수 없습니다.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], linewidth=2, color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 전체 학습 히스토리 그래프가 저장되었습니다: {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='학습 완료된 모델의 accuracy 그래프 생성')
    parser.add_argument('--history', type=str, default='checkpoints/training_history.json',
                       help='training_history.json 파일 경로')
    parser.add_argument('--output', type=str, default='checkpoints/accuracy_plot.png',
                       help='저장할 그래프 파일 경로')
    parser.add_argument('--full', action='store_true',
                       help='전체 히스토리 그래프도 함께 생성 (Loss, Accuracy, LR)')
    
    args = parser.parse_args()
    
    # Accuracy 그래프 생성
    plot_accuracy_from_history(args.history, args.output)
    
    # 전체 히스토리 그래프도 생성 (옵션)
    if args.full:
        full_output = Path(args.output).parent / 'training_history.png'
        plot_full_history(args.history, full_output)





