import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

from model import create_model
from dataset import create_dataloaders


def calculate_class_weights(dataloader, device):
    """
    학습 데이터셋의 클래스 비율을 계산하여 Loss 가중치 생성
    (Inverse Class Frequency 방법 사용)
    """
    print("\n⚖️ Calculating class weights...")
    
    # 데이터셋에서 타겟 라벨 추출
    # ImageFolder나 일반적인 Dataset을 가정
    try:
        if hasattr(dataloader.dataset, 'targets'):
            labels = dataloader.dataset.targets
        elif hasattr(dataloader.dataset, 'labels'):
            labels = dataloader.dataset.labels
        else:
            # 타겟 속성이 없는 경우 직접 순회 (시간이 좀 걸릴 수 있음)
            labels = []
            for _, label, _ in tqdm(dataloader, desc="Scanning dataset for weights"):
                labels.extend(label.tolist())
    except Exception as e:
        print(f"Warning: Could not auto-calculate weights ({e}). Using equal weights.")
        return None

    # 클래스별 개수 카운트
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(counts)
    
    # 클래스 인덱스 순서대로 정렬 (0, 1, 2...)
    sorted_counts = [counts[i] for i in range(n_classes)]
    
    print(f"   Class counts: {dict(sorted(counts.items()))}")
    
    # 가중치 계산: N_samples / (N_classes * Count_class)
    # 개수가 적은 클래스는 가중치가 커지고, 많은 클래스는 작아짐
    weights = [n_samples / (n_classes * c) for c in sorted_counts]
    
    # Tensor 변환 및 Device 이동
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"   Computed weights: {class_weights.cpu().numpy()}")
    return class_weights


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, config, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Loss 설정 (Class Weights 적용)
        if class_weights is not None:
            print(f"✅ Using Weighted CrossEntropyLoss: {class_weights.cpu().numpy()}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            print("ℹ️ Using Standard CrossEntropyLoss")
            self.criterion = nn.CrossEntropyLoss()
            
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config['epochs'], eta_min=1e-6
            )
        
        # 학습 기록
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # 저장 디렉토리
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """1 에포크 학습"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """검증"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """전체 학습 루프"""
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model_name']}")
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"{'='*50}")
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate()
            
            # Scheduler 업데이트 (Loss 기준)
            if self.config['scheduler'] == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 기록
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 결과 출력
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Best model 저장
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = self.save_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, self.best_model_path)
                print(f"✓ Best model saved with val_acc: {val_acc:.2f}%")
            
            # 주기적으로 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
        
        # 학습 히스토리 저장
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best model saved at: {self.best_model_path}")
        
        return self.history
    
    def test(self, model_path=None):
        """테스트 데이터 평가"""
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_filenames = []
        
        print("\nEvaluating on test set...")
        with torch.no_grad():
            for images, labels, filenames in tqdm(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_filenames.extend(filenames)
        
        # Classification report
        # 데이터셋의 클래스 이름 가져오기
        if hasattr(self.test_loader.dataset, 'classes'):
            label_names = self.test_loader.dataset.classes
        else:
            label_names = ['OTE', 'Velum', 'None']  # 기본값
            
        report = classification_report(
            all_labels, all_predictions, 
            target_names=label_names,
            digits=4
        )
        
        print("\n=== Test Results ===")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm, label_names)
        
        # 결과 저장
        results = {
            'labels': [int(l) for l in all_labels],
            'predictions': [int(p) for p in all_predictions],
            'filenames': all_filenames,
            'classification_report': report
        }
        
        results_path = self.save_dir / 'test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names):
        """Confusion matrix 시각화"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        save_path = self.save_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_training_history(self):
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning Rate
        axes[2].plot(self.history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)
        
        plt.tight_layout()
        save_path = self.save_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()


def main():
    # 설정
    config = {
        'model_name': 'resnet50',  # 'efficientnet_b0'보다 resnet50 추천 (안정성)
        'num_classes': 3,
        'pretrained': True,
        'batch_size': 32,
        'epochs': 20,           # 데이터 재정비했으므로 20 에포크면 충분
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'img_size': 224,
        'scheduler': 'plateau',
        'save_dir': 'checkpoints',
        'annotation_file': 'processed_dataset/annotations.json',
        'data_root': 'processed_dataset'
    }
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터로더 생성
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        annotation_file=config['annotation_file'],
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # ⭐ Class Weights 계산 (여기서 자동 계산해서 Trainer에 넘김)
    class_weights = calculate_class_weights(train_loader, device)
    
    # 모델 생성
    print(f"\nCreating model: {config['model_name']}")
    model = create_model(
        model_name=config['model_name'],
        num_classes=num_classes,
        pretrained=config['pretrained']
    )
    
    # Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config,
        class_weights=class_weights  # ⭐ 가중치 전달
    )
    
    # 학습
    history = trainer.train()
    
    # 학습 히스토리 플롯
    trainer.plot_training_history()
    
    # 테스트
    trainer.test(model_path=trainer.best_model_path)
    
    print("\nAll done!")


if __name__ == '__main__':
    main()