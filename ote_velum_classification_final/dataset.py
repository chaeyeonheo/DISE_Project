import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class VideoFrameDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        """
        Args:
            annotations: 프레임 정보 리스트
            root_dir: 데이터셋 루트 디렉토리
            transform: 이미지 변환
        """
        self.annotations = annotations
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 레이블 매핑
        self.label_map = {'OTE': 0, 'Velum': 1, 'None': 2}
        self.num_classes = len(self.label_map)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 이미지 경로 구성
        label_name = ann['label']
        img_path = self.root_dir / label_name / ann['filename']
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # 레이블 변환
        label = self.label_map[label_name]
        
        return image, label, ann['filename']


def get_transforms(img_size=224, is_train=True):
    """데이터 증강 변환 정의"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(annotation_file, root_dir, batch_size=32, 
                       img_size=224, val_split=0.15, test_split=0.15,
                       num_workers=4):
    """데이터로더 생성"""
    # 어노테이션 로드
    with open(annotation_file, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)
    
    # 레이블별로 그룹화
    label_groups = {'OTE': [], 'Velum': [], 'None': []}
    for ann in all_annotations:
        label_groups[ann['label']].append(ann)
    
    # 통계 출력
    print("\n=== Dataset Statistics ===")
    for label, anns in label_groups.items():
        print(f"{label}: {len(anns)} frames")
    
    # 각 클래스별로 train/val/test 분할
    train_anns, val_anns, test_anns = [], [], []
    
    for label, anns in label_groups.items():
        # 비디오 이름으로 그룹화 (같은 비디오의 프레임이 train/val/test에 섞이지 않도록)
        video_groups = {}
        for ann in anns:
            video_name = ann['video_name']
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(ann)
        
        video_names = list(video_groups.keys())
        
        # 비디오 단위로 분할
        train_videos, temp_videos = train_test_split(
            video_names, test_size=(val_split + test_split), random_state=42
        )
        
        val_videos, test_videos = train_test_split(
            temp_videos, test_size=test_split/(val_split + test_split), random_state=42
        )
        
        # 각 그룹에 프레임 추가
        for video in train_videos:
            train_anns.extend(video_groups[video])
        for video in val_videos:
            val_anns.extend(video_groups[video])
        for video in test_videos:
            test_anns.extend(video_groups[video])
    
    print(f"\nTrain: {len(train_anns)} frames")
    print(f"Val: {len(val_anns)} frames")
    print(f"Test: {len(test_anns)} frames")
    
    # 데이터셋 생성
    train_dataset = VideoFrameDataset(
        train_anns, root_dir, transform=get_transforms(img_size, is_train=True)
    )
    val_dataset = VideoFrameDataset(
        val_anns, root_dir, transform=get_transforms(img_size, is_train=False)
    )
    test_dataset = VideoFrameDataset(
        test_anns, root_dir, transform=get_transforms(img_size, is_train=False)
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes


if __name__ == '__main__':
    # 테스트
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        annotation_file='processed_dataset/annotations.json',
        root_dir='processed_dataset',
        batch_size=32
    )
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 샘플 배치 확인
    for images, labels, filenames in train_loader:
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels[:5]}")
        print(f"Filenames: {filenames[:5]}")
        break
