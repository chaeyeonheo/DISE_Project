import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

class VideoFrameClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name='resnet50', pretrained=True):
        """
        비디오 프레임 분류기
        
        Args:
            num_classes: 클래스 수 (OTE, Velum, None)
            model_name: 백본 모델 ('resnet50', 'efficientnet_b0', 'resnet18')
            pretrained: ImageNet 사전학습 가중치 사용 여부
        """
        super(VideoFrameClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = models.resnet50(weights=None)
            
            # 마지막 FC layer 교체
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            
            # 마지막 classifier 교체
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """백본 네트워크 freeze (feature extraction)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 마지막 layer만 학습
        if self.model_name in ['resnet50', 'resnet18']:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'efficientnet_b0':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        """백본 네트워크 unfreeze (fine-tuning)"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class CustomCNN(nn.Module):
    """커스텀 CNN 모델 (경량화)"""
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(model_name='resnet50', num_classes=3, pretrained=True):
    """모델 생성 헬퍼 함수"""
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes)
    else:
        model = VideoFrameClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained
        )
    
    return model


if __name__ == '__main__':
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = ['resnet50', 'resnet18', 'efficientnet_b0', 'custom_cnn']
    
    for model_name in models_to_test:
        print(f"\n=== Testing {model_name} ===")
        model = create_model(model_name=model_name, num_classes=3, pretrained=True)
        model = model.to(device)
        
        # 더미 입력
        dummy_input = torch.randn(4, 3, 224, 224).to(device)
        
        # Forward pass
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
