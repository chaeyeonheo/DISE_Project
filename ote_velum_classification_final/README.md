# DISE 프레임 분류기 (학습 + 추론)
OTE / Velum / None 분류 모델을 직접 전처리·학습·추론할 수 있는 코드입니다. 통합 웹앱이 동일한 가중치를 사용합니다.

## 구조
```
ote_velum_classification_final/
├── preprocess_with_analyzer.py  # 원본 비디오 → 프레임 전처리/라벨 생성
├── preprocess_videos.py         # 간단 전처리 유틸
├── dataset.py                   # DataLoader/transform 정의
├── train.py                     # 학습 루프 (config 인자/기본값)
├── inference.py                 # 이미지/비디오 추론
├── regenerate_annotations.py    # 어노테이션 재검증/재생성
├── plot_accuracy.py             # 학습 로그 시각화
├── plot_from_checkpoint.py      # 체크포인트 기반 플롯
├── check_test_data.py           # 데이터 무결성 점검
├── model.py                     # ResNet 기반 분류 모델 정의
├── download_model.py            # (옵션) 가중치 다운로드
├── requirements.txt             # 분류 모듈 의존성
└── checkpoints/                 # best_model.pth 배치 경로
```

## 빠른 학습/추론 흐름
```bash
# 0) 의존성
pip install -r requirements.txt

# 1) 전처리 & 라벨 생성
python preprocess_with_analyzer.py \
  --dataset "D:/path/to/raw_videos" \
  --output processed_dataset

# 2) 학습
python train.py   # 또는 config 인자 사용

# 3) 추론
python inference.py \
  --model checkpoints/best_model.pth \
  --input path/to/video.mp4 \
  --frame-interval 5 \
  --visualize
```

## 수동으로 가중치 받기
- `python download_model_from_drive.py` (루트에서 실행)  
  → `checkpoints/best_model.pth`에 저장 (Drive: https://drive.google.com/file/d/161GXpszELcLSc6ACP1Uzdpz26a8jXYDK/view?usp=drive_link)


