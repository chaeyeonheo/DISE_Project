# DISE Video Frame Classifier

DISE(Drug-Induced Sleep Endoscopy) 비디오에서 `OTE / Velum / None` 프레임을 자동 분류하는 파이프라인입니다.  
AirwayOcclusionAnalyzer 기반 전처리 → 이미지 분류 학습 → 추론까지 한 번에 수행할 수 있습니다.

---

## 1. 빠른 시작

```bash
# 0. 의존성 설치
pip install -r requirements.txt

# 1. 프레임 추출 & 레이블링
python preprocess_with_analyzer.py \
  --dataset "D:/chaeyeon/.../DISE_DATA(AIHub)" \
  --output processed_dataset

# 2. 학습
python train.py

# 3. 추론
python inference.py \
  --model checkpoints/best_model.pth \
  --input sample_video.mp4 \
  --visualize
```

---

## 2. 프로젝트 구조

```
ote_velum_classification_final/
├── dataset/                     # 원본 비디오(OTE/Velum)
├── processed_dataset/           # 전처리 결과(클래스별 폴더 + annotations.json)
├── checkpoints/                 # 학습 산출물(best_model, history 등)
├── preprocess_with_analyzer.py  # Analyzer 기반 전처리
├── dataset.py                   # DataLoader/transform
├── model.py                     # ResNet/EfficientNet/Custom CNN
├── train.py                     # Trainer 루프
├── inference.py                 # 이미지/비디오 추론
├── download_model.py / upload_model.py
├── regenerate_annotations.py    # 프레임/어노테이션 재생성 도우미
└── requirements.txt
```

---

## 3. 전처리 파이프라인

`preprocess_with_analyzer.py`는 AirwayOcclusionAnalyzer 결과를 바탕으로 프레임 품질을 평가하고 `OTE / Velum / None`으로 분류합니다.

### 사용 예시
```bash
python preprocess_with_analyzer.py \
  --dataset "D:/.../DISE_DATA(AIHub)" \
  --output processed_dataset
```

### 주요 특징
- ROI 면적·밝기·선명도·조직색 등을 이용한 휴리스틱 필터
- 클래스별 폴더 저장 + `annotations.json`, `dataset_stats.json` 생성
- Velum/OTE 비디오를 분리 처리하여 데이터 불균형 최소화

전처리 규칙을 변경했다면 `processed_dataset`을 정리하고 다시 실행한 뒤,  
`regenerate_annotations.py`로 실제 파일과 어노테이션이 일치하는지 확인하세요.

---

## 4. 학습

`train.py`는 `config` 딕셔너리 기반으로 Trainer를 구성합니다.

```python
config = {
    'model_name': 'resnet50',      # resnet18 / resnet50 / efficientnet_b0 / custom_cnn
    'num_classes': 3,
    'pretrained': True,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'plateau',        # or 'cosine'
    'img_size': 224,
    'annotation_file': 'processed_dataset/annotations.json',
    'data_root': 'processed_dataset',
    'save_dir': 'checkpoints'
}
```

산출물:
- `checkpoints/best_model.pth`
- `training_history.json`, `training_history.png`
- `confusion_matrix.png`, `test_results.json`

> **Tip**: `create_dataloaders`는 클래스별로 비디오 단위 `train/val/test` split을 수행합니다.  
> 특정 클래스(예: Velum) 프레임이 0이면 `train_test_split`에서 ValueError가 발생하므로 전처리 후 통계를 먼저 확인하세요.

---

## 5. 추론

### 단일 이미지
```bash
python inference.py \
  --model checkpoints/best_model.pth \
  --input path/to/image.jpg
```

### 비디오 전체 분석
```bash
python inference.py \
  --model checkpoints/best_model.pth \
  --input path/to/video.mp4 \
  --frame-interval 5 \
  --visualize \
  --output results.json
```

### 세그먼트 다수결
```bash
python inference.py \
  --model checkpoints/best_model.pth \
  --input path/to/video.mp4 \
  --segment-analysis \
  --segment-duration 3.0
```

주요 옵션:
- `--visualize`: 예측 결과를 오버레이한 비디오 생성
- `--frame-interval`: 프레임 샘플링 간격
- `--segment-analysis`: 구간별 다수결 집계

---

## 6. 모델 & 커스터마이징

- **모델 선택**: `resnet18`, `resnet50`, `efficientnet_b0`, `custom_cnn`
- **데이터 증강**: `dataset.py`의 `get_transforms()`에서 수정
- **하이퍼파라미터**: `train.py`의 `config`에서 변경
- **전처리 로직**: `classify_frame()`을 수정하여 휴리스틱 미세 조정
- **어노테이션 재검증**: `regenerate_annotations.py` 실행

---

## 7. 문제 해결 가이드

| 증상 | 원인/대응 |
| --- | --- |
| `ModuleNotFoundError: torch` | 가상환경에 PyTorch 미설치 → `pip install torch torchvision torchaudio ...` |
| `ValueError: n_samples=0` | 특정 클래스 어노테이션 0개 → 전처리 재실행 또는 2클래스 학습 |
| CUDA OOM | `batch_size`, `img_size` 축소 또는 경량 모델 선택 |
| None 프레임 과다 | `preprocess_with_analyzer`의 밝기/ROI/품질 임계값 조정 |
| 프레임/어노테이션 불일치 | `regenerate_annotations.py`로 재생성 |

---

## 8. 모델 다운로드 · 업로드

### 다운로드 (`download_model.py`)
```bash
python download_model.py --list
python download_model.py --model resnet50_ote_velum_v1
python download_model.py --model ... --load    # 즉시 로드
```

Google Drive / Hugging Face / 커스텀 URL도 지원합니다.

### 업로드 (`upload_model.py`)
```bash
python upload_model.py \
  --model checkpoints/best_model.pth \
  --platform gdrive \
  --gdrive-creds credentials.json

python upload_model.py \
  --model checkpoints/best_model.pth \
  --platform huggingface \
  --hf-repo your-username/ote-velum-classifier \
  --hf-token YOUR_HF_TOKEN
```

`--platform both` 옵션으로 두 플랫폼 동시 업로드도 가능합니다.

---

## 9. 참고 Tips

- 원본 비디오는 `.mp4` 권장
- GPU 메모리가 부족하면 `batch_size`를 줄이고 `resnet18`로 교체
- 전처리 로그에 `📹 Found X Velum videos` 메시지가 출력되는지 확인
- `processed_dataset/dataset_stats.json`으로 클래스 분포를 바로 확인 가능

---

## 10. 라이선스

본 프로젝트는 연구용으로 제공됩니다. 결과물을 재배포할 경우 출처를 명시해 주세요.


