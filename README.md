# DISE 기도 폐색 자동 분석 서비스
수면 내시경(DISE) 영상을 업로드하면 OTE/Velum 분류와 기도 폐색을 자동 분석하고, 결과 기반 VQA를 제공하는 간단한 Flask 서비스입니다.

## 사용법
```bash
pip install -r requirements.txt
# (선택) 가중치 미리 다운로드
python download_model_from_drive.py

python integrated_app.py
# http://localhost:5000 접속 후 영상 업로드
```

## 필요 정보
- 필수: `GEMINI_API_KEY` 환경 변수 (VQA/요약용)
- 모델 가중치: `ote_velum_classification_final/checkpoints/best_model.pth`
  - 없으면 서버가 자동 다운로드하며, 수동은 `python download_model_from_drive.py` (루트) 실행하면 됩니다.  
    Drive: https://drive.google.com/file/d/161GXpszELcLSc6ACP1Uzdpz26a8jXYDK/view?usp=drive_link

## 학습/전처리 워크플로우
> 데이터가 있을 때 분류 모델을 직접 학습하고 싶다면 아래 순서로 진행하세요.

```bash
# 0) 분류 모듈 의존성 설치
cd ote_velum_classification_final
pip install -r requirements.txt

# 1) 원본 비디오 → 프레임 전처리 및 라벨 생성
python preprocess_with_analyzer.py \
  --dataset "D:/path/to/raw_videos" \
  --output processed_dataset

# 2) 학습
python train.py \
  --config configs/resnet50.yaml   # (옵션) 인자 없이 실행하면 내부 기본 config 사용

# 3) 추론/평가
python inference.py \
  --model checkpoints/best_model.pth \
  --input path/to/video.mp4 \
  --frame-interval 5 \
  --visualize
```

- 추가 유틸: `preprocess_videos.py`(간단 전처리), `regenerate_annotations.py`(어노테이션 재검증), `plot_accuracy.py`/`plot_from_checkpoint.py`(학습 로그 시각화), `check_test_data.py`(데이터 점검).

## 주요 파일
- `integrated_app.py` : 업로드/분석/VQA 엔드포인트
- `integrated_analyzer.py` : 프레임 추출·분류·ROI/이벤트 분석
- `integrated_report_generator.py` : 리포트/차트/VQA 컨텍스트 생성
- `templates/index.html` : 업로드 UI
- `ote_velum_classification_final/` : 분류 모델 코드 및 체크포인트 위치



