# Google Drive 폴더에서 파일 다운로드 가이드

## 문제 발견

제공된 링크가 **폴더 링크**입니다:
```
https://drive.google.com/drive/u/0/folders/1pH9VUsm0sxsdV94ZSNRU5SNFEQbFkgUx
```

이것은 파일이 아니라 **폴더**입니다. 폴더 안에 모델 파일(`best_model.pth`)이 있을 것입니다.

## 해결 방법

### 방법 1: 폴더 안의 실제 파일 ID 찾기 (권장)

1. **폴더 접속**
   - 제공된 링크로 폴더 접속
   - `https://drive.google.com/drive/u/0/folders/1pH9VUsm0sxsdV94ZSNRU5SNFEQbFkgUx`

2. **모델 파일 찾기**
   - 폴더 안에서 `best_model.pth` 파일 찾기

3. **파일 우클릭 → 링크 가져오기**
   - 파일을 우클릭
   - "링크 가져오기" 또는 "Get link" 선택
   - 링크 형식: `https://drive.google.com/file/d/[실제_파일_ID]/view?usp=sharing`

4. **파일 ID 추출**
   - 링크에서 파일 ID 추출
   - 예: `https://drive.google.com/file/d/ABC123XYZ456/view?usp=sharing`
   - 파일 ID: `ABC123XYZ456`

5. **코드에 파일 ID 업데이트**
   - `download_model_from_drive.py`의 `GOOGLE_DRIVE_FILE_ID` 변경

### 방법 2: 폴더 공유 설정

1. **폴더 공유 설정**
   - 폴더 우클릭 → "공유" 또는 "Share"
   - "링크가 있는 모든 사용자" 또는 "Anyone with the link" 선택
   - "뷰어" 권한 부여

2. **폴더 안 파일도 공유 확인**
   - 폴더 안의 각 파일도 개별적으로 공유 설정 확인

### 방법 3: 폴더에서 파일 찾기 (자동화)

폴더 ID를 사용하여 폴더 안의 파일을 찾는 방법도 있지만, 복잡합니다.

---

## 가장 간단한 해결책

**폴더 안의 실제 파일 링크를 제공해주세요:**

1. 폴더 접속
2. `best_model.pth` 파일 찾기
3. 파일 우클릭 → "링크 가져오기"
4. 그 링크를 알려주세요

예시:
```
https://drive.google.com/file/d/실제파일ID여기/view?usp=sharing
```

그러면 파일 ID를 추출하여 코드에 업데이트하겠습니다!

