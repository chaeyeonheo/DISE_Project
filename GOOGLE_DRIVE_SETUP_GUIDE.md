# Google Drive 공유 설정 가이드

## 문제: 모델 파일 다운로드 실패

에러 메시지:
```
Cannot retrieve the public link of the file. You may need to change
the permission to 'Anyone with the link', or have had many accesses.
```

## 해결 방법

### Step 1: Google Drive에서 파일 공유 설정

1. **Google Drive 접속**
   - https://drive.google.com 접속

2. **모델 파일 찾기**
   - 파일 ID: `1pH9VUsm0sxsdV94ZSNRU5SNFEQbFkgUx`

3. **파일 우클릭**
   - 파일을 우클릭
   - "공유" 또는 "Share" 선택

4. **공유 설정 변경**
   - "링크가 있는 모든 사용자" 또는 "Anyone with the link" 선택
   - **중요**: "편집자", "뷰어", "댓글 작성자" 중 하나 선택
     - 권장: **"뷰어"** (읽기 전용)
   - "완료" 또는 "Done" 클릭

5. **링크 복사**
   - "링크 복사" 클릭
   - 링크 형식: `https://drive.google.com/file/d/1pH9VUsm0sxsdV94ZSNRU5SNFEQbFkgUx/view?usp=sharing`

### Step 2: 공유 설정 확인

**올바른 설정:**
- ✅ "링크가 있는 모든 사용자" 또는 "Anyone with the link"
- ✅ "뷰어" 권한
- ✅ 링크가 활성화되어 있음

**잘못된 설정:**
- ❌ "특정 사용자만" 또는 "Specific people"
- ❌ "링크 없음" 또는 "No link"
- ❌ "편집자" 권한 (불필요)

### Step 3: 직접 다운로드 테스트

브라우저에서 다음 URL로 접속:
```
https://drive.google.com/uc?id=1pH9VUsm0sxsdV94ZSNRU5SNFEQbFkgUx
```

또는:
```
https://drive.google.com/file/d/1pH9VUsm0sxsdV94ZSNRU5SNFEQbFkgUx/view?usp=sharing
```

**예상 결과:**
- ✅ 파일이 다운로드되거나 미리보기 표시
- ✅ 파일 크기가 수백 MB (정상)
- ❌ "액세스 권한이 없습니다" 또는 "Access denied" → 공유 설정 재확인

---

## 대안: 다른 방법

### 방법 1: 직접 다운로드 링크 생성

Google Drive에서:
1. 파일 우클릭 → "링크 가져오기"
2. 링크 형식 변경:
   - 기존: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - 변경: `https://drive.google.com/uc?export=download&id=FILE_ID`

### 방법 2: 다른 클라우드 스토리지 사용

**Dropbox:**
- 공유 링크 생성
- 직접 다운로드 링크 사용

**OneDrive:**
- 공유 링크 생성
- 직접 다운로드 링크 사용

**GitHub Releases:**
- GitHub Releases에 파일 업로드
- 직접 다운로드 링크 사용

---

## 확인 체크리스트

- [ ] Google Drive에서 파일 공유 설정 확인
- [ ] "Anyone with the link" 또는 "링크가 있는 모든 사용자"로 설정
- [ ] 브라우저에서 직접 다운로드 테스트
- [ ] 파일 크기가 수백 MB인지 확인
- [ ] Render 재배포 후 로그 확인

---

## 추가 팁

1. **파일 크기 확인**: 모델 파일이 실제로 수백 MB인지 확인
2. **다른 계정으로 테스트**: 시크릿 모드에서 링크 접근 테스트
3. **Google Drive 제한**: 하루 다운로드 제한이 있을 수 있음 (수백 번 이상)

---

**가장 중요한 것**: Google Drive에서 파일을 **"링크가 있는 모든 사용자"**로 공유하고, **"뷰어"** 권한을 부여해야 합니다!

