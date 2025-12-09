# Render 빠른 배포 가이드 (5분 완성)

가장 간단한 방법으로 Render에 배포하는 단계별 가이드입니다.

---

## 🚀 3단계로 배포하기

### 1단계: GitHub에 코드 업로드

```bash
# 프로젝트 디렉토리에서
git init
git add .
git commit -m "Ready for deployment"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

> ⚠️ **중요**: `.env` 파일은 절대 올리지 마세요! `.gitignore`에 포함되어 있는지 확인하세요.

---

### 2단계: Render에서 Web Service 생성

1. [https://render.com](https://render.com) 접속 및 가입 (GitHub 계정으로)
2. "New +" → "Web Service" 클릭
3. GitHub 저장소 연결 및 선택
4. 다음 설정 입력:

   **기본 설정:**
   - Name: `dise-analyzer` (원하는 이름)
   - Environment: `Python 3`
   - Region: `Singapore` (또는 가장 가까운 지역)
   - Branch: `main`

   **빌드 명령어:**
   ```
   pip install -r requirements.txt && python download_model_from_drive.py
   ```
   
   > 💡 **설명**: 모델 파일을 Google Drive에서 자동으로 다운로드합니다.

   **시작 명령어 (중요!):**
   ```
   gunicorn integrated_app:app
   ```

5. "Advanced" 섹션 열기 → "Environment Variables" 추가:

   | Key | Value |
   |-----|-------|
   | `GEMINI_API_KEY` | `your_gemini_api_key_here` |
   | `PYTHONUNBUFFERED` | `1` |

6. "Create Web Service" 클릭

---

### 3단계: 배포 완료 확인

1. 배포 로그 확인 (자동으로 표시됨)
2. "Your service is live!" 메시지 확인
3. 제공된 URL로 접속 (예: `https://dise-analyzer.onrender.com`)
4. 웹 인터페이스가 정상적으로 표시되는지 확인

---

## ✅ 체크리스트

배포 전 확인사항:

- [ ] `requirements.txt`에 `gunicorn` 포함됨
- [ ] `.env` 파일이 `.gitignore`에 포함됨
- [ ] GitHub에 코드 업로드 완료
- [ ] Render에서 Start Command가 `gunicorn integrated_app:app`로 설정됨
- [ ] `GEMINI_API_KEY` 환경 변수 설정 완료

---

## 🐛 문제 해결

### "Module not found" 오류
→ `requirements.txt`에 모든 패키지가 포함되어 있는지 확인

### "GEMINI_API_KEY not found" 오류
→ Render 대시보드의 Environment Variables에서 확인

### 배포는 성공했지만 페이지가 안 열림
→ 로그 확인 (대시보드의 "Logs" 탭)

---

## 💡 팁

- **무료 플랜**: 15분간 접속이 없으면 서비스가 잠듭니다. 다시 깨어날 때 30초~1분 걸립니다.
- **자동 배포**: GitHub에 푸시하면 자동으로 재배포됩니다.
- **로그 확인**: 문제 발생 시 대시보드의 "Logs" 탭에서 실시간 로그 확인 가능

---

**완료!** 이제 웹에서 접속 가능합니다! 🎉

