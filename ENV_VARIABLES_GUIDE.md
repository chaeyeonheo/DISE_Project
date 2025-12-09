# 환경 변수 설정 가이드: .env 파일을 Render에 배포하기

## 🔒 핵심 원칙

**절대 하지 말아야 할 것:**
- ❌ `.env` 파일을 GitHub에 커밋
- ❌ API 키를 코드에 하드코딩
- ❌ 환경 변수를 GitHub에 공개

**반드시 해야 할 것:**
- ✅ `.env` 파일을 `.gitignore`에 추가
- ✅ Render 대시보드에서 환경 변수 설정
- ✅ 코드에서는 `os.getenv()` 사용

---

## Step 1: .env 파일 확인

### 1-1. 현재 .env 파일 내용 확인

로컬 프로젝트의 `.env` 파일을 열어서 어떤 환경 변수가 있는지 확인합니다:

```env
GEMINI_API_KEY=your_actual_api_key_here
DATABASE_URL=postgresql://...
SECRET_KEY=your_secret_key
```

> ⚠️ **중요**: 실제 API 키 값은 절대 공유하거나 GitHub에 올리지 마세요!

### 1-2. .gitignore 확인

`.gitignore` 파일에 `.env`가 포함되어 있는지 확인:

```gitignore
# Environment Variables
.env
.env.local
.env.*.local
```

확인 방법:
```bash
cat .gitignore | grep .env
```

---

## Step 2: 코드에서 환경 변수 사용 확인

### 2-1. 현재 코드 확인

`integrated_app.py`에서 환경 변수를 어떻게 사용하는지 확인:

```python
from dotenv import load_dotenv
import os

# .env 파일 로드 (로컬 환경에서만)
load_dotenv(override=True)

# 환경 변수에서 API 키 가져오기
gemini_api_key = os.getenv('GEMINI_API_KEY', '').strip()

if not gemini_api_key:
    print("⚠️ 경고: GEMINI_API_KEY가 환경 변수에서 로드되지 않았습니다.")
else:
    print(f"✅ GEMINI_API_KEY 로드 완료")
    
app.config['GEMINI_API_KEY'] = gemini_api_key
```

### 2-2. 코드가 올바른지 확인

✅ **올바른 방법:**
```python
api_key = os.getenv('GEMINI_API_KEY')  # ✅ 환경 변수에서 가져오기
```

❌ **잘못된 방법:**
```python
api_key = "sk-1234567890abcdef"  # ❌ 하드코딩 (절대 안 됨!)
```

---

## Step 3: Render에서 환경 변수 설정

### 3-1. Render 대시보드 접속

1. [Render.com](https://render.com) 로그인
2. 배포한 서비스 선택 (또는 새로 생성)

### 3-2. Environment Variables 섹션 찾기

1. 서비스 대시보드에서 **"Environment"** 탭 클릭
2. 또는 왼쪽 메뉴에서 **"Environment"** 선택

### 3-3. 환경 변수 추가

**방법 1: 하나씩 추가**

1. **"Add Environment Variable"** 버튼 클릭
2. **Key** 입력: `GEMINI_API_KEY`
3. **Value** 입력: `.env` 파일에 있던 실제 API 키 값
4. **"Save Changes"** 클릭

**방법 2: 여러 개 한 번에 추가**

1. **"Add Environment Variable"** 버튼을 여러 번 클릭
2. 각각의 Key-Value 쌍 입력
3. 모두 입력 후 **"Save Changes"** 클릭

### 3-4. 환경 변수 목록 예시

프로젝트에 따라 다음 환경 변수들을 추가할 수 있습니다:

| Key | Value | 설명 |
|-----|-------|------|
| `GEMINI_API_KEY` | `your_actual_key` | Gemini API 키 |
| `PYTHONUNBUFFERED` | `1` | Python 출력 버퍼링 비활성화 |
| `FLASK_ENV` | `production` | Flask 환경 설정 |
| `DATABASE_URL` | `postgresql://...` | 데이터베이스 URL (필요시) |

> 💡 **팁**: `PYTHONUNBUFFERED=1`은 로그를 실시간으로 보기 위해 권장됩니다.

---

## Step 4: 환경 변수 확인

### 4-1. 배포 후 로그 확인

배포가 완료되면 로그에서 환경 변수가 제대로 로드되었는지 확인:

**성공 로그 예시:**
```
✅ GEMINI_API_KEY 로드 완료 (길이: 39자, 시작: AIzaSyAbc...)
🚀 Flask 서버 시작
```

**실패 로그 예시:**
```
⚠️ 경고: GEMINI_API_KEY가 환경 변수에서 로드되지 않았습니다.
```

### 4-2. 환경 변수 값 확인 (디버깅용)

코드에서 환경 변수가 제대로 로드되는지 확인하려면:

```python
# 디버깅용 (배포 후 제거 권장)
gemini_api_key = os.getenv('GEMINI_API_KEY', '')
if gemini_api_key:
    # API 키의 일부만 출력 (보안)
    print(f"API Key loaded: {gemini_api_key[:10]}...")
else:
    print("API Key not found!")
```

> ⚠️ **주의**: 전체 API 키를 로그에 출력하지 마세요!

---

## Step 5: 환경 변수 업데이트

### 5-1. API 키 변경 시

1. Render 대시보드 → Environment 탭
2. 해당 환경 변수 찾기
3. **"Edit"** 클릭
4. 새 값 입력
5. **"Save Changes"** 클릭
6. 서비스가 자동으로 재배포됨

### 5-2. 새 환경 변수 추가 시

1. Environment 탭에서 **"Add Environment Variable"** 클릭
2. Key-Value 입력
3. **"Save Changes"** 클릭
4. 자동 재배포

---

## 보안 모범 사례

### ✅ DO (해야 할 것)

1. **환경 변수 사용**
   ```python
   api_key = os.getenv('API_KEY')
   ```

2. **기본값 제공**
   ```python
   api_key = os.getenv('API_KEY', '')
   if not api_key:
       raise ValueError("API_KEY is required")
   ```

3. **.gitignore에 .env 추가**
   ```gitignore
   .env
   .env.local
   ```

4. **Render에서만 환경 변수 설정**
   - 로컬: `.env` 파일 사용
   - 배포: Render 대시보드에서 설정

### ❌ DON'T (하지 말아야 할 것)

1. **하드코딩**
   ```python
   api_key = "sk-1234567890"  # ❌ 절대 안 됨!
   ```

2. **GitHub에 .env 커밋**
   ```bash
   git add .env  # ❌ 절대 안 됨!
   git commit -m "Add env file"
   ```

3. **코드에 주석으로 API 키 작성**
   ```python
   # API_KEY = "sk-1234567890"  # ❌ 이것도 안 됨!
   ```

4. **공개 저장소에 환경 변수 공유**
   - 이슈, PR, 댓글에 API 키 작성 ❌

---

## 문제 해결

### 문제 1: "API Key not found" 오류

**증상:**
```
⚠️ 경고: GEMINI_API_KEY가 환경 변수에서 로드되지 않았습니다.
```

**해결:**
1. Render 대시보드 → Environment 탭 확인
2. `GEMINI_API_KEY`가 있는지 확인
3. Value가 올바르게 입력되었는지 확인 (공백 없이)
4. **"Save Changes"** 클릭
5. 서비스 재배포 대기

---

### 문제 2: 환경 변수는 설정했는데 여전히 오류

**해결:**
1. 환경 변수 이름 확인 (대소문자 구분)
   - `GEMINI_API_KEY` ✅
   - `gemini_api_key` ❌ (다를 수 있음)
2. 코드에서 사용하는 이름과 일치하는지 확인
3. 재배포 후 로그 확인

---

### 문제 3: .env 파일이 실수로 GitHub에 올라갔을 때

**긴급 조치:**
1. **즉시 API 키 변경**
   - 해당 서비스에서 새 API 키 발급
   - Render에서 환경 변수 업데이트

2. **Git 히스토리에서 제거**
   ```bash
   # .env 파일을 Git에서 제거 (히스토리 포함)
   git rm --cached .env
   git commit -m "Remove .env file from repository"
   git push
   ```

3. **.gitignore 확인**
   ```bash
   echo ".env" >> .gitignore
   git add .gitignore
   git commit -m "Add .env to gitignore"
   git push
   ```

---

## 체크리스트

배포 전 확인:

- [ ] `.env` 파일이 `.gitignore`에 포함됨
- [ ] 코드에서 `os.getenv()` 사용 (하드코딩 없음)
- [ ] `.env` 파일이 GitHub에 올라가지 않음
- [ ] Render 대시보드에서 모든 환경 변수 설정 완료
- [ ] 환경 변수 이름이 코드와 일치함
- [ ] 배포 후 로그에서 환경 변수 로드 확인

---

## 요약

1. **로컬 개발**: `.env` 파일 사용
2. **배포 환경**: Render 대시보드에서 환경 변수 설정
3. **코드**: `os.getenv()`로 환경 변수 읽기
4. **보안**: 절대 GitHub에 `.env` 커밋하지 않기

**핵심**: `.env` 파일은 로컬에서만 사용하고, 배포 환경에서는 Render 대시보드에서 직접 설정합니다! 🔒

