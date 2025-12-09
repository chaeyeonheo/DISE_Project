# Render 자동 배포 확인 가이드

## Render 자동 배포 작동 방식

### ✅ 자동 배포가 활성화되어 있는 경우

Render는 기본적으로 **GitHub에 푸시하면 자동으로 재배포**됩니다.

**확인 방법:**
1. Render 대시보드 → 서비스 선택
2. "Settings" 탭 클릭
3. "Auto-Deploy" 섹션 확인
   - ✅ **"Yes"** → 자동 배포 활성화
   - ❌ **"No"** → 수동 배포만 가능

### 자동 배포가 작동하지 않는 경우

#### 1. Auto-Deploy가 비활성화된 경우

**해결:**
1. Render 대시보드 → 서비스 → Settings
2. "Auto-Deploy" → "Yes"로 변경
3. "Save Changes" 클릭

#### 2. GitHub 웹훅 문제

**확인:**
1. Render 대시보드 → 서비스 → Settings
2. "GitHub" 섹션 확인
3. 저장소가 제대로 연결되어 있는지 확인

**해결:**
1. "Disconnect" 클릭
2. 다시 "Connect" 클릭
3. GitHub 권한 재승인

#### 3. 브랜치 설정 문제

**확인:**
- Render에서 배포하는 브랜치가 `main`인지 확인
- GitHub에 푸시한 브랜치와 일치하는지 확인

### 수동 배포 방법

자동 배포가 작동하지 않을 때:

1. Render 대시보드 → 서비스 선택
2. "Manual Deploy" 버튼 클릭
3. 배포할 브랜치/커밋 선택
4. "Deploy" 클릭

---

## 현재 배포 상태 확인

### 1. Render 대시보드에서 확인

- **"Events"** 탭: 최근 배포 이력 확인
- **"Logs"** 탭: 실시간 로그 확인
- **"Metrics"** 탭: 서비스 상태 확인

### 2. GitHub 커밋 확인

```bash
# 최근 커밋 확인
git log --oneline -5

# 원격 저장소와 동기화 확인
git status
```

### 3. 배포 로그에서 확인

Render 로그에서 다음을 확인:
- ✅ "Cloning from https://github.com/..."
- ✅ "Checking out commit ..."
- ✅ "Build successful"
- ✅ "Deploying..."

---

## 문제 해결 체크리스트

- [ ] GitHub에 코드가 푸시되었는지 확인
- [ ] Render에서 Auto-Deploy가 "Yes"로 설정되어 있는지 확인
- [ ] Render와 GitHub 저장소가 연결되어 있는지 확인
- [ ] 배포할 브랜치가 올바른지 확인
- [ ] Render 로그에서 에러 메시지 확인

---

## 빠른 해결 방법

**즉시 재배포하려면:**

1. Render 대시보드 접속
2. 서비스 선택
3. **"Manual Deploy"** 클릭
4. **"Deploy latest commit"** 선택
5. **"Deploy"** 클릭

이렇게 하면 최신 코드로 즉시 재배포됩니다!

