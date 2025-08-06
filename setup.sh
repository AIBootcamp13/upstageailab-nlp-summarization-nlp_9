```bash
#!/bin/bash

# =======================================================
# 새로운 서버 생성 후, 이 스크립트만 실행하면 모든 세팅이 완료됩니다.
# 실행 방법: 터미널에 bash setup.sh 입력
# =======================================================

echo "🚀 프로젝트 초기 설정을 시작합니다..."

# 0. oh-my-zsh 설치
echo "🔧 oh-my-zsh 설치 중..."
sudo apt update
sudo apt install -y zsh git curl

echo "✅ 기본 패키지 설치 완료"

# oh-my-zsh 설치 (zsh 설정은 기본값으로 진행)
echo "📦 oh-my-zsh 설치 진행..."
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

echo "✅ oh-my-zsh 설치 완료"

# 1. GitHub 리포지토리 클론
echo "📁 GitHub 리포지토리 클론 중..."
git clone https://github.com/welovecherry/NLP_jungmin_fork.git
cd NLP_jungmin_fork
git checkout jungmin-exp

echo "✅ 1/5: GitHub 리포지토리 준비 완료"

# 2. 파이썬 가상 환경 설정
echo "🐍 가상 환경 생성 및 활성화 중..."
python3 -m venv venv
source venv/bin/activate

echo "✅ 2/5: 파이썬 가상 환경 활성화 완료"

# 3. 필요한 라이브러리 설치
echo "📦 라이브러리 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ 3/5: 모든 라이브러리 설치 완료"

# 4. 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
# export WANDB_API_KEY="your-wandb-api-key"

echo "✅ 4/5: 환경 변수 설정 완료"

# 5. 데이터 복사 (로컬 → 서버 수동 복사 필요)
echo "📁 데이터는 로컬에서 수동으로 복사하세요:"
echo "   로컬 경로: /Users/jungminhong/Documents/nlp_jungmin_fork/nlp/data"
echo "   서버 경로: ~/NLP_jungmin_fork/data"

echo "✅ 5/5: 데이터 준비 완료 (수동 복사 안내 출력)"

echo "🎉 모든 설정이 완료되었습니다! 이제 훈련을 시작할 수 있습니다."
```