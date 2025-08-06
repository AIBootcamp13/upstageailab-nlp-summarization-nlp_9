#!/bin/bash
# =======================================================
# Sweep 대신, 여러 개의 긴 명령어를 순서대로 실행하는 스크립트
# 실행 방법: 터미널에 bash run_experiments.sh 입력
# =======================================================

echo "🚀 실험 1: '챔피언 후보' 레시피로 전체 데이터 훈련 시작..."
# 이전에 만들었던 'flan-t5-large-champion.yaml' 설정을 기본으로 사용
python src/train.py \
  --model_name flan-t5-large-champion


echo "-----------------------------------------------------"
echo "🚀 실험 2: '안정성 위주' 레시피로 전체 데이터 훈련 시작..."
# 기본 large 모델 설정에서, 학습률과 배치 사이즈만 명령어로 바꿔치기
python src/train.py \
  --model_name flan-t5-large \
  model.learning_rate=3e-5 \
  data.batch_size=2 \
  trainer.accumulate_grad_batches=8


echo "🎉 모든 실험이 완료되었습니다."