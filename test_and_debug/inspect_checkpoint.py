# inspect_checkpoint.py
import torch
import sys

def inspect_ckpt(ckpt_path):
    print(f"🕵️‍♂️ 체크포인트 파일을 분석합니다: {ckpt_path}")
    try:
        # CPU를 사용해서 안전하게 파일을 로드
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # 1. 어휘 사전 크기 확인
        embedding_shape = ckpt['state_dict']['model.base_model.model.shared.weight'].shape
        vocab_size = embedding_shape[0]
        print(f"✅ 어휘 사전 크기 (Vocabulary size): {vocab_size}")

        # 2. 저장된 하이퍼파라미터 확인
        if 'hyper_parameters' in ckpt and 'model_cfg' in ckpt['hyper_parameters']:
            hparams = ckpt['hyper_parameters']['model_cfg']
            model_name = hparams.get('pretrained_model_name_or_path', 'N/A')
            print(f"✅ 저장된 기본 모델 이름: {model_name}")

            if vocab_size == 32128 and model_name == "google/flan-t5-large":
                print("\n🎉🎉🎉 완벽합니다! 이 체크포인트는 'flan-t5-large'와 100% 호환됩니다.")
            else:
                print("\n🔥🔥🔥 경고! 체크포인트 정보가 예상과 다릅니다.")
        else:
            print("⚠️ 체크포인트에 하이퍼파라미터 정보가 없습니다.")
            
    except Exception as e:
        print(f"🚨 체크포인트 분석 중 에러 발생: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python inspect_checkpoint.py <체크포인트_파일_경로>")
    else:
        inspect_ckpt(sys.argv[1])