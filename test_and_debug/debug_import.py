# debug_import.py

print("--- [DEBUG] Import Test Start ---")

try:
    # 우리가 문제를 겪고 있는 바로 그 import 구문을 실행해봅니다.
    # 여기서는 상대 경로가 아닌, 절대 경로로 테스트합니다.
    from src.data_module import SummaryDataModule
    
    print("\n✅ [SUCCESS] 'SummaryDataModule'을 성공적으로 불러왔습니다!")
    print("   - 타입:", type(SummaryDataModule))
    print("   - 모듈:", SummaryDataModule.__module__)

except ImportError as e:
    print("\n❌ [FAIL] ImportError가 발생했습니다.")
    print("   - 에러 메시지:", e)

except Exception as e:
    print("\n❌ [FAIL] 예상치 못한 다른 에러가 발생했습니다.")
    print("   - 에러 타입:", type(e).__name__)
    print("   - 에러 메시지:", e)

finally:
    print("\n--- [DEBUG] Import Test End ---")