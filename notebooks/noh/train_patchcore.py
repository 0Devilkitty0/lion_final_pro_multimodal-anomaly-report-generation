import os
from pathlib import Path

# --- [초강수 1: Lightning 콜백 무력화] ---
import lightning.pytorch.callbacks as callbacks
class FakeProgressBar(callbacks.Callback): pass
callbacks.RichProgressBar = FakeProgressBar

# --- [초강수 2: Anomalib 내부 rich 유틸리티 무력화] ---
import anomalib.utils.rich as anomalib_rich
from unittest.mock import MagicMock

# 에러가 발생한 'CacheRichLiveState'를 아무 일도 안 하는 객체로 교체
anomalib_rich.CacheRichLiveState = MagicMock()
# 진행 상황을 추적하는 'safe_track'이 rich를 안 쓰고 그냥 루프만 돌게 교체
anomalib_rich.safe_track = lambda sequence, *args, **kwargs: sequence
# ----------------------------------------------

from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# --- [사용자 설정 구간] ---
DATA_ROOT = Path('/content/drive/Othercomputers/my_notebook/lion_final_pro_multimodal-anomaly-report-generation/dataset/MMAD/MVTec-AD')
CATEGORY = "bottle"
RESULT_DIR = Path('/content/drive/Othercomputers/my_notebook/lion_final_pro_multimodal-anomaly-report-generation/notebooks/noh/results')

def main():
    datamodule = MVTec(
        root=DATA_ROOT,
        category=CATEGORY,
        train_batch_size=32,
        eval_batch_size=32,
        image_size=(256, 256),
        num_workers=2
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"]
    )

    engine = Engine(
        task="segmentation",
        default_root_dir=RESULT_DIR / CATEGORY,
        enable_progress_bar=False 
    )

    print(f"🚀 [{CATEGORY}] 내부 유틸리티까지 모두 패치했습니다. 분석 시작...")

    try:
        engine.fit(model=model, datamodule=datamodule)
        results = engine.test(model=model, datamodule=datamodule)
        
        print("\n" + "="*50)
        print(f"✅ {CATEGORY} 최종 분석 결과:")
        print(results)
        print("="*50)
        
    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()