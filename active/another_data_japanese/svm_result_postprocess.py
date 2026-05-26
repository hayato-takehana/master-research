from __future__ import annotations

from pathlib import Path
import importlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs" / "another_data_japanese"
DATASET_DIR_NAME = "dcstmnce_edinet_text_only"
DATASET_LABEL = "EDINET-Bench fraud_detection text only"
ENGINE_MODULE_NAME = "active.another_data.svm_result_postprocess"


def configure_engine():
    engine = importlib.import_module(ENGINE_MODULE_NAME)
    engine.OUTPUT_ROOT = OUTPUT_ROOT
    engine.DATASET_DIR_NAME = DATASET_DIR_NAME
    engine.DATASET_LABEL = DATASET_LABEL
    return engine


def main() -> int:
    return configure_engine().main()


if __name__ == "__main__":
    raise SystemExit(main())
