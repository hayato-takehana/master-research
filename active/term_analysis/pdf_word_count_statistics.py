from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths  # noqa: E402

bootstrap_project_paths(PROJECT_ROOT)

from dataset_loader import load_documents  # noqa: E402


# 必要に応じてここを書き換えてください。
REAL_SCAM = False
KEEP_DIGITS = False
PERCENT_MODE = "drop"
STD_DDOF = 0
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "ta" / "word_count_stats" / "pdf_word_count_statistics.csv"


def count_words(document: str) -> int:
    return len(str(document).split())


def build_statistics(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("No documents were loaded.")

    return {
        "平均値": float(np.mean(array)),
        "標準偏差": float(np.std(array, ddof=STD_DDOF)),
        "最小値": float(np.min(array)),
        "中央値": float(np.median(array)),
        "最大値": float(np.max(array)),
    }


def build_summary_dataframe(label1_word_counts: list[int], label0_word_counts: list[int]) -> pd.DataFrame:
    all_word_counts = label1_word_counts + label0_word_counts
    all_stats = build_statistics(all_word_counts)
    label1_stats = build_statistics(label1_word_counts)
    label0_stats = build_statistics(label0_word_counts)

    rows = []
    for statistic_name in ["平均値", "標準偏差", "最小値", "中央値", "最大値"]:
        rows.append(
            {
                "統計量": statistic_name,
                "全データ": all_stats[statistic_name],
                "ラベル1のデータ": label1_stats[statistic_name],
                "ラベル0のデータ": label0_stats[statistic_name],
            }
        )
    return pd.DataFrame(rows, columns=["統計量", "全データ", "ラベル1のデータ", "ラベル0のデータ"])


def main() -> int:
    label1_documents, label0_documents = load_documents(
        real_scam=REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    label1_word_counts = [count_words(document) for document in label1_documents]
    label0_word_counts = [count_words(document) for document in label0_documents]

    summary_df = build_summary_dataframe(label1_word_counts, label0_word_counts)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"saved: {OUTPUT_PATH}")
    print(f"label1 documents: {len(label1_word_counts)}")
    print(f"label0 documents: {len(label0_word_counts)}")
    print(f"all documents: {len(label1_word_counts) + len(label0_word_counts)}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
