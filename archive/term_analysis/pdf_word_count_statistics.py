from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
TEST_OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "ta" / "word_count_stats" / "pdf_word_count_label_test.csv"
HISTOGRAM_PATH = PROJECT_ROOT / "data" / "outputs" / "ta" / "word_count_stats" / "pdf_word_count_histogram.png"
BOXPLOT_PATH = PROJECT_ROOT / "data" / "outputs" / "ta" / "word_count_stats" / "pdf_word_count_boxplot.png"
LOG_HISTOGRAM_PATH = PROJECT_ROOT / "data" / "outputs" / "ta" / "word_count_stats" / "pdf_word_count_histogram_log1p.png"
LOG_BOXPLOT_PATH = PROJECT_ROOT / "data" / "outputs" / "ta" / "word_count_stats" / "pdf_word_count_boxplot_log1p.png"
ALPHA = 0.05
PERMUTATION_RESAMPLES = 100_000
RANDOM_SEED = 0


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


def mean_difference(label1_values: np.ndarray, label0_values: np.ndarray) -> float:
    return float(np.mean(label1_values) - np.mean(label0_values))


def approximate_permutation_test_mean_difference(
    label1_values: np.ndarray,
    label0_values: np.ndarray,
    *,
    n_resamples: int,
    random_seed: int,
) -> tuple[float, float]:
    observed_statistic = mean_difference(label1_values, label0_values)
    combined_values = np.concatenate([label1_values, label0_values])
    label1_size = label1_values.size
    rng = np.random.default_rng(random_seed)
    extreme_count = 0

    for _ in range(n_resamples):
        permuted_values = rng.permutation(combined_values)
        permuted_label1 = permuted_values[:label1_size]
        permuted_label0 = permuted_values[label1_size:]
        permuted_statistic = mean_difference(permuted_label1, permuted_label0)
        if abs(permuted_statistic) >= abs(observed_statistic):
            extreme_count += 1

    p_value = (extreme_count + 1) / (n_resamples + 1)
    return observed_statistic, float(p_value)


def build_label_difference_test_dataframe(label1_word_counts: list[int], label0_word_counts: list[int]) -> pd.DataFrame:
    label1_array = np.asarray(label1_word_counts, dtype=float)
    label0_array = np.asarray(label0_word_counts, dtype=float)
    if label1_array.size < 2 or label0_array.size < 2:
        raise ValueError("Label difference tests require at least two documents in each label.")

    log_label1_array = np.log1p(label1_array)
    log_label0_array = np.log1p(label0_array)
    label1_mean = float(np.mean(label1_array))
    label0_mean = float(np.mean(label0_array))
    label1_median = float(np.median(label1_array))
    label0_median = float(np.median(label0_array))
    log_label1_mean = float(np.mean(log_label1_array))
    log_label0_mean = float(np.mean(log_label0_array))

    welch_result = stats.ttest_ind(label1_array, label0_array, equal_var=False, alternative="two-sided")
    log_welch_result = stats.ttest_ind(log_label1_array, log_label0_array, equal_var=False, alternative="two-sided")
    mann_whitney_result = stats.mannwhitneyu(label1_array, label0_array, alternative="two-sided")
    permutation_statistic, permutation_p_value = approximate_permutation_test_mean_difference(
        label1_array,
        label0_array,
        n_resamples=PERMUTATION_RESAMPLES,
        random_seed=RANDOM_SEED,
    )
    log_permutation_statistic, log_permutation_p_value = approximate_permutation_test_mean_difference(
        log_label1_array,
        log_label0_array,
        n_resamples=PERMUTATION_RESAMPLES,
        random_seed=RANDOM_SEED,
    )

    common_values = {
        "ラベル1件数": int(label1_array.size),
        "ラベル0件数": int(label0_array.size),
        "ラベル1平均": label1_mean,
        "ラベル0平均": label0_mean,
        "平均差（ラベル1-ラベル0）": label1_mean - label0_mean,
        "ラベル1中央値": label1_median,
        "ラベル0中央値": label0_median,
        "中央値差（ラベル1-ラベル0）": label1_median - label0_median,
        "ラベル1_log1p平均": log_label1_mean,
        "ラベル0_log1p平均": log_label0_mean,
        "log1p平均差（ラベル1-ラベル0）": log_label1_mean - log_label0_mean,
        "有意水準": ALPHA,
    }

    rows = [
        {
            "検定": "Welchのt検定（両側）",
            "対象": "平均単語数",
            "帰無仮説": "ラベル1とラベル0の平均単語数は等しい",
            **common_values,
            "t値": float(welch_result.statistic),
            "U値": None,
            "置換統計量": None,
            "置換回数": None,
            "乱数シード": None,
            "p値": float(welch_result.pvalue),
            "判定": "差あり" if float(welch_result.pvalue) < ALPHA else "差ありとはいえない",
        },
        {
            "検定": "Welchのt検定（両側）",
            "対象": "log1p平均単語数",
            "帰無仮説": "ラベル1とラベル0のlog1p平均単語数は等しい",
            **common_values,
            "t値": float(log_welch_result.statistic),
            "U値": None,
            "置換統計量": None,
            "置換回数": None,
            "乱数シード": None,
            "p値": float(log_welch_result.pvalue),
            "判定": "差あり" if float(log_welch_result.pvalue) < ALPHA else "差ありとはいえない",
        },
        {
            "検定": "Mann-Whitney U検定（両側）",
            "対象": "単語数の分布・順位",
            "帰無仮説": "ラベル1とラベル0の単語数分布は同じ",
            **common_values,
            "t値": None,
            "U値": float(mann_whitney_result.statistic),
            "置換統計量": None,
            "置換回数": None,
            "乱数シード": None,
            "p値": float(mann_whitney_result.pvalue),
            "判定": "差あり" if float(mann_whitney_result.pvalue) < ALPHA else "差ありとはいえない",
        },
        {
            "検定": "置換検定（両側・近似）",
            "対象": "平均単語数",
            "帰無仮説": "ラベルと単語数は独立であり、平均差はラベル割当に依存しない",
            **common_values,
            "t値": None,
            "U値": None,
            "置換統計量": permutation_statistic,
            "置換回数": PERMUTATION_RESAMPLES,
            "乱数シード": RANDOM_SEED,
            "p値": permutation_p_value,
            "判定": "差あり" if permutation_p_value < ALPHA else "差ありとはいえない",
        },
        {
            "検定": "置換検定（両側・近似）",
            "対象": "log1p平均単語数",
            "帰無仮説": "ラベルとlog1p単語数は独立であり、平均差はラベル割当に依存しない",
            **common_values,
            "t値": None,
            "U値": None,
            "置換統計量": log_permutation_statistic,
            "置換回数": PERMUTATION_RESAMPLES,
            "乱数シード": RANDOM_SEED,
            "p値": log_permutation_p_value,
            "判定": "差あり" if log_permutation_p_value < ALPHA else "差ありとはいえない",
        },
    ]
    return pd.DataFrame(rows)


def _plot_histogram(label1_values: np.ndarray, label0_values: np.ndarray, output_path: Path, xlabel: str) -> None:
    combined_values = np.concatenate([label1_values, label0_values])
    bins = np.histogram_bin_edges(combined_values, bins=30)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(label1_values, bins=bins, alpha=0.65, label=f"Label 1 (n={label1_values.size})", color="#2f6f9f")
    ax.hist(label0_values, bins=bins, alpha=0.65, label=f"Label 0 (n={label0_values.size})", color="#c75c5c")
    ax.set_title("Word Count Histogram by Label")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_boxplot(label1_values: np.ndarray, label0_values: np.ndarray, output_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    boxplot = ax.boxplot(
        [label1_values, label0_values],
        tick_labels=["Label 1", "Label 0"],
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(boxplot["boxes"], ["#2f6f9f", "#c75c5c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_title("Word Count Boxplot by Label")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_distribution_plots(label1_word_counts: list[int], label0_word_counts: list[int]) -> list[Path]:
    label1_array = np.asarray(label1_word_counts, dtype=float)
    label0_array = np.asarray(label0_word_counts, dtype=float)
    if label1_array.size == 0 or label0_array.size == 0:
        raise ValueError("Distribution plots require documents in both labels.")

    HISTOGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    _plot_histogram(label1_array, label0_array, HISTOGRAM_PATH, "Word count")
    _plot_boxplot(label1_array, label0_array, BOXPLOT_PATH, "Word count")

    log_label1_array = np.log1p(label1_array)
    log_label0_array = np.log1p(label0_array)
    _plot_histogram(log_label1_array, log_label0_array, LOG_HISTOGRAM_PATH, "log1p(word count)")
    _plot_boxplot(log_label1_array, log_label0_array, LOG_BOXPLOT_PATH, "log1p(word count)")

    return [HISTOGRAM_PATH, BOXPLOT_PATH, LOG_HISTOGRAM_PATH, LOG_BOXPLOT_PATH]


def main() -> int:
    label1_documents, label0_documents = load_documents(
        real_scam=REAL_SCAM,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    label1_word_counts = [count_words(document) for document in label1_documents]
    label0_word_counts = [count_words(document) for document in label0_documents]

    summary_df = build_summary_dataframe(label1_word_counts, label0_word_counts)
    test_df = build_label_difference_test_dataframe(label1_word_counts, label0_word_counts)
    plot_paths = save_distribution_plots(label1_word_counts, label0_word_counts)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"saved: {OUTPUT_PATH}")
    print(f"saved: {TEST_OUTPUT_PATH}")
    for plot_path in plot_paths:
        print(f"saved: {plot_path}")
    print(f"label1 documents: {len(label1_word_counts)}")
    print(f"label0 documents: {len(label0_word_counts)}")
    print(f"all documents: {len(label1_word_counts) + len(label0_word_counts)}")
    print(summary_df.to_string(index=False))
    print(test_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
