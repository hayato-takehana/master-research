from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data" / "outputs" / "ta" / "dcstmnce"
OUTPUT_FILENAME = "organized_metrics.csv"
OUTPUT_COLUMNS = [
    "外側の層",
    "特徴数",
    "ハイパーパラメータC",
    "ガンマ",
    "accuracy",
    "recall",
    "prescision",
    "f1_score",
]


def format_number(value, decimals=None) -> str:
    if pd.isna(value) or value == "":
        return ""
    if decimals is None:
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), "g")
    formatted = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def build_output_dataframe(folds_df: pd.DataFrame) -> pd.DataFrame:
    output_df = pd.DataFrame(
        {
            "外側の層": folds_df["outer_fold"],
            "特徴数": folds_df["selected_term_count"].map(format_number),
            "ハイパーパラメータC": folds_df["c"].map(format_number),
            "ガンマ": folds_df["gamma"].map(format_number),
            "accuracy": folds_df["accuracy"].map(lambda value: format_number(value, 3)),
            "recall": folds_df["recall"].map(lambda value: format_number(value, 3)),
            "prescision": folds_df["precision"].map(lambda value: format_number(value, 3)),
            "f1_score": folds_df["f1_score"].map(lambda value: format_number(value, 3)),
        }
    )

    average_row = {
        "外側の層": "平均",
        "特徴数": format_number(folds_df["selected_term_count"].mean(), 1),
        "ハイパーパラメータC": "",
        "ガンマ": "",
        "accuracy": format_number(folds_df["accuracy"].mean(), 3),
        "recall": format_number(folds_df["recall"].mean(), 3),
        "prescision": format_number(folds_df["precision"].mean(), 3),
        "f1_score": format_number(folds_df["f1_score"].mean(), 3),
    }

    return pd.concat([output_df, pd.DataFrame([average_row])], ignore_index=True)[OUTPUT_COLUMNS]


def save_organized_csv(source_path: Path) -> Path:
    folds_df = pd.read_csv(source_path)
    output_df = build_output_dataframe(folds_df)
    output_path = source_path.with_name(OUTPUT_FILENAME)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    svm_fold_paths = sorted(SOURCE_ROOT.rglob("svm_folds.csv"))
    if not svm_fold_paths:
        raise FileNotFoundError(f"No svm_folds.csv found under: {SOURCE_ROOT}")

    saved_paths = [save_organized_csv(path) for path in svm_fold_paths]

    print(f"source root: {SOURCE_ROOT}")
    print(f"saved files: {len(saved_paths)}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
