from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import os
import sys
import time
import warnings


def _bootstrap_project_root() -> Path:
    """`active` と `archive` を含むプロジェクトルートを探索して import パスへ追加する。"""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths, redirect_relative_outputs

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
SAVE_DIR = PROJECT_ROOT / "data" / "outputs" / "ta" / "dcstmnce"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
redirect_relative_outputs(SAVE_DIR)

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from dataset_loader import load_documents
from text_vectorizer import Tf_idf

try:
    import shap
except ImportError:
    shap = None


ABS_SCORE_THRESHOLDS = [0.07, 0.08, 0.09]
MIN_LEN = 0
MIN_DF = 0.0
USE_STEMMING = True
KEEP_DIGITS = False
PERCENT_MODE = "drop"

LINEAR_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
RBF_GAMMA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
MARGIN_THRESHOLD = 0.9990
N_SPLITS = 10
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TOL = 1e-10
CANDIDATE_PROGRESS_INTERVAL = 10
LINEAR_ANALYSIS_TOP_K = 20
RBF_SHAP_BACKGROUND_SIZE = 20
RBF_SHAP_NSAMPLES = 100
RBF_ANALYSIS_TOP_K = 20
RBF_SHAP_L1_REG = "num_features(20)"

FEATURE_MODE_DIR_NAMES = {
    "binary": "bin",
    "count": "cnt",
}
KERNEL_DIR_NAMES = {
    "linear": "lin",
    "rbf": "rbf",
}
FIXED_DIR_NAME = "fix"
TUNED_DIR_NAME = "tun"
THRESHOLD_COUNT_DIR_NAME = "tcnt"
METRICS_DIR_NAME = "m"
FEATURE_OUTPUT_DIR_NAME = "feat"
MISCLASSIFIED_OUTPUT_DIR_NAME = "mis"
ERROR_ANALYSIS_OUTPUT_DIR_NAME = "err"

FIXED_THRESHOLD_SUMMARY_COLUMNS = [
    "score_threshold",
    "feature_mode",
    "kernel",
    "mean_selected_term_count",
    "selected_term_counts",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
    "valid_param_count",
    "dropped_param_count",
]
FIXED_THRESHOLD_BEST_RESULT_COLUMNS = [
    "score_threshold",
    "feature_mode",
    "kernel",
    "mean_selected_term_count",
    "selected_term_counts",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
]
TUNED_THRESHOLD_SUMMARY_COLUMNS = [
    "candidate_score_thresholds",
    "feature_mode",
    "kernel",
    "selected_score_thresholds",
    "mean_selected_term_count",
    "selected_term_counts",
    "mean_accuracy",
    "mean_recall",
    "mean_precision",
    "mean_f1",
    "best_cs",
    "best_gammas",
    "valid_param_count",
    "dropped_param_count",
]


def prompt_choice(prompt_label, option_lines, alias_to_value, default_input):
    """起動時の選択肢入力を受け取り、正規化した値を返す。"""
    print(prompt_label)
    for option_line in option_lines:
        print(option_line)

    while True:
        raw = input(f"選択してください [{default_input}]: ").strip().lower()
        if raw == "":
            raw = default_input.lower()
        if raw in alias_to_value:
            return alias_to_value[raw]
        print("入力が正しくありません。候補の番号または名前で入力してください。")


def prompt_experiment_configuration():
    """実行条件を起動時に対話入力で受け取る。"""
    kernel_mode = prompt_choice(
        "\n1. 実行するSVMを選択してください。",
        [
            "  1: 線形SVMのみ",
            "  2: RBF非線形SVMのみ",
            "  3: 両方",
        ],
        {
            "1": "linear",
            "linear": "linear",
            "lin": "linear",
            "2": "rbf",
            "rbf": "rbf",
            "nonlinear": "rbf",
            "3": "both",
            "both": "both",
            "all": "both",
        },
        default_input="3",
    )

    feature_mode = prompt_choice(
        "\n2. 使う特徴量を選択してください。",
        [
            "  1: 出現頻度のみ",
            "  2: バイナリー変数のみ",
            "  3: 両方",
        ],
        {
            "1": "count",
            "count": "count",
            "frequency": "count",
            "2": "binary",
            "binary": "binary",
            "bin": "binary",
            "3": "both",
            "both": "both",
            "all": "both",
        },
        default_input="3",
    )

    threshold_mode = prompt_choice(
        "\n3. スコア閾値の扱いを選択してください。",
        [
            "  1: 固定閾値ごとに10分割を実行する",
            "  2: スコア閾値をハイパーパラメータとして扱う",
            "  3: 両方",
        ],
        {
            "1": "fixed",
            "fixed": "fixed",
            "per_threshold": "fixed",
            "2": "tuned",
            "tuned": "tuned",
            "hyperparameter": "tuned",
            "3": "both",
            "both": "both",
            "all": "both",
        },
        default_input="3",
    )

    return {
        "kernel_mode": kernel_mode,
        "feature_mode": feature_mode,
        "threshold_mode": threshold_mode,
    }


def format_user_selection(value, label_map):
    return label_map[value]


def print_experiment_configuration(config):
    """起動時に選ばれた実行条件を表示する。"""
    kernel_label = format_user_selection(
        config["kernel_mode"],
        {
            "linear": "線形SVMのみ",
            "rbf": "RBF非線形SVMのみ",
            "both": "線形SVM + RBF非線形SVM",
        },
    )
    feature_label = format_user_selection(
        config["feature_mode"],
        {
            "count": "出現頻度のみ",
            "binary": "バイナリー変数のみ",
            "both": "出現頻度 + バイナリー変数",
        },
    )
    threshold_label = format_user_selection(
        config["threshold_mode"],
        {
            "fixed": "固定閾値ごとの10分割のみ",
            "tuned": "閾値をハイパーパラメータとして扱う方式のみ",
            "both": "固定閾値方式 + 閾値チューニング方式",
        },
    )

    log_progress("\n[config] selected runtime options")
    log_progress(f"[config] svm mode      : {kernel_label}")
    log_progress(f"[config] feature mode  : {feature_label}")
    log_progress(f"[config] threshold mode: {threshold_label}")


def load_corpus():
    """文書本文・ラベル・文書IDを読み込み、後段で扱いやすい配列へまとめる。"""
    documents_1, documents_0 = load_documents(
        real_scam=False,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    documents = np.array(documents_1 + documents_0, dtype=object)
    labels = np.array([1] * len(documents_1) + [0] * len(documents_0), dtype=int)
    doc_ids = np.arange(len(documents), dtype=int)
    return documents, labels, doc_ids


def log_progress(message):
    """長時間処理の進捗を即時表示する。"""
    print(message, flush=True)


def format_elapsed_seconds(elapsed_seconds):
    """経過秒数を人が読みやすい `h/m/s` 形式へ整形する。"""
    elapsed_seconds = int(round(elapsed_seconds))
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def vectorize_train_test_documents(train_docs, test_docs):
    """train 側で語彙を学習し、その語彙で train/test の出現回数行列を作る。

    ここで必ず train 側だけで vectorizer を fit することで、test 側の語彙が
    先に混ざる情報リークを防ぐ。
    """
    train_vectorizer = Tf_idf(list(train_docs), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_train_count, feature_names, fitted_count_vectorizer = train_vectorizer.term_frequency(
        MIN_DF,
        ngram_range=(1, 1),
    )

    # test 文書は train で確定した語彙に射影するだけで、新しい語は追加しない。
    test_vectorizer = Tf_idf(list(test_docs), True, MIN_LEN, use_stemming=USE_STEMMING)
    X_test_count = fitted_count_vectorizer.transform(test_vectorizer.processed_docs)

    return X_train_count.tocsr(), X_test_count.tocsr(), np.array(feature_names)


def _nonzero_doc_counts(X):
    """各特徴語が「何文書に出たか」を列ごとに数える。"""
    if hasattr(X, "getnnz"):
        return np.asarray(X.getnnz(axis=0)).ravel()
    return np.asarray(np.sum(X != 0, axis=0)).ravel()


def build_doc_count_score_table(X, labels, feature_names):
    """文書出現率差にもとづく語の偏りスコア表を作る。

    `label1_score` は「label1 文書に出る割合 - label0 文書に出る割合」、
    `label0_score` はその逆向きの値である。
    """
    mask1 = labels == 1
    mask0 = labels == 0
    X_label1 = X[mask1]
    X_label0 = X[mask0]
    n_label1 = X_label1.shape[0]
    n_label0 = X_label0.shape[0]

    label1_doc_count = _nonzero_doc_counts(X_label1)
    label0_doc_count = _nonzero_doc_counts(X_label0)

    score_label1 = (label1_doc_count / n_label1) - (label0_doc_count / n_label0)
    score_label0 = (label0_doc_count / n_label0) - (label1_doc_count / n_label1)

    # スコアの符号から「どちらのラベル側に偏った語か」を判定しておく。
    dominant_label = np.where(score_label1 > 0, "label1", np.where(score_label1 < 0, "label0", "neutral"))
    selected_score_source = np.where(
        score_label1 > 0,
        "label1_score",
        np.where(score_label1 < 0, "label0_score", ""),
    )
    selected_score = np.where(score_label1 > 0, score_label1, np.where(score_label1 < 0, score_label0, 0.0))

    return pd.DataFrame(
        {
            "term": feature_names,
            "label1_doc_count": label1_doc_count,
            "label0_doc_count": label0_doc_count,
            "label1_total_docs": n_label1,
            "label0_total_docs": n_label0,
            "label1_score": score_label1,
            "label0_score": score_label0,
            "abs_score": np.abs(score_label1),
            "dominant_label": dominant_label,
            "selected_score_source": selected_score_source,
            "selected_score": selected_score,
            "doc_count_diff_label1_minus_label0": label1_doc_count - label0_doc_count,
            "doc_count_diff_label0_minus_label1": label0_doc_count - label1_doc_count,
        }
    )


def select_terms_by_abs_score(score_df, min_abs_score):
    """絶対値スコアが閾値以上の語を採用し、偏り方向ごとにも分けて返す。"""
    selected_terms_df = (
        score_df.loc[score_df["abs_score"] >= min_abs_score]
        .sort_values(
            # 絶対値の強い語を優先し、同点なら偏り量や出現文書数で順序を安定化させる。
            by=["abs_score", "selected_score", "label1_doc_count", "label0_doc_count", "term"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    label1_selected = selected_terms_df.loc[selected_terms_df["dominant_label"] == "label1"].reset_index(drop=True)
    label0_selected = selected_terms_df.loc[selected_terms_df["dominant_label"] == "label0"].reset_index(drop=True)
    return selected_terms_df, label1_selected, label0_selected


def build_selected_feature_matrices(X_train_count, X_test_count, feature_names, selected_terms_df):
    """採用語だけを抜き出し、count/binary の両特徴行列を返す。"""
    selected_terms = selected_terms_df["term"].tolist()
    vocab_index = {term: idx for idx, term in enumerate(feature_names)}
    selected_indices = [vocab_index[term] for term in selected_terms]

    # 同じ採用語集合から、出現回数版と出た/出ない版の 2 種類を作る。
    X_train_selected_count = X_train_count[:, selected_indices]
    X_test_selected_count = X_test_count[:, selected_indices]

    return {
        "selected_terms": selected_terms,
        "count": {
            "train": X_train_selected_count,
            "test": X_test_selected_count,
        },
        "binary": {
            "train": (X_train_selected_count != 0).astype(np.int8),
            "test": (X_test_selected_count != 0).astype(np.int8),
        },
    }


class SplitFeatureContext:
    """1 つの train/test split に必要な特徴量関連情報をまとめて持つコンテナ。"""

    def __init__(self, train_docs, train_labels, test_docs, test_labels, train_doc_ids, test_doc_ids):
        """split 単位でベクトル化とスコア表を事前計算して再利用できるようにする。"""
        self.train_docs = np.array(train_docs, dtype=object)
        self.train_labels = np.array(train_labels, dtype=int)
        self.test_docs = np.array(test_docs, dtype=object)
        self.test_labels = np.array(test_labels, dtype=int)
        self.train_doc_ids = np.array(train_doc_ids, dtype=int)
        self.test_doc_ids = np.array(test_doc_ids, dtype=int)
        self.X_train_count, self.X_test_count, self.feature_names = vectorize_train_test_documents(
            self.train_docs,
            self.test_docs,
        )
        self.score_df = build_doc_count_score_table(self.X_train_count, self.train_labels, self.feature_names)
        self.threshold_cache = {}

    def get_threshold_bundle(self, score_threshold):
        """指定閾値に対する採用語と特徴行列をキャッシュ付きで返す。"""
        if score_threshold not in self.threshold_cache:
            selected_terms_df, label1_selected, label0_selected = select_terms_by_abs_score(self.score_df, score_threshold)
            # 同じ split・同じ threshold は何度も使うので、一度だけ作って保持する。
            feature_matrices = build_selected_feature_matrices(
                self.X_train_count,
                self.X_test_count,
                self.feature_names,
                selected_terms_df,
            )
            self.threshold_cache[score_threshold] = {
                "score_threshold": score_threshold,
                "selected_terms_df": selected_terms_df,
                "label1_selected": label1_selected,
                "label0_selected": label0_selected,
                "selected_term_count": len(feature_matrices["selected_terms"]),
                "feature_matrices": feature_matrices,
            }
        return self.threshold_cache[score_threshold]


def build_outer_fold_contexts(documents, labels, doc_ids):
    """outer/inner の全 split に対する `SplitFeatureContext` を先に構築する。

    候補パラメータごとに毎回ベクトル化し直すと非常に遅いため、split ごとの
    前処理・語彙化・スコア表をここでまとめて用意する。
    """
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_fold_contexts = []
    build_start = time.perf_counter()

    log_progress(f"[split setup] building outer/inner feature contexts for {N_SPLITS}-fold nested CV")

    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(documents, labels), start=1):
        outer_fold_start = time.perf_counter()
        log_progress(f"[split setup] outer fold {outer_fold}/{N_SPLITS}: preparing train/test feature contexts")
        outer_train_docs = documents[train_index]
        outer_train_labels = labels[train_index]
        outer_test_docs = documents[test_index]
        outer_test_labels = labels[test_index]
        outer_train_doc_ids = doc_ids[train_index]
        outer_test_doc_ids = doc_ids[test_index]

        outer_context = SplitFeatureContext(
            outer_train_docs,
            outer_train_labels,
            outer_test_docs,
            outer_test_labels,
            outer_train_doc_ids,
            outer_test_doc_ids,
        )

        # outer train の中だけで inner CV を作り、inner 側でもリークを防ぐ。
        inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        inner_contexts = []
        for inner_train_index, inner_test_index in inner_cv.split(outer_train_docs, outer_train_labels):
            inner_contexts.append(
                SplitFeatureContext(
                    outer_train_docs[inner_train_index],
                    outer_train_labels[inner_train_index],
                    outer_train_docs[inner_test_index],
                    outer_train_labels[inner_test_index],
                    outer_train_doc_ids[inner_train_index],
                    outer_train_doc_ids[inner_test_index],
                )
            )

        outer_fold_contexts.append(
            {
                "outer_fold": outer_fold,
                "outer_context": outer_context,
                "inner_contexts": inner_contexts,
            }
        )
        log_progress(
            f"[split setup] outer fold {outer_fold}/{N_SPLITS} completed "
            f"({len(inner_contexts)} inner contexts, elapsed {format_elapsed_seconds(time.perf_counter() - outer_fold_start)})"
        )

    log_progress(
        f"[split setup] all feature contexts completed "
        f"(elapsed {format_elapsed_seconds(time.perf_counter() - build_start)})"
    )
    return outer_fold_contexts


def _build_svc(kernel, c, gamma=None):
    """指定カーネルとハイパーパラメータから SVC を生成する。"""
    kwargs = {
        "kernel": kernel,
        "C": c,
        "class_weight": CLASS_WEIGHT,
        "tol": TOL,
    }
    if gamma is not None:
        kwargs["gamma"] = gamma
    return svm.SVC(**kwargs)


def _compute_margins(model, X_train, y_train):
    """学習データ上の signed margin を計算する。"""
    decision_values = model.decision_function(X_train)
    y_signed = np.where(y_train == 0, -1, 1)
    return y_signed * decision_values


def format_candidate_label(score_threshold, c, gamma=None):
    """threshold/C/gamma の組み合わせをログ・CSV 用の文字列にする。"""
    if gamma is None:
        return f"threshold={score_threshold}, C={c}"
    return f"threshold={score_threshold}, C={c}, gamma={gamma}"


def format_label_name(label):
    """0/1 ラベルを人が読める名前へ変換する。"""
    return "label1" if label == 1 else "label0"


def fit_and_evaluate_split_context(split_context, score_threshold, feature_mode, kernel, c, gamma, include_misclassified=False):
    """1 つの split で特徴選択から学習・評価までを実行する。

    train 側でのみ語を採用し、その語で train/test を特徴化した後に SVM を学習する。
    margin 条件を満たさない候補はここで `None` として落とす。
    """
    threshold_bundle = split_context.get_threshold_bundle(score_threshold)
    selected_term_count = threshold_bundle["selected_term_count"]
    if selected_term_count == 0:
        return None

    X_train = threshold_bundle["feature_matrices"][feature_mode]["train"]
    X_test = threshold_bundle["feature_matrices"][feature_mode]["test"]

    model = _build_svc(kernel, c, gamma=gamma)
    model.fit(X_train, split_context.train_labels)

    # この実験では「train 全例が margin threshold を超える」候補だけ有効とする。
    margins = _compute_margins(model, X_train, split_context.train_labels)
    if not np.all(margins >= MARGIN_THRESHOLD):
        return None

    y_pred = model.predict(X_test)
    result = {
        "selected_term_count": selected_term_count,
        "accuracy": float(accuracy_score(split_context.test_labels, y_pred)),
        "recall": float(recall_score(split_context.test_labels, y_pred, zero_division=0)),
        "precision": float(precision_score(split_context.test_labels, y_pred, zero_division=0)),
        "f1_score": float(f1_score(split_context.test_labels, y_pred, zero_division=0)),
    }

    if include_misclassified:
        misclassified_rows = []
        for row_idx, (actual_label, predicted_label) in enumerate(zip(split_context.test_labels, y_pred)):
            if actual_label == predicted_label:
                continue
            # outer test で間違えた文書を、後で見直せるよう本文つきで残す。
            misclassified_rows.append(
                {
                    "document_id": int(split_context.test_doc_ids[row_idx]),
                    "actual_label": int(actual_label),
                    "actual_label_name": format_label_name(int(actual_label)),
                    "predicted_label": int(predicted_label),
                    "predicted_label_name": format_label_name(int(predicted_label)),
                    "document_text": str(split_context.test_docs[row_idx]),
                }
            )
        result["misclassified_count"] = len(misclassified_rows)
        result["misclassified_rows"] = misclassified_rows

    return result


def evaluate_candidate_across_outer_folds(outer_fold_contexts, score_threshold, feature_mode, kernel, c, gamma):
    """1 つの候補設定を outer 全 fold で評価する。

    どこか 1 fold でも inner/outer の margin 条件を満たせなければ、その候補は
    実験全体から除外する。
    """
    candidate_label = format_candidate_label(score_threshold, c, gamma)
    fold_records = []

    for fold_bundle in outer_fold_contexts:
        inner_scores = []
        for inner_context in fold_bundle["inner_contexts"]:
            inner_result = fit_and_evaluate_split_context(
                inner_context,
                score_threshold,
                feature_mode,
                kernel,
                c,
                gamma,
            )
            if inner_result is None:
                return None
            inner_scores.append(inner_result["accuracy"])

        # inner で生き残った候補だけを outer test で評価する。
        outer_result = fit_and_evaluate_split_context(
            fold_bundle["outer_context"],
            score_threshold,
            feature_mode,
            kernel,
            c,
            gamma,
        )
        if outer_result is None:
            return None

        fold_records.append(
            {
                "outer_fold": fold_bundle["outer_fold"],
                "score_threshold": score_threshold,
                "selected_term_count": outer_result["selected_term_count"],
                "c": c,
                "gamma": gamma,
                "param_label": candidate_label,
                # この inner 平均精度を、fold ごとの候補選択基準として使う。
                "inner_score": float(sum(inner_scores) / len(inner_scores)),
                "accuracy": outer_result["accuracy"],
                "recall": outer_result["recall"],
                "precision": outer_result["precision"],
                "f1_score": outer_result["f1_score"],
            }
        )

    return fold_records


def round_half_up(value, digits=3):
    """CSV 用に四捨五入したい数値を丸める。空値はそのまま返す。"""
    if value in ("", None):
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return value
    if not np.isfinite(numeric_value):
        return numeric_value
    quantize_exp = Decimal("1").scaleb(-digits)
    try:
        return float(Decimal(str(numeric_value)).quantize(quantize_exp, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return numeric_value


def build_empty_result(feature_mode, kernel, score_threshold_candidates):
    """候補が全滅した場合にも同じ形式で扱える空の結果辞書を作る。"""
    result = {
        "feature_mode": feature_mode,
        "kernel": kernel,
        "margin_threshold": MARGIN_THRESHOLD,
        "candidate_score_thresholds": list(score_threshold_candidates),
        "valid_param_labels": [],
        "dropped_param_labels": [],
        "selected_records": [],
        "best_score_thresholds": [],
        "best_cs": [],
        "selected_term_counts": [],
        "misclassified_counts": [],
    }
    if kernel == "rbf":
        result["best_gammas"] = []
    return result


def run_nested_cv_for_condition(
    outer_fold_contexts,
    feature_mode,
    kernel,
    score_threshold_candidates,
):
    """1 条件の nested CV を最後まで回し、fold ごとの最良候補を選ぶ。

    固定閾値実験では `score_threshold_candidates` が 1 個、
    閾値チューニング実験では複数個になる。
    """
    c_values = LINEAR_C_VALUES if kernel == "linear" else RBF_C_VALUES
    gamma_candidates = [None] if kernel == "linear" else RBF_GAMMA_VALUES
    total_candidates = len(score_threshold_candidates) * len(c_values) * len(gamma_candidates)
    condition_label = format_condition_label(feature_mode, kernel)
    condition_start = time.perf_counter()

    valid_candidate_results = {}
    dropped_param_labels = []
    processed_candidates = 0

    log_progress(
        f"[condition start] {condition_label}: "
        f"{total_candidates} candidate settings across thresholds={score_threshold_candidates}"
    )

    for score_threshold in score_threshold_candidates:
        for c in c_values:
            for gamma in gamma_candidates:
                candidate_label = format_candidate_label(score_threshold, c, gamma)
                # 1 候補について outer 全 fold を走らせ、全 fold で通るか確認する。
                candidate_records = evaluate_candidate_across_outer_folds(
                    outer_fold_contexts,
                    score_threshold,
                    feature_mode,
                    kernel,
                    c,
                    gamma,
                )
                processed_candidates += 1
                if candidate_records is None:
                    dropped_param_labels.append(candidate_label)
                else:
                    valid_candidate_results[(score_threshold, c, gamma)] = candidate_records

                if (
                    processed_candidates == 1
                    or processed_candidates == total_candidates
                    or processed_candidates % CANDIDATE_PROGRESS_INTERVAL == 0
                ):
                    log_progress(
                        f"[condition progress] {condition_label}: "
                        f"{processed_candidates}/{total_candidates} candidates processed "
                        f"(valid={len(valid_candidate_results)}, dropped={len(dropped_param_labels)}, "
                        f"elapsed {format_elapsed_seconds(time.perf_counter() - condition_start)})"
                    )

    result = build_empty_result(feature_mode, kernel, score_threshold_candidates)
    result["valid_param_labels"] = [
        format_candidate_label(score_threshold, c, gamma)
        for score_threshold, c, gamma in valid_candidate_results.keys()
    ]
    result["dropped_param_labels"] = dropped_param_labels

    if not valid_candidate_results:
        log_progress(
            f"[condition done] {condition_label}: "
            f"no valid candidates after {format_elapsed_seconds(time.perf_counter() - condition_start)}"
        )
        return result

    selected_records = []
    for fold_idx in range(len(outer_fold_contexts)):
        best_record = None
        for candidate_records in valid_candidate_results.values():
            candidate = candidate_records[fold_idx]
            if best_record is None:
                best_record = candidate
                continue
            if candidate["inner_score"] > best_record["inner_score"]:
                best_record = candidate
                continue
            if candidate["inner_score"] == best_record["inner_score"] and candidate["param_label"] < best_record["param_label"]:
                best_record = candidate
        selected_records.append(best_record)

    result["selected_records"] = selected_records
    result["best_score_thresholds"] = [record["score_threshold"] for record in selected_records]
    result["best_cs"] = [record["c"] for record in selected_records]
    result["selected_term_counts"] = [record["selected_term_count"] for record in selected_records]
    if kernel == "rbf":
        result["best_gammas"] = [record["gamma"] for record in selected_records]

    # fold ごとに最終採用された候補だけを使って、誤分類文書を保存用に再取得する。
    for fold_idx, selected_record in enumerate(result["selected_records"]):
        outer_detail = fit_and_evaluate_split_context(
            outer_fold_contexts[fold_idx]["outer_context"],
            selected_record["score_threshold"],
            feature_mode,
            kernel,
            selected_record["c"],
            selected_record["gamma"],
            include_misclassified=True,
        )
        selected_record["misclassified_count"] = outer_detail["misclassified_count"]
        selected_record["misclassified_rows"] = outer_detail["misclassified_rows"]

    result["misclassified_counts"] = [record["misclassified_count"] for record in selected_records]

    result["mean_accuracy"] = float(np.mean([record["accuracy"] for record in selected_records]))
    result["mean_recall"] = float(np.mean([record["recall"] for record in selected_records]))
    result["mean_precision"] = float(np.mean([record["precision"] for record in selected_records]))
    result["mean_f1"] = float(np.mean([record["f1_score"] for record in selected_records]))
    result["mean_selected_term_count"] = float(np.mean(result["selected_term_counts"]))
    log_progress(
        f"[condition done] {condition_label}: "
        f"completed in {format_elapsed_seconds(time.perf_counter() - condition_start)}"
    )
    return result


def fit_selected_outer_model_detail(outer_context, selected_record, feature_mode, kernel):
    """outer fold で最終採用された設定を再学習し、分析用の詳細を返す。"""
    threshold_bundle = outer_context.get_threshold_bundle(selected_record["score_threshold"])
    X_train = threshold_bundle["feature_matrices"][feature_mode]["train"]
    X_test = threshold_bundle["feature_matrices"][feature_mode]["test"]

    model = _build_svc(kernel, selected_record["c"], gamma=selected_record["gamma"])
    model.fit(X_train, outer_context.train_labels)

    y_pred = model.predict(X_test)
    decision_values = np.asarray(model.decision_function(X_test)).ravel()
    misclassified_indices = np.flatnonzero(y_pred != outer_context.test_labels)

    return {
        "model": model,
        "threshold_bundle": threshold_bundle,
        "X_train": X_train,
        "X_test": X_test,
        "y_pred": np.asarray(y_pred, dtype=int),
        "decision_values": decision_values,
        "misclassified_indices": misclassified_indices,
        "outer_context": outer_context,
    }


def get_row_feature_indices_and_values(matrix_row):
    """1 行の特徴ベクトルから、非ゼロの特徴 index と値を返す。"""
    if hasattr(matrix_row, "indices") and hasattr(matrix_row, "data"):
        return np.asarray(matrix_row.indices), np.asarray(matrix_row.data)

    row_array = np.asarray(matrix_row).ravel()
    indices = np.flatnonzero(row_array)
    return indices, row_array[indices]


def flatten_model_vector(vector_like):
    """疎行列/密行列のどちらでも 1 次元の数値ベクトルへ揃える。"""
    if hasattr(vector_like, "toarray"):
        return np.asarray(vector_like.toarray()).ravel()
    return np.asarray(vector_like).ravel()


def to_dense_2d(matrix_like):
    """疎行列/密行列のどちらでも 2 次元の密配列へ揃える。"""
    if hasattr(matrix_like, "toarray"):
        return np.asarray(matrix_like.toarray())
    return np.asarray(matrix_like)


def get_signed_effect_label_name(value):
    """符号付き値が label1 / label0 のどちらへ押すかを返す。"""
    if value > 0:
        return format_label_name(1)
    if value < 0:
        return format_label_name(0)
    return "neutral"


def get_error_effect_role(effect_label_name, actual_label_name, predicted_label_name):
    """ある寄与が誤分類を強めたか、緩和したかを返す。"""
    if effect_label_name == predicted_label_name:
        return "pushes_predicted_label"
    if effect_label_name == actual_label_name:
        return "pushes_true_label"
    return "neutral"


def build_top_feature_summary(rows, value_key, top_k=5):
    """上位寄与語を短い文字列へまとめる。"""
    if not rows:
        return ""

    top_rows = sorted(rows, key=lambda row: abs(row[value_key]), reverse=True)[:top_k]
    return " | ".join(
        f"{row['term']}({round_half_up(row[value_key])}, bias={row['term_selection_bias_label']})"
        for row in top_rows
    )


def build_linear_misclassification_analysis_rows(selected_record, model_detail, feature_mode, kernel):
    """線形 SVM の `coef_` を使い、誤分類文書ごとの語寄与を展開する。"""
    # 疎行列入力で学習した SVC(kernel="linear") の coef_ は csr_matrix になることがある。
    # np.asarray(...) だけだと「疎行列オブジェクト1個の配列」になってしまうため、
    # toarray() を優先して数値ベクトルへ展開する。
    coefficient_vector = flatten_model_vector(model_detail["model"].coef_)
    selected_terms = model_detail["threshold_bundle"]["feature_matrices"]["selected_terms"]
    if coefficient_vector.shape[0] != len(selected_terms):
        raise RuntimeError(
            "linear coef_ length does not match selected term count: "
            f"coef_len={coefficient_vector.shape[0]}, selected_terms={len(selected_terms)}"
        )
    term_metadata_lookup = (
        model_detail["threshold_bundle"]["selected_terms_df"].set_index("term").to_dict(orient="index")
    )

    detail_rows = []
    document_summary_rows = []

    for doc_rank, misclassified_idx in enumerate(model_detail["misclassified_indices"], start=1):
        actual_label = int(model_detail["outer_context"].test_labels[misclassified_idx])
        predicted_label = int(model_detail["y_pred"][misclassified_idx])
        actual_label_name = format_label_name(actual_label)
        predicted_label_name = format_label_name(predicted_label)
        decision_value = float(model_detail["decision_values"][misclassified_idx])
        doc_id = int(model_detail["outer_context"].test_doc_ids[misclassified_idx])
        document_text = str(model_detail["outer_context"].test_docs[misclassified_idx])

        feature_indices, feature_values = get_row_feature_indices_and_values(model_detail["X_test"][misclassified_idx])
        doc_detail_rows = []

        for feature_idx, feature_value in zip(feature_indices, feature_values):
            term = selected_terms[int(feature_idx)]
            term_metadata = term_metadata_lookup.get(term, {})
            coefficient = float(coefficient_vector[int(feature_idx)])
            contribution = float(coefficient * float(feature_value))
            row = {
                "analysis_method": "linear_coef",
                "outer_fold": selected_record["outer_fold"],
                "document_rank_in_fold": doc_rank,
                "document_id": doc_id,
                "actual_label": actual_label,
                "actual_label_name": actual_label_name,
                "predicted_label": predicted_label,
                "predicted_label_name": predicted_label_name,
                "score_threshold": selected_record["score_threshold"],
                "feature_mode": feature_mode,
                "kernel": kernel,
                "selected_term_count": selected_record["selected_term_count"],
                "c": selected_record["c"],
                "gamma": selected_record["gamma"],
                "decision_value": decision_value,
                "term": term,
                "feature_value": float(feature_value),
                "coefficient": coefficient,
                "coefficient_support_label": get_signed_effect_label_name(coefficient),
                "contribution": contribution,
                "contribution_support_label": get_signed_effect_label_name(contribution),
                "error_effect_role": get_error_effect_role(
                    get_signed_effect_label_name(contribution),
                    actual_label_name,
                    predicted_label_name,
                ),
                "term_selection_bias_label": term_metadata.get("dominant_label", ""),
                "term_selection_score_source": term_metadata.get("selected_score_source", ""),
                "term_selection_score": float(term_metadata.get("selected_score", 0.0)),
                "term_abs_score": float(term_metadata.get("abs_score", 0.0)),
                "term_label1_score": float(term_metadata.get("label1_score", 0.0)),
                "term_label0_score": float(term_metadata.get("label0_score", 0.0)),
                "document_text": document_text,
            }
            doc_detail_rows.append(row)

        doc_detail_rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
        for contribution_rank, row in enumerate(doc_detail_rows, start=1):
            row["feature_rank_in_document"] = contribution_rank
        detail_rows.extend(doc_detail_rows)

        predicted_push_rows = [
            row for row in doc_detail_rows if row["error_effect_role"] == "pushes_predicted_label"
        ]
        true_push_rows = [
            row for row in doc_detail_rows if row["error_effect_role"] == "pushes_true_label"
        ]
        document_summary_rows.append(
            {
                "analysis_method": "linear_coef",
                "outer_fold": selected_record["outer_fold"],
                "document_rank_in_fold": doc_rank,
                "document_id": doc_id,
                "actual_label": actual_label,
                "actual_label_name": actual_label_name,
                "predicted_label": predicted_label,
                "predicted_label_name": predicted_label_name,
                "score_threshold": selected_record["score_threshold"],
                "feature_mode": feature_mode,
                "kernel": kernel,
                "selected_term_count": selected_record["selected_term_count"],
                "c": selected_record["c"],
                "gamma": selected_record["gamma"],
                "decision_value": decision_value,
                "active_feature_count": len(doc_detail_rows),
                "predicted_push_feature_count": len(predicted_push_rows),
                "true_push_feature_count": len(true_push_rows),
                "top_predicted_push_terms": build_top_feature_summary(
                    predicted_push_rows,
                    "contribution",
                    top_k=LINEAR_ANALYSIS_TOP_K,
                ),
                "top_true_push_terms": build_top_feature_summary(
                    true_push_rows,
                    "contribution",
                    top_k=LINEAR_ANALYSIS_TOP_K,
                ),
                "document_text": document_text,
            }
        )

    return detail_rows, document_summary_rows


def _extract_shap_vector(shap_values):
    """SHAP の返り値を 1 文書分の 1 次元配列へそろえる。"""
    if hasattr(shap_values, "values"):
        shap_array = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[0])
    else:
        shap_array = np.asarray(shap_values)

    if shap_array.ndim == 2:
        shap_array = shap_array[0]
    return np.asarray(shap_array).ravel()


def build_analysis_status_row(selected_record, feature_mode, kernel, status, detail_row_count=0, summary_row_count=0, error=None):
    """fold 単位の分析成否を CSV に残す。"""
    return {
        "outer_fold": selected_record["outer_fold"],
        "score_threshold": selected_record["score_threshold"],
        "feature_mode": feature_mode,
        "kernel": kernel,
        "selected_term_count": selected_record["selected_term_count"],
        "c": selected_record["c"],
        "gamma": selected_record["gamma"],
        "status": status,
        "detail_row_count": detail_row_count,
        "summary_row_count": summary_row_count,
        "error_type": type(error).__name__ if error is not None else "",
        "error_message": str(error) if error is not None else "",
    }


def build_rbf_misclassification_analysis_rows(selected_record, model_detail, feature_mode, kernel):
    """RBF SVM の誤分類文書に対して SHAP で局所寄与を計算する。"""
    if shap is None:
        raise RuntimeError("shap is not installed.")
    if len(model_detail["misclassified_indices"]) == 0:
        return [], []

    selected_terms = model_detail["threshold_bundle"]["feature_matrices"]["selected_terms"]
    term_metadata_lookup = (
        model_detail["threshold_bundle"]["selected_terms_df"].set_index("term").to_dict(orient="index")
    )

    background_size = min(RBF_SHAP_BACKGROUND_SIZE, model_detail["X_train"].shape[0])
    if background_size == 0:
        return [], []
    background_indices = np.linspace(0, model_detail["X_train"].shape[0] - 1, num=background_size, dtype=int)
    background_dense = to_dense_2d(model_detail["X_train"][background_indices])

    def decision_function_for_shap(data):
        return np.asarray(model_detail["model"].decision_function(data)).ravel()

    explainer = shap.KernelExplainer(decision_function_for_shap, background_dense)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    detail_rows = []
    document_summary_rows = []

    for doc_rank, misclassified_idx in enumerate(model_detail["misclassified_indices"], start=1):
        actual_label = int(model_detail["outer_context"].test_labels[misclassified_idx])
        predicted_label = int(model_detail["y_pred"][misclassified_idx])
        actual_label_name = format_label_name(actual_label)
        predicted_label_name = format_label_name(predicted_label)
        decision_value = float(model_detail["decision_values"][misclassified_idx])
        doc_id = int(model_detail["outer_context"].test_doc_ids[misclassified_idx])
        document_text = str(model_detail["outer_context"].test_docs[misclassified_idx])

        doc_dense = to_dense_2d(model_detail["X_test"][misclassified_idx])
        feature_values = doc_dense.ravel()
        # KernelExplainer の既定 l1_reg="aic" は、nsamples より特徴数が多いと
        # 内部の LassoLarsIC が分散推定できず落ちることがある。
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Linear regression equation is singular, a least squares solutions is used instead.",
                category=UserWarning,
            )
            shap_vector = _extract_shap_vector(
                explainer.shap_values(
                    doc_dense,
                    nsamples=RBF_SHAP_NSAMPLES,
                    l1_reg=RBF_SHAP_L1_REG,
                    silent=True,
                )
            )
        if not np.all(np.isfinite(shap_vector)):
            shap_vector = np.nan_to_num(shap_vector, nan=0.0, posinf=0.0, neginf=0.0)
        if shap_vector.shape[0] != len(selected_terms):
            raise RuntimeError(
                "rbf shap length does not match selected term count: "
                f"shap_len={shap_vector.shape[0]}, selected_terms={len(selected_terms)}"
            )

        nonzero_indices = np.flatnonzero(feature_values)
        if len(nonzero_indices) == 0:
            ranked_indices = np.argsort(np.abs(shap_vector))[::-1][: min(RBF_ANALYSIS_TOP_K, len(shap_vector))]
        else:
            ranked_indices = nonzero_indices[np.argsort(np.abs(shap_vector[nonzero_indices]))[::-1]]
            ranked_indices = ranked_indices[: min(RBF_ANALYSIS_TOP_K, len(ranked_indices))]

        doc_detail_rows = []
        for feature_idx in ranked_indices:
            term = selected_terms[int(feature_idx)]
            term_metadata = term_metadata_lookup.get(term, {})
            shap_value = float(shap_vector[int(feature_idx)])
            row = {
                "analysis_method": "rbf_shap",
                "outer_fold": selected_record["outer_fold"],
                "document_rank_in_fold": doc_rank,
                "document_id": doc_id,
                "actual_label": actual_label,
                "actual_label_name": actual_label_name,
                "predicted_label": predicted_label,
                "predicted_label_name": predicted_label_name,
                "score_threshold": selected_record["score_threshold"],
                "feature_mode": feature_mode,
                "kernel": kernel,
                "selected_term_count": selected_record["selected_term_count"],
                "c": selected_record["c"],
                "gamma": selected_record["gamma"],
                "expected_value": expected_value,
                "decision_value": decision_value,
                "term": term,
                "feature_value": float(feature_values[int(feature_idx)]),
                "shap_value": shap_value,
                "shap_support_label": get_signed_effect_label_name(shap_value),
                "error_effect_role": get_error_effect_role(
                    get_signed_effect_label_name(shap_value),
                    actual_label_name,
                    predicted_label_name,
                ),
                "term_selection_bias_label": term_metadata.get("dominant_label", ""),
                "term_selection_score_source": term_metadata.get("selected_score_source", ""),
                "term_selection_score": float(term_metadata.get("selected_score", 0.0)),
                "term_abs_score": float(term_metadata.get("abs_score", 0.0)),
                "term_label1_score": float(term_metadata.get("label1_score", 0.0)),
                "term_label0_score": float(term_metadata.get("label0_score", 0.0)),
                "document_text": document_text,
            }
            doc_detail_rows.append(row)

        doc_detail_rows.sort(key=lambda row: abs(row["shap_value"]), reverse=True)
        for shap_rank, row in enumerate(doc_detail_rows, start=1):
            row["feature_rank_in_document"] = shap_rank
        detail_rows.extend(doc_detail_rows)

        predicted_push_rows = [
            row for row in doc_detail_rows if row["error_effect_role"] == "pushes_predicted_label"
        ]
        true_push_rows = [
            row for row in doc_detail_rows if row["error_effect_role"] == "pushes_true_label"
        ]
        document_summary_rows.append(
            {
                "analysis_method": "rbf_shap",
                "outer_fold": selected_record["outer_fold"],
                "document_rank_in_fold": doc_rank,
                "document_id": doc_id,
                "actual_label": actual_label,
                "actual_label_name": actual_label_name,
                "predicted_label": predicted_label,
                "predicted_label_name": predicted_label_name,
                "score_threshold": selected_record["score_threshold"],
                "feature_mode": feature_mode,
                "kernel": kernel,
                "selected_term_count": selected_record["selected_term_count"],
                "c": selected_record["c"],
                "gamma": selected_record["gamma"],
                "expected_value": expected_value,
                "decision_value": decision_value,
                "explained_feature_count": len(doc_detail_rows),
                "predicted_push_feature_count": len(predicted_push_rows),
                "true_push_feature_count": len(true_push_rows),
                "top_predicted_push_terms": build_top_feature_summary(
                    predicted_push_rows,
                    "shap_value",
                    top_k=RBF_ANALYSIS_TOP_K,
                ),
                "top_true_push_terms": build_top_feature_summary(
                    true_push_rows,
                    "shap_value",
                    top_k=RBF_ANALYSIS_TOP_K,
                ),
                "document_text": document_text,
            }
        )

    return detail_rows, document_summary_rows


def save_outer_fold_misclassification_feature_analysis(result, outer_fold_contexts, feature_mode, kernel, output_dir):
    """誤分類文書の語寄与分析を、既存出力とは別の CSV 群として保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if kernel == "linear":
        detail_filename = "lin_term.csv"
        summary_filename = "lin_doc.csv"
        detail_columns = [
            "analysis_method",
            "outer_fold",
            "document_rank_in_fold",
            "document_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "score_threshold",
            "feature_mode",
            "kernel",
            "selected_term_count",
            "c",
            "gamma",
            "decision_value",
            "feature_rank_in_document",
            "term",
            "feature_value",
            "coefficient",
            "coefficient_support_label",
            "contribution",
            "contribution_support_label",
            "error_effect_role",
            "term_selection_bias_label",
            "term_selection_score_source",
            "term_selection_score",
            "term_abs_score",
            "term_label1_score",
            "term_label0_score",
            "document_text",
        ]
        summary_columns = [
            "analysis_method",
            "outer_fold",
            "document_rank_in_fold",
            "document_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "score_threshold",
            "feature_mode",
            "kernel",
            "selected_term_count",
            "c",
            "gamma",
            "decision_value",
            "active_feature_count",
            "predicted_push_feature_count",
            "true_push_feature_count",
            "top_predicted_push_terms",
            "top_true_push_terms",
            "document_text",
        ]
    else:
        detail_filename = "rbf_shap.csv"
        summary_filename = "rbf_doc.csv"
        detail_columns = [
            "analysis_method",
            "outer_fold",
            "document_rank_in_fold",
            "document_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "score_threshold",
            "feature_mode",
            "kernel",
            "selected_term_count",
            "c",
            "gamma",
            "expected_value",
            "decision_value",
            "feature_rank_in_document",
            "term",
            "feature_value",
            "shap_value",
            "shap_support_label",
            "error_effect_role",
            "term_selection_bias_label",
            "term_selection_score_source",
            "term_selection_score",
            "term_abs_score",
            "term_label1_score",
            "term_label0_score",
            "document_text",
        ]
        summary_columns = [
            "analysis_method",
            "outer_fold",
            "document_rank_in_fold",
            "document_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "score_threshold",
            "feature_mode",
            "kernel",
            "selected_term_count",
            "c",
            "gamma",
            "expected_value",
            "decision_value",
            "explained_feature_count",
            "predicted_push_feature_count",
            "true_push_feature_count",
            "top_predicted_push_terms",
            "top_true_push_terms",
            "document_text",
        ]

    detail_rows = []
    summary_rows = []
    status_rows = []

    log_progress(f"[analysis] {format_condition_label(feature_mode, kernel)}: building misclassification feature analysis")
    for selected_record in result.get("selected_records", []):
        outer_fold = selected_record["outer_fold"]
        log_progress(
            f"[analysis] {format_condition_label(feature_mode, kernel)}: "
            f"outer fold {outer_fold}/{len(result.get('selected_records', []))}"
        )
        try:
            outer_context = outer_fold_contexts[outer_fold - 1]["outer_context"]
            model_detail = fit_selected_outer_model_detail(outer_context, selected_record, feature_mode, kernel)
            if len(model_detail["misclassified_indices"]) == 0:
                status_rows.append(
                    build_analysis_status_row(
                        selected_record,
                        feature_mode,
                        kernel,
                        status="no_misclassified",
                    )
                )
                continue

            if kernel == "linear":
                fold_detail_rows, fold_summary_rows = build_linear_misclassification_analysis_rows(
                    selected_record,
                    model_detail,
                    feature_mode,
                    kernel,
                )
            else:
                if shap is None:
                    status_rows.append(
                        build_analysis_status_row(
                            selected_record,
                            feature_mode,
                            kernel,
                            status="shap_unavailable",
                        )
                    )
                    continue
                fold_detail_rows, fold_summary_rows = build_rbf_misclassification_analysis_rows(
                    selected_record,
                    model_detail,
                    feature_mode,
                    kernel,
                )

            detail_rows.extend(fold_detail_rows)
            summary_rows.extend(fold_summary_rows)
            status_rows.append(
                build_analysis_status_row(
                    selected_record,
                    feature_mode,
                    kernel,
                    status="ok",
                    detail_row_count=len(fold_detail_rows),
                    summary_row_count=len(fold_summary_rows),
                )
            )
        except Exception as exc:
            log_progress(
                f"[analysis warning] {format_condition_label(feature_mode, kernel)} "
                f"outer fold {outer_fold}: {type(exc).__name__}: {exc}"
            )
            status_rows.append(
                build_analysis_status_row(
                    selected_record,
                    feature_mode,
                    kernel,
                    status="error",
                    error=exc,
                )
            )

    detail_path = output_dir / detail_filename
    summary_path = output_dir / summary_filename
    status_path = output_dir / "status.csv"

    pd.DataFrame(detail_rows, columns=detail_columns).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows, columns=summary_columns).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        status_rows,
        columns=[
            "outer_fold",
            "score_threshold",
            "feature_mode",
            "kernel",
            "selected_term_count",
            "c",
            "gamma",
            "status",
            "detail_row_count",
            "summary_row_count",
            "error_type",
            "error_message",
        ],
    ).to_csv(status_path, index=False, encoding="utf-8-sig")

    if kernel == "linear":
        result.setdefault("saved_csv_paths", {})
        result["saved_csv_paths"]["linear_error_analysis"] = str(detail_path)
        result["saved_csv_paths"]["linear_error_summary"] = str(summary_path)
        result["saved_csv_paths"]["linear_error_status"] = str(status_path)
    else:
        result.setdefault("saved_csv_paths", {})
        result["saved_csv_paths"]["rbf_error_analysis"] = str(detail_path)
        result["saved_csv_paths"]["rbf_error_summary"] = str(summary_path)
        result["saved_csv_paths"]["rbf_error_status"] = str(status_path)


def run_optional_output_step(result, step_label, callback, *args):
    """補助出力で失敗しても主結果を落とさず、警告として保持する。"""
    try:
        callback(*args)
    except Exception as exc:
        log_progress(f"[output warning] {step_label} failed: {type(exc).__name__}: {exc}")
        result.setdefault("output_warnings", []).append(
            f"{step_label}: {type(exc).__name__}: {exc}"
        )


def format_threshold_slug(score_threshold):
    """閾値をディレクトリ名に使いやすい文字列へ変換する。"""
    return f"t{str(score_threshold).replace('-', 'neg').replace('.', 'p')}"


def get_condition_metrics_dir(metrics_root_dir, feature_mode, kernel):
    """特徴種別とカーネルごとの出力ディレクトリを返す。"""
    condition_dir = Path(metrics_root_dir) / FEATURE_MODE_DIR_NAMES[feature_mode] / KERNEL_DIR_NAMES[kernel]
    condition_dir.mkdir(parents=True, exist_ok=True)
    return condition_dir


def save_nested_cv_result_csvs(result, output_dir, output_prefix):
    """nested CV の主要結果を 3 種類の CSV に保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    selected_path = output_dir / f"{output_prefix}_folds.csv"
    params_path = output_dir / f"{output_prefix}_params.csv"

    # 条件全体の平均指標を 1 行で見られるよう、summary 用 CSV を先に作る。
    metrics_row = {
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "margin_threshold": result["margin_threshold"],
        "candidate_score_thresholds": ",".join(map(str, result.get("candidate_score_thresholds", []))),
        "selected_score_thresholds": ",".join(map(str, result.get("best_score_thresholds", []))),
        "mean_selected_term_count": round_half_up(result.get("mean_selected_term_count", "")),
        "selected_term_counts": ",".join(map(str, result.get("selected_term_counts", []))),
        "mean_accuracy": round_half_up(result.get("mean_accuracy", "")),
        "mean_recall": round_half_up(result.get("mean_recall", "")),
        "mean_precision": round_half_up(result.get("mean_precision", "")),
        "mean_f1": round_half_up(result.get("mean_f1", "")),
        "best_cs": ",".join(map(str, result.get("best_cs", []))),
        "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # fold ごとに実際に採用された候補と指標も別 CSV へ保存する。
    selected_records_df = pd.DataFrame(
        result.get("selected_records", []),
        columns=[
            "outer_fold",
            "score_threshold",
            "selected_term_count",
            "misclassified_count",
            "c",
            "gamma",
            "param_label",
            "inner_score",
            "accuracy",
            "recall",
            "precision",
            "f1_score",
        ],
    )
    if not selected_records_df.empty:
        selected_records_df = selected_records_df.copy()
        for column in ["inner_score", "accuracy", "recall", "precision", "f1_score"]:
            selected_records_df[column] = selected_records_df[column].map(round_half_up)
    selected_records_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    # 全候補のうち margin 条件を通ったもの / 落ちたものも後で確認できるようにする。
    param_status_rows = [{"status": "valid", "param_label": label} for label in result.get("valid_param_labels", [])]
    param_status_rows.extend(
        {"status": "dropped", "param_label": label} for label in result.get("dropped_param_labels", [])
    )
    pd.DataFrame(param_status_rows, columns=["status", "param_label"]).to_csv(
        params_path,
        index=False,
        encoding="utf-8-sig",
    )

    result["saved_csv_paths"] = {
        "metrics": str(metrics_path),
        "selected_records": str(selected_path),
        "param_status": str(params_path),
    }


def save_outer_fold_selected_terms(result, outer_fold_contexts, output_dir):
    """outer fold ごとに、最終採用された語一覧を保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for record in result.get("selected_records", []):
        outer_fold = record["outer_fold"]
        threshold_bundle = outer_fold_contexts[outer_fold - 1]["outer_context"].get_threshold_bundle(record["score_threshold"])
        selected_terms_df = threshold_bundle["selected_terms_df"].copy()
        selected_terms_df.insert(0, "outer_fold", outer_fold)
        selected_terms_df.insert(1, "score_threshold", record["score_threshold"])
        file_path = output_dir / f"f{outer_fold:02d}_terms.csv"
        selected_terms_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        # どの fold でどの閾値・何語採用されたかだけを一覧でも持っておく。
        summary_rows.append(
            {
                "outer_fold": outer_fold,
                "score_threshold": record["score_threshold"],
                "selected_term_count": record["selected_term_count"],
                "file_path": str(file_path),
            }
        )

    pd.DataFrame(
        summary_rows,
        columns=["outer_fold", "score_threshold", "selected_term_count", "file_path"],
    ).to_csv(
        output_dir / "feat_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_outer_fold_misclassified_documents(result, output_dir):
    """outer test で誤分類した文書を fold 別・全 fold 結合の両方で保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    summary_rows = []

    for record in result.get("selected_records", []):
        outer_fold = record["outer_fold"]
        fold_rows = []
        for row in record.get("misclassified_rows", []):
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "score_threshold": record["score_threshold"],
                    "selected_term_count": record["selected_term_count"],
                    "c": record["c"],
                    "gamma": record["gamma"],
                    "param_label": record["param_label"],
                    **row,
                }
            )

        # fold 単位の誤分類 CSV と、全 fold 結合 CSV の両方を作る。
        file_path = output_dir / f"f{outer_fold:02d}_mis.csv"
        pd.DataFrame(
            fold_rows,
            columns=[
                "outer_fold",
                "score_threshold",
                "selected_term_count",
                "c",
                "gamma",
                "param_label",
                "document_id",
                "actual_label",
                "actual_label_name",
                "predicted_label",
                "predicted_label_name",
                "document_text",
            ],
        ).to_csv(file_path, index=False, encoding="utf-8-sig")

        combined_rows.extend(fold_rows)
        summary_rows.append(
            {
                "outer_fold": outer_fold,
                "score_threshold": record["score_threshold"],
                "selected_term_count": record["selected_term_count"],
                "misclassified_count": record.get("misclassified_count", 0),
                "file_path": str(file_path),
            }
        )

    combined_path = output_dir / "mis_all.csv"
    pd.DataFrame(
        combined_rows,
        columns=[
            "outer_fold",
            "score_threshold",
            "selected_term_count",
            "c",
            "gamma",
            "param_label",
            "document_id",
            "actual_label",
            "actual_label_name",
            "predicted_label",
            "predicted_label_name",
            "document_text",
        ],
    ).to_csv(combined_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "mis_summary.csv"
    pd.DataFrame(
        summary_rows,
        columns=["outer_fold", "score_threshold", "selected_term_count", "misclassified_count", "file_path"],
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    result.setdefault("saved_csv_paths", {})
    result["saved_csv_paths"]["misclassified_all"] = str(combined_path)
    result["saved_csv_paths"]["misclassified_summary"] = str(summary_path)


def save_threshold_term_count_outputs(outer_fold_contexts, score_threshold_candidates, output_dir):
    """閾値ごとの採用語数と偏り内訳を outer fold 単位・要約の両方で保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold_bundle in outer_fold_contexts:
        outer_fold = fold_bundle["outer_fold"]
        outer_context = fold_bundle["outer_context"]
        for score_threshold in score_threshold_candidates:
            threshold_bundle = outer_context.get_threshold_bundle(score_threshold)
            label1_selected_count = len(threshold_bundle["label1_selected"])
            label0_selected_count = len(threshold_bundle["label0_selected"])
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "score_threshold": score_threshold,
                    "selected_term_count": threshold_bundle["selected_term_count"],
                    "label1_selected_term_count": label1_selected_count,
                    "label0_selected_term_count": label0_selected_count,
                }
            )

    fold_counts_path = output_dir / "fold_th_counts.csv"
    pd.DataFrame(
        fold_rows,
        columns=[
            "outer_fold",
            "score_threshold",
            "selected_term_count",
            "label1_selected_term_count",
            "label0_selected_term_count",
        ],
    ).to_csv(fold_counts_path, index=False, encoding="utf-8-sig")

    # 閾値ごとの平均件数も別 CSV にして、閾値比較しやすくする。
    summary_rows = []
    for score_threshold in score_threshold_candidates:
        threshold_rows = [row for row in fold_rows if row["score_threshold"] == score_threshold]
        summary_rows.append(
            {
                "score_threshold": score_threshold,
                "mean_selected_term_count": round_half_up(np.mean([row["selected_term_count"] for row in threshold_rows])),
                "mean_label1_selected_term_count": round_half_up(
                    np.mean([row["label1_selected_term_count"] for row in threshold_rows])
                ),
                "mean_label0_selected_term_count": round_half_up(
                    np.mean([row["label0_selected_term_count"] for row in threshold_rows])
                ),
                "selected_term_counts": ",".join(str(row["selected_term_count"]) for row in threshold_rows),
                "label1_selected_term_counts": ",".join(str(row["label1_selected_term_count"]) for row in threshold_rows),
                "label0_selected_term_counts": ",".join(str(row["label0_selected_term_count"]) for row in threshold_rows),
            }
        )

    summary_path = output_dir / "th_count_summary.csv"
    pd.DataFrame(
        summary_rows,
        columns=[
            "score_threshold",
            "mean_selected_term_count",
            "mean_label1_selected_term_count",
            "mean_label0_selected_term_count",
            "selected_term_counts",
            "label1_selected_term_counts",
            "label0_selected_term_counts",
        ],
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    return {
        "outer_fold_threshold_term_counts": str(fold_counts_path),
        "threshold_term_count_summary": str(summary_path),
    }


def format_condition_label(feature_mode, kernel):
    """条件名をログ表示向けの日本語へ整形する。"""
    feature_mode_label = "バイナリ特徴" if feature_mode == "binary" else "出現頻度特徴"
    kernel_label = "線形SVM" if kernel == "linear" else "非線形SVM"
    return f"{feature_mode_label} / {kernel_label}"


def build_fixed_threshold_summary_row(score_threshold, result):
    """固定閾値実験の 1 行 summary を作る。"""
    row = {
        "score_threshold": score_threshold,
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "mean_selected_term_count": "",
        "selected_term_counts": "",
        "mean_accuracy": "",
        "mean_recall": "",
        "mean_precision": "",
        "mean_f1": "",
        "best_cs": "",
        "best_gammas": "",
        "valid_param_count": len(result.get("valid_param_labels", [])),
        "dropped_param_count": len(result.get("dropped_param_labels", [])),
    }

    if result.get("selected_records"):
        row.update(
            {
                "mean_selected_term_count": round_half_up(result["mean_selected_term_count"]),
                "selected_term_counts": ",".join(map(str, result.get("selected_term_counts", []))),
                "mean_accuracy": round_half_up(result["mean_accuracy"]),
                "mean_recall": round_half_up(result["mean_recall"]),
                "mean_precision": round_half_up(result["mean_precision"]),
                "mean_f1": round_half_up(result["mean_f1"]),
                "best_cs": ",".join(map(str, result.get("best_cs", []))),
                "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
            }
        )

    return row


def build_tuned_threshold_summary_row(result):
    """閾値チューニング実験の 1 行 summary を作る。"""
    row = {
        "candidate_score_thresholds": ",".join(map(str, result.get("candidate_score_thresholds", []))),
        "feature_mode": result["feature_mode"],
        "kernel": result["kernel"],
        "selected_score_thresholds": "",
        "mean_selected_term_count": "",
        "selected_term_counts": "",
        "mean_accuracy": "",
        "mean_recall": "",
        "mean_precision": "",
        "mean_f1": "",
        "best_cs": "",
        "best_gammas": "",
        "valid_param_count": len(result.get("valid_param_labels", [])),
        "dropped_param_count": len(result.get("dropped_param_labels", [])),
    }

    if result.get("selected_records"):
        row.update(
            {
                "selected_score_thresholds": ",".join(map(str, result.get("best_score_thresholds", []))),
                "mean_selected_term_count": round_half_up(result["mean_selected_term_count"]),
                "selected_term_counts": ",".join(map(str, result.get("selected_term_counts", []))),
                "mean_accuracy": round_half_up(result["mean_accuracy"]),
                "mean_recall": round_half_up(result["mean_recall"]),
                "mean_precision": round_half_up(result["mean_precision"]),
                "mean_f1": round_half_up(result["mean_f1"]),
                "best_cs": ",".join(map(str, result.get("best_cs", []))),
                "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
            }
        )

    return row


def save_summary_csv(rows, columns, output_path):
    """列順を固定した summary CSV を保存する。"""
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_best_result_csv(best_row, columns, output_path):
    """best result があれば 1 行、なければヘッダーだけの CSV を保存する。"""
    if best_row is None:
        pd.DataFrame(columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    pd.DataFrame([{column: best_row.get(column, "") for column in columns}]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_fixed_threshold_global_outputs(summary_rows_by_condition, best_results_by_condition, output_root):
    """固定閾値フェーズ全体の summary と best result を条件別に保存する。"""
    metrics_root = Path(output_root) / METRICS_DIR_NAME
    metrics_root.mkdir(parents=True, exist_ok=True)

    for feature_mode, kernel in get_active_conditions():
        condition_dir = get_condition_metrics_dir(metrics_root, feature_mode, kernel)
        save_summary_csv(
            summary_rows_by_condition[(feature_mode, kernel)],
            FIXED_THRESHOLD_SUMMARY_COLUMNS,
            condition_dir / "fix_summary.csv",
        )
        save_best_result_csv(
            best_results_by_condition[(feature_mode, kernel)],
            FIXED_THRESHOLD_BEST_RESULT_COLUMNS,
            condition_dir / "fix_best.csv",
        )


def save_tuned_threshold_global_outputs(summary_rows, output_root):
    """閾値チューニングフェーズ全体の summary を保存する。"""
    metrics_root = Path(output_root) / METRICS_DIR_NAME
    metrics_root.mkdir(parents=True, exist_ok=True)
    save_summary_csv(
        summary_rows,
        TUNED_THRESHOLD_SUMMARY_COLUMNS,
        metrics_root / "tun_summary.csv",
    )


def get_active_conditions(kernel_mode, feature_mode):
    """起動時入力にもとづいて実行対象の特徴種別×カーネルの組を返す。"""
    feature_modes = ["binary", "count"] if feature_mode == "both" else [feature_mode]
    kernels = ["linear", "rbf"] if kernel_mode == "both" else [kernel_mode]

    conditions = []
    for kernel in kernels:
        for current_feature_mode in feature_modes:
            conditions.append((current_feature_mode, kernel))
    return conditions


def print_result(result, heading):
    """1 条件ぶんの結果をコンソールに見やすく表示する。"""
    print(f"\n{heading}")
    if not result.get("selected_records"):
        print(f"margin >= {result['margin_threshold']} を全foldで満たす組み合わせがありませんでした。")
        if "saved_csv_paths" in result:
            print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
            print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
            print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")
            if "misclassified_all" in result["saved_csv_paths"]:
                print(f"saved misclassified csv: {result['saved_csv_paths']['misclassified_all']}")
                print(f"saved misclassified summary csv: {result['saved_csv_paths']['misclassified_summary']}")
            if "linear_error_analysis" in result["saved_csv_paths"]:
                print(f"saved linear error analysis csv: {result['saved_csv_paths']['linear_error_analysis']}")
                print(f"saved linear error summary csv: {result['saved_csv_paths']['linear_error_summary']}")
            if "rbf_error_analysis" in result["saved_csv_paths"]:
                print(f"saved rbf shap analysis csv: {result['saved_csv_paths']['rbf_error_analysis']}")
                print(f"saved rbf shap summary csv: {result['saved_csv_paths']['rbf_error_summary']}")
                if "rbf_error_status" in result["saved_csv_paths"]:
                    print(f"saved rbf shap status csv: {result['saved_csv_paths']['rbf_error_status']}")
        for warning_message in result.get("output_warnings", []):
            print(f"warning: {warning_message}")
        return

    print(f"selected thresholds: {result['best_score_thresholds']}")
    print(f"selected term counts: {result['selected_term_counts']}")
    print(f"accuracy : {result['mean_accuracy']:.4f}")
    print(f"recall   : {result['mean_recall']:.4f}")
    print(f"precision: {result['mean_precision']:.4f}")
    print(f"f1       : {result['mean_f1']:.4f}")
    print(f"selected C values: {result['best_cs']}")
    if "best_gammas" in result:
        print(f"selected gamma values: {result['best_gammas']}")
    print(f"globally valid params: {result['valid_param_labels']}")
    if "saved_csv_paths" in result:
        print(f"saved metrics csv: {result['saved_csv_paths']['metrics']}")
        print(f"saved selected-records csv: {result['saved_csv_paths']['selected_records']}")
        print(f"saved param-status csv: {result['saved_csv_paths']['param_status']}")
        if "misclassified_all" in result["saved_csv_paths"]:
            print(f"saved misclassified csv: {result['saved_csv_paths']['misclassified_all']}")
            print(f"saved misclassified summary csv: {result['saved_csv_paths']['misclassified_summary']}")
        if "linear_error_analysis" in result["saved_csv_paths"]:
            print(f"saved linear error analysis csv: {result['saved_csv_paths']['linear_error_analysis']}")
            print(f"saved linear error summary csv: {result['saved_csv_paths']['linear_error_summary']}")
            if "linear_error_status" in result["saved_csv_paths"]:
                print(f"saved linear error status csv: {result['saved_csv_paths']['linear_error_status']}")
        if "rbf_error_analysis" in result["saved_csv_paths"]:
            print(f"saved rbf shap analysis csv: {result['saved_csv_paths']['rbf_error_analysis']}")
            print(f"saved rbf shap summary csv: {result['saved_csv_paths']['rbf_error_summary']}")
            if "rbf_error_status" in result["saved_csv_paths"]:
                print(f"saved rbf shap status csv: {result['saved_csv_paths']['rbf_error_status']}")
    for warning_message in result.get("output_warnings", []):
        print(f"warning: {warning_message}")


if __name__ == "__main__":
    runtime_config = prompt_experiment_configuration()
    print_experiment_configuration(runtime_config)

    documents, labels, doc_ids = load_corpus()
    active_conditions = get_active_conditions(
        runtime_config["kernel_mode"],
        runtime_config["feature_mode"],
    )
    log_progress("[main] corpus loaded; building split contexts")
    outer_fold_contexts = build_outer_fold_contexts(documents, labels, doc_ids)

    fixed_root = SAVE_DIR / FIXED_DIR_NAME
    tuned_root = SAVE_DIR / TUNED_DIR_NAME
    threshold_count_paths = save_threshold_term_count_outputs(
        outer_fold_contexts,
        ABS_SCORE_THRESHOLDS,
        SAVE_DIR / THRESHOLD_COUNT_DIR_NAME,
    )

    log_progress(f"saved threshold term-count csv: {threshold_count_paths['outer_fold_threshold_term_counts']}")
    log_progress(f"saved threshold term-count summary csv: {threshold_count_paths['threshold_term_count_summary']}")

    if runtime_config["threshold_mode"] in ("fixed", "both"):
        fixed_summary_rows_by_condition = {condition: [] for condition in active_conditions}
        fixed_best_results_by_condition = {condition: None for condition in active_conditions}

        log_progress("\n[main] running fixed-threshold nested CV")
        log_progress(f"[main] threshold candidates: {ABS_SCORE_THRESHOLDS}")
        for score_threshold in ABS_SCORE_THRESHOLDS:
            log_progress(f"\n[main] fixed threshold = {score_threshold}")
            threshold_root = fixed_root / format_threshold_slug(score_threshold)

            for feature_mode, kernel in active_conditions:
                result = run_nested_cv_for_condition(
                    outer_fold_contexts,
                    feature_mode,
                    kernel,
                    [score_threshold],
                )

                condition_dir = get_condition_metrics_dir(threshold_root / METRICS_DIR_NAME, feature_mode, kernel)
                save_nested_cv_result_csvs(result, condition_dir, "svm")
                run_optional_output_step(
                    result,
                    "selected term export",
                    save_outer_fold_selected_terms,
                    result,
                    outer_fold_contexts,
                    condition_dir / FEATURE_OUTPUT_DIR_NAME,
                )
                run_optional_output_step(
                    result,
                    "misclassified document export",
                    save_outer_fold_misclassified_documents,
                    result,
                    condition_dir / MISCLASSIFIED_OUTPUT_DIR_NAME,
                )
                run_optional_output_step(
                    result,
                    "misclassification feature analysis",
                    save_outer_fold_misclassification_feature_analysis,
                    result,
                    outer_fold_contexts,
                    feature_mode,
                    kernel,
                    condition_dir / ERROR_ANALYSIS_OUTPUT_DIR_NAME,
                )
                print_result(result, f"{format_condition_label(feature_mode, kernel)} / 固定閾値={score_threshold}")

                summary_row = build_fixed_threshold_summary_row(score_threshold, result)
                fixed_summary_rows_by_condition[(feature_mode, kernel)].append(summary_row)

                if result.get("selected_records"):
                    current_best = fixed_best_results_by_condition[(feature_mode, kernel)]
                    if current_best is None or result["mean_accuracy"] > current_best["raw_accuracy"]:
                        fixed_best_results_by_condition[(feature_mode, kernel)] = {
                            **summary_row,
                            "raw_accuracy": result["mean_accuracy"],
                        }

        save_fixed_threshold_global_outputs(
            fixed_summary_rows_by_condition,
            fixed_best_results_by_condition,
            fixed_root,
        )

    if runtime_config["threshold_mode"] in ("tuned", "both"):
        tuned_summary_rows = []
        log_progress("\n[main] running tuned-threshold nested CV")
        log_progress(f"[main] threshold candidates: {ABS_SCORE_THRESHOLDS}")
        for feature_mode, kernel in active_conditions:
            result = run_nested_cv_for_condition(
                outer_fold_contexts,
                feature_mode,
                kernel,
                ABS_SCORE_THRESHOLDS,
            )

            condition_dir = get_condition_metrics_dir(tuned_root / METRICS_DIR_NAME, feature_mode, kernel)
            save_nested_cv_result_csvs(result, condition_dir, "svm")
            run_optional_output_step(
                result,
                "selected term export",
                save_outer_fold_selected_terms,
                result,
                outer_fold_contexts,
                condition_dir / FEATURE_OUTPUT_DIR_NAME,
            )
            run_optional_output_step(
                result,
                "misclassified document export",
                save_outer_fold_misclassified_documents,
                result,
                condition_dir / MISCLASSIFIED_OUTPUT_DIR_NAME,
            )
            run_optional_output_step(
                result,
                "misclassification feature analysis",
                save_outer_fold_misclassification_feature_analysis,
                result,
                outer_fold_contexts,
                feature_mode,
                kernel,
                condition_dir / ERROR_ANALYSIS_OUTPUT_DIR_NAME,
            )
            print_result(result, f"{format_condition_label(feature_mode, kernel)} / 閾値チューニング")
            tuned_summary_rows.append(build_tuned_threshold_summary_row(result))

        save_tuned_threshold_global_outputs(tuned_summary_rows, tuned_root)
