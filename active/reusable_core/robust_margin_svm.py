import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def _build_svc(kernel, c, gamma=None, class_weight="balanced", tol=1e-10):
    kwargs = {
        "kernel": kernel,
        "C": c,
        "class_weight": class_weight,
        "tol": tol,
    }
    if gamma is not None:
        kwargs["gamma"] = gamma
    return svm.SVC(**kwargs)


def _compute_margins(model, X_train, y_train):
    decision_values = model.decision_function(X_train)
    y_signed = np.where(y_train == 0, -1, 1)
    return y_signed * decision_values


def _format_param_label(c, gamma=None):
    if gamma is None:
        return f"C={c}"
    return f"C={c}, gamma={gamma}"


def _round_half_up(value, digits=3):
    if value in ("", None):
        return ""
    quantize_exp = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantize_exp, rounding=ROUND_HALF_UP))


def _save_result_csvs(result, output_dir, output_prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{output_prefix}_final_metrics.csv"
    selected_path = output_dir / f"{output_prefix}_selected_fold_results.csv"
    params_path = output_dir / f"{output_prefix}_param_status.csv"

    metrics_row = {
        "kernel": result["kernel"],
        "margin_threshold": result["margin_threshold"],
        "mean_accuracy": _round_half_up(result.get("mean_accuracy", "")),
        "mean_recall": _round_half_up(result.get("mean_recall", "")),
        "mean_precision": _round_half_up(result.get("mean_precision", "")),
        "mean_f1": _round_half_up(result.get("mean_f1", "")),
        "best_cs": ",".join(map(str, result.get("best_cs", []))),
        "best_gammas": ",".join(map(str, result.get("best_gammas", []))) if "best_gammas" in result else "",
    }

    with open(metrics_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)

    selected_fieldnames = [
        "outer_fold",
        "param_label",
        "c",
        "gamma",
        "inner_score",
        "accuracy",
        "recall",
        "precision",
        "f1_score",
    ]
    with open(selected_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=selected_fieldnames)
        writer.writeheader()
        for record in result.get("selected_records", []):
            writer.writerow(
                {
                    "outer_fold": record["outer_fold"],
                    "param_label": record["param_label"],
                    "c": record["c"],
                    "gamma": record["gamma"],
                    "inner_score": _round_half_up(record["inner_score"]),
                    "accuracy": _round_half_up(record["accuracy"]),
                    "recall": _round_half_up(record["recall"]),
                    "precision": _round_half_up(record["precision"]),
                    "f1_score": _round_half_up(record["f1_score"]),
                }
            )

    param_fieldnames = ["status", "param_label"]
    with open(params_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=param_fieldnames)
        writer.writeheader()
        for label in result.get("valid_param_labels", []):
            writer.writerow({"status": "valid", "param_label": label})
        for label in result.get("dropped_param_labels", []):
            writer.writerow({"status": "dropped", "param_label": label})

    result["saved_csv_paths"] = {
        "metrics": str(metrics_path),
        "selected_records": str(selected_path),
        "param_status": str(params_path),
    }


def _evaluate_param_across_outer_folds(
    *,
    kernel,
    c,
    gamma,
    X,
    y,
    outer_splits,
    margin_threshold,
    n_splits,
    random_state,
    class_weight,
    tol,
):
    fold_records = []

    for outer_fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        inner_scores = []
        X_train_outer = X[train_index]
        y_train_outer = y[train_index]

        for inner_train_index, inner_test_index in inner_cv.split(X_train_outer, y_train_outer):
            inner_train_global = train_index[inner_train_index]
            inner_test_global = train_index[inner_test_index]

            X_train_inner = X[inner_train_global]
            y_train_inner = y[inner_train_global]
            X_test_inner = X[inner_test_global]
            y_test_inner = y[inner_test_global]

            model_inner = _build_svc(
                kernel,
                c,
                gamma,
                class_weight=class_weight,
                tol=tol,
            )
            model_inner.fit(X_train_inner, y_train_inner)
            margins_inner = _compute_margins(model_inner, X_train_inner, y_train_inner)
            if not np.all(margins_inner >= margin_threshold):
                return None

            inner_scores.append(model_inner.score(X_test_inner, y_test_inner))

        model_outer = _build_svc(
            kernel,
            c,
            gamma,
            class_weight=class_weight,
            tol=tol,
        )
        model_outer.fit(X_train_outer, y_train_outer)
        margins_outer = _compute_margins(model_outer, X_train_outer, y_train_outer)
        if not np.all(margins_outer >= margin_threshold):
            return None

        y_pred = model_outer.predict(X[test_index])
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "c": c,
                "gamma": gamma,
                "param_label": _format_param_label(c, gamma),
                "inner_score": float(sum(inner_scores) / len(inner_scores)),
                "accuracy": float(accuracy_score(y[test_index], y_pred)),
                "recall": float(recall_score(y[test_index], y_pred, zero_division=0)),
                "precision": float(precision_score(y[test_index], y_pred, zero_division=0)),
                "f1_score": float(f1_score(y[test_index], y_pred, zero_division=0)),
            }
        )

    return fold_records


def run_robust_margin_nested_cv(
    *,
    X,
    y,
    kernel,
    c_values,
    gamma_values=None,
    n_splits=10,
    random_state=42,
    margin_threshold=0.9990,
    class_weight="balanced",
    tol=1e-10,
    output_dir=None,
    output_prefix=None,
):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_splits = list(outer_cv.split(X, y))
    gamma_candidates = gamma_values if gamma_values is not None else [None]
    param_grid = [(c, gamma) for c in c_values for gamma in gamma_candidates]

    valid_param_results = {}
    dropped_param_labels = []

    for c, gamma in param_grid:
        fold_records = _evaluate_param_across_outer_folds(
            kernel=kernel,
            c=c,
            gamma=gamma,
            X=X,
            y=y,
            outer_splits=outer_splits,
            margin_threshold=margin_threshold,
            n_splits=n_splits,
            random_state=random_state,
            class_weight=class_weight,
            tol=tol,
        )
        if fold_records is None:
            dropped_param_labels.append(_format_param_label(c, gamma))
            continue
        valid_param_results[(c, gamma)] = fold_records

    result = {
        "kernel": kernel,
        "margin_threshold": margin_threshold,
        "valid_param_labels": [_format_param_label(c, gamma) for c, gamma in valid_param_results.keys()],
        "dropped_param_labels": dropped_param_labels,
        "selected_records": [],
        "best_cs": [],
    }

    if not valid_param_results:
        if output_dir is not None and output_prefix:
            _save_result_csvs(result, output_dir, output_prefix)
        return result

    selected_records = []
    for fold_idx in range(len(outer_splits)):
        best_record = None
        for records in valid_param_results.values():
            candidate = records[fold_idx]
            if best_record is None:
                best_record = candidate
                continue
            if candidate["inner_score"] > best_record["inner_score"]:
                best_record = candidate
                continue
            if (
                candidate["inner_score"] == best_record["inner_score"]
                and candidate["param_label"] < best_record["param_label"]
            ):
                best_record = candidate
        selected_records.append(best_record)

    result["selected_records"] = selected_records
    result["best_cs"] = [record["c"] for record in selected_records]
    if gamma_values is not None:
        result["best_gammas"] = [record["gamma"] for record in selected_records]

    result["mean_accuracy"] = float(np.mean([record["accuracy"] for record in selected_records]))
    result["mean_recall"] = float(np.mean([record["recall"] for record in selected_records]))
    result["mean_precision"] = float(np.mean([record["precision"] for record in selected_records]))
    result["mean_f1"] = float(np.mean([record["f1_score"] for record in selected_records]))

    if output_dir is not None and output_prefix:
        _save_result_csvs(result, output_dir, output_prefix)

    return result
