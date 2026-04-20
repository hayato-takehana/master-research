import os
import sys
from pathlib import Path


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths, get_output_dir

bootstrap_project_paths(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ==========================================================
# 1. 外部モジュールのインポート
# ==========================================================
# 【重要】再利用コアの dataset_loader.py から common 関数を読み込みます。
from dataset_loader import common, load_document_filenames
from pdf_text_loader import normalize_text


# ==========================================================
# 2. データ管理クラス (DataManager)
# ==========================================================
class DataManager:
    """
    外部のcommon関数を利用してデータを読み込み、
    特定の単語による特徴量抽出や、不要なデータ（外れ値）の削除を行うクラスです。
    """

    def __init__(self, real_scam=False, tf_idf_true=False, keep_digits=False, percent_mode="drop"):
        self.real_scam = real_scam
        self.tf_idf_true = tf_idf_true
        self.keep_digits = keep_digits
        self.percent_mode = percent_mode
        self.M = None
        self.labels = None
        self.feature_names = None
        self.vocab_index = {}
        self.document_filenames = None

    def load_data(self):
        """common関数を呼び出してデータをロード"""
        self.M, self.labels, self.feature_names, _ = common(
            self.real_scam,
            self.tf_idf_true,
            keep_digits=self.keep_digits,
            percent_mode=self.percent_mode,
        )
        label1_files, label0_files = load_document_filenames(real_scam=self.real_scam)
        self.document_filenames = label1_files + label0_files
        self.labels = np.array(self.labels).astype(int)
        self.vocab_index = {w: i for i, w in enumerate(self.feature_names)}
        print(f"データロード完了: 文書数={self.M.shape[0]}, 語彙数={len(self.feature_names)}")
        return self.M, self.labels, self.feature_names

    def extract_specific_features(self, target_words):
        """指定された単語リストに対応する列のみを抽出する"""
        if self.M is None:
            raise ValueError("先に load_data() を実行してください。")

        target_indices = []
        found_words = []
        normalized_pairs = []
        seen_words = set()

        print("\n--- 特徴量抽出処理 ---")
        for word in target_words:
            normalized_word = normalize_text(
                word,
                keep_digits=self.keep_digits,
                percent_mode=self.percent_mode,
            ).lower()

            if not normalized_word:
                print(f"除外（正規化で空文字）: {word}")
                continue

            if normalized_word in seen_words:
                continue
            seen_words.add(normalized_word)
            normalized_pairs.append((word, normalized_word))

            if normalized_word in self.vocab_index:
                target_indices.append(self.vocab_index[normalized_word])
                found_words.append(normalized_word)
            else:
                print(f"除外（語彙に無し）: {word} -> {normalized_word}")

        X_all = self.M.toarray() if hasattr(self.M, "toarray") else self.M
        X_selected = X_all[:, target_indices]

        print(f"抽出対象単語: {len(target_words)}語 -> 実際に抽出された単語: {len(found_words)}語")
        print(f"抽出後のデータ形状 X: {X_selected.shape}")

        return X_selected, found_words

    def remove_outlier(self, X, y, index_to_remove=152):
        """特定インデックスの行を削除する（外れ値処理）"""
        if X.shape[0] > index_to_remove:
            print(f"削除前: X={X.shape}, labels={y.shape}")
            X_new = np.delete(X, index_to_remove, axis=0)
            y_new = np.delete(y, index_to_remove, axis=0)
            print(f"削除後: X={X_new.shape}, labels={y_new.shape}")
            print(f"インデックス {index_to_remove} を削除しました。")
            return X_new, y_new
        else:
            print(
                f"警告: データ行数({X.shape[0]})が削除インデックス({index_to_remove})以下のため、削除をスキップします。")
            return X, y

    @staticmethod
    def remove_outlier_from_names(document_names, index_to_remove=152):
        if document_names is None:
            return None
        if len(document_names) > index_to_remove:
            return [name for idx, name in enumerate(document_names) if idx != index_to_remove]
        return document_names


# ==========================================================
# 3. 特徴量分析クラス (FeatureAnalyzer)
# ==========================================================
class FeatureAnalyzer:
    """
    特定単語の分布を可視化したり、統計情報を計算するクラス
    """

    def __init__(self, M, labels, feature_names, save_dir="data"):
        self.M = M
        self.labels = labels
        self.feature_names = feature_names
        self.vocab_index = {w: i for i, w in enumerate(feature_names)}
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _extract_counts(self, col_idx, row_indices):
        """指定された行・列の値を1次元ndarrayで返す"""
        sub = self.M[row_indices, col_idx]
        if hasattr(sub, "toarray"):
            return np.asarray(sub.toarray()).ravel()
        return np.asarray(sub).ravel()

    def visualize_word_distribution(self, target_words, save_png=True, filename_suffix=""):
        """単語ごとの出現分布を棒グラフ（色分け）で表示"""
        mask1 = (self.labels == 1)
        mask0 = (self.labels == 0)
        idx1 = np.where(mask1)[0]
        idx0 = np.where(mask0)[0]
        x_all = [f"L1_{i}" for i in range(1, len(idx1) + 1)] + [f"L0_{i}" for i in range(1, len(idx0) + 1)]

        for tok in target_words:
            if tok not in self.vocab_index:
                continue

            j = self.vocab_index[tok]
            vals_l1 = self._extract_counts(j, idx1)
            vals_l0 = self._extract_counts(j, idx0)
            vals_all = np.concatenate([vals_l1, vals_l0])
            colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))

            plt.figure(figsize=(10, 4))
            plt.bar(range(len(vals_all)), vals_all, color=colors)
            plt.xticks(range(len(vals_all)), x_all, rotation=90, fontsize=4)
            plt.title(f"Dist: '{tok}'")
            plt.legend(
                handles=[Patch(facecolor="blue", label="Sagi (L1)"), Patch(facecolor="red", label="Not Sagi (L0)")])
            plt.tight_layout()

            if save_png:
                plt.savefig(os.path.join(self.save_dir,f"{filename_suffix}_{tok}_グラフ.png"), dpi=100)
            #plt.show()
            plt.close()

    def visualize_group_total_distribution(self, target_words, save_png=True, filename_suffix=""):
        """指定された単語リストに含まれる単語の「合計出現回数」を表示"""
        target_indices = []
        for word in target_words:
            if word in self.vocab_index:
                target_indices.append(self.vocab_index[word])

        if not target_indices:
            print("指定された単語はデータ内に存在しません。")
            return

        M_sub = self.M[:, target_indices]
        doc_totals = np.array(M_sub.sum(axis=1)).ravel()

        mask1 = (self.labels == 1)
        mask0 = (self.labels == 0)
        idx1 = np.where(mask1)[0]
        idx0 = np.where(mask0)[0]

        vals_l1 = doc_totals[mask1]
        vals_l0 = doc_totals[mask0]
        vals_all = np.concatenate([vals_l1, vals_l0])

        x_all = [f"L1_{i}" for i in range(1, len(idx1) + 1)] + [f"L0_{i}" for i in range(1, len(idx0) + 1)]
        colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(vals_all)), vals_all, color=colors)
        plt.xticks(range(len(vals_all)), x_all, rotation=90, fontsize=4)
        plt.ylabel("Total Frequency (Group Sum)")
        plt.title(f"Distribution of Selected Words Group Sum (Total {len(target_indices)} words)")
        plt.legend(handles=[Patch(facecolor="blue", label="Sagi (L1)"), Patch(facecolor="red", label="Not Sagi (L0)")])
        plt.tight_layout()

        if save_png:
            filename = f"{filename_suffix}_特徴量全ての合計のグラフ.png"
            plt.savefig(os.path.join(self.save_dir,filename), dpi=100)
            print(f"グラフを保存しました: {filename}")
        plt.show()
        plt.close()

    def calculate_90percentile_scores(self, target_words):
        """L1(詐欺)の90パーセンタイルを閾値としたスコア計算"""
        results = []
        mask1 = (self.labels == 1)
        mask0 = (self.labels == 0)
        total_l1 = np.sum(mask1)
        total_l0 = np.sum(mask0)

        for word in target_words:
            if word not in self.vocab_index:
                continue

            j = self.vocab_index[word]
            counts_l1 = self._extract_counts(j, np.where(mask1)[0])
            counts_l0 = self._extract_counts(j, np.where(mask0)[0])

            n = np.percentile(counts_l1, 90) if len(counts_l1) > 0 else 0
            r_l1 = np.sum(counts_l1 > n) / total_l1 if total_l1 > 0 else 0
            r_l0 = np.sum(counts_l0 > n) / total_l0 if total_l0 > 0 else 0

            results.append({
                "word": word,
                "threshold": n,
                "ratio_L0": r_l0,
                "ratio_L1": r_l1,
                "score(L0-L1)": r_l0 - r_l1
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="score(L0-L1)", ascending=False)
        return df

    def analyze_target_words_statistics(self, target_words):
        """指定された単語リストの個別統計"""
        results = []
        mask1 = (self.labels == 1)
        mask0 = (self.labels == 0)
        idx1 = np.where(mask1)[0]
        idx0 = np.where(mask0)[0]
        n_l1 = len(idx1)
        n_l0 = len(idx0)

        for word in target_words:
            if word not in self.vocab_index:
                # print(f"Warning: '{word}' はデータ内に存在しません。") # ログが多い場合はコメントアウト
                continue

            j = self.vocab_index[word]
            vals_l1 = self._extract_counts(j, idx1)
            vals_l0 = self._extract_counts(j, idx0)

            mean_l1 = np.mean(vals_l1) if n_l1 > 0 else 0.0
            mean_l0 = np.mean(vals_l0) if n_l0 > 0 else 0.0
            diff_mean = mean_l0 - mean_l1

            count_l1 = np.count_nonzero(vals_l1)
            count_l0 = np.count_nonzero(vals_l0)
            diff_count = count_l0 - count_l1

            ratio_l1 = count_l1 / n_l1 if n_l1 > 0 else 0.0
            ratio_l0 = count_l0 / n_l0 if n_l0 > 0 else 0.0
            diff_ratio = ratio_l0 - ratio_l1

            results.append({
                "word": word,
                "詐欺じゃないグループの平均単語出現回数": mean_l0,
                "詐欺グループの平均単語出現回数": mean_l1,
                "平均単語出現頻度の差(詐欺じゃない-詐欺)": diff_mean,
                "詐欺じゃないグループの平均文書数（割合）": ratio_l0,
                "詐欺グループの平均文書数（割合）": ratio_l1,
                "出現文書数の割合の差(詐欺じゃない-詐欺)": diff_ratio,
                "詐欺じゃないグループの出現文書数": count_l0,
                "詐欺グループの出現文書数": count_l1,
                "出現文書数の差(詐欺じゃない-詐欺)": diff_count
            })

        df = pd.DataFrame(results)
        cols = [
            "word",
            "詐欺じゃないグループの平均単語出現回数", "詐欺グループの平均単語出現回数",
            "平均単語出現頻度の差(詐欺じゃない-詐欺)",
            "詐欺じゃないグループの平均文書数（割合）", "詐欺グループの平均文書数（割合）",
            "出現文書数の割合の差(詐欺じゃない-詐欺)",
            "詐欺じゃないグループの出現文書数", "詐欺グループの出現文書数", "出現文書数の差(詐欺じゃない-詐欺)"
        ]

        if not df.empty:
            df = df[cols]
            df = df.sort_values(by="出現文書数の割合の差(詐欺じゃない-詐欺)", ascending=False)
        return df

    def analyze_group_statistics(self, target_words):
        """グループ全体の統計"""
        target_indices = []
        valid_words = []
        for word in target_words:
            if word in self.vocab_index:
                target_indices.append(self.vocab_index[word])
                valid_words.append(word)

        if not target_indices:
            print("指定された単語はデータ内に存在しません。")
            return None

        M_sub = self.M[:, target_indices]
        mask1 = (self.labels == 1)
        mask0 = (self.labels == 0)
        n_l1 = np.sum(mask1)
        n_l0 = np.sum(mask0)

        doc_totals = np.array(M_sub.sum(axis=1)).ravel()
        vals_l1 = doc_totals[mask1]
        vals_l0 = doc_totals[mask0]

        mean_sum_l1 = np.mean(vals_l1) if n_l1 > 0 else 0.0
        mean_sum_l0 = np.mean(vals_l0) if n_l0 > 0 else 0.0
        diff_mean_sum = mean_sum_l0 - mean_sum_l1

        cov_l1 = np.count_nonzero(vals_l1) / n_l1 if n_l1 > 0 else 0.0
        cov_l0 = np.count_nonzero(vals_l0) / n_l0 if n_l0 > 0 else 0.0
        diff_cov = cov_l0 - cov_l1

        M_sub_l1 = M_sub[mask1, :]
        M_sub_l0 = M_sub[mask0, :]
        if hasattr(M_sub_l1, "getnnz"):
            word_counts_l1 = M_sub_l1.getnnz(axis=0)
            word_counts_l0 = M_sub_l0.getnnz(axis=0)
        else:
            word_counts_l1 = np.count_nonzero(M_sub_l1, axis=0)
            word_counts_l0 = np.count_nonzero(M_sub_l0, axis=0)

        ratios_l1 = word_counts_l1 / n_l1 if n_l1 > 0 else np.zeros(len(valid_words))
        ratios_l0 = word_counts_l0 / n_l0 if n_l0 > 0 else np.zeros(len(valid_words))

        avg_ratio_l1 = np.mean(ratios_l1)
        avg_ratio_l0 = np.mean(ratios_l0)
        diff_avg_ratio = avg_ratio_l0 - avg_ratio_l1

        result = {
            "Group Name": "Selected Words Group",
            "Num Words": len(valid_words),
            "詐欺じゃないグループの平均単語出現回数": mean_sum_l0,
            "詐欺グループの平均単語出現回数": mean_sum_l1,
            "平均単語出現回数の差(詐欺じゃない-詐欺)": diff_mean_sum,
            "詐欺じゃないグループの平均出現文書数割合": avg_ratio_l0,
            "詐欺グループの平均出現文書数割合": avg_ratio_l1,
            "出現文書数の割合の差(詐欺じゃない-詐欺)": diff_avg_ratio,
            "詐欺じゃないグループのカバー率": cov_l0,
            "詐欺グループのカバー率": cov_l1,
            "カバー率の差(詐欺じゃないグループ-詐欺)": diff_cov
        }

        cols = [
            "Num Words",
            "詐欺じゃないグループの平均単語出現回数", "詐欺グループの平均単語出現回数",
            "平均単語出現回数の差(詐欺じゃない-詐欺)",
            "詐欺じゃないグループの平均出現文書数割合", "詐欺グループの平均出現文書数割合",
            "出現文書数の割合の差(詐欺じゃない-詐欺)",
            "詐欺じゃないグループのカバー率", "詐欺グループのカバー率", "カバー率の差(詐欺じゃないグループ-詐欺)"
        ]

        return pd.DataFrame([result])[cols]

    #詐欺群の一番多い値から上を採用することで何％の詐欺じゃない群があるかを見つけるための関数
    def visualize_group_total_with_max_sagiline(self, target_words, thresholds=None, save_png=True, filename_suffix=""):
        """
        L1(詐欺群)の最大値および指定閾値を超えた正常データ(L0)の
        件数・割合・単語内訳を算出し、print出力・グラフ表示・CSV出力を行う
        """
        # 1. 準備：インデックス取得と合計計算
        target_indices = [self.vocab_index[w] for w in target_words if w in self.vocab_index]
        if not target_indices:
            print("指定された単語はデータ内に存在しません。")
            return

        # 指定単語のみの行列から各文書の合計出現回数を計算
        M_sub = self.M[:, target_indices]
        doc_totals = np.array(M_sub.sum(axis=1)).ravel()

        # 2. ラベルごとに分離
        mask1 = (self.labels == 1);
        mask0 = (self.labels == 0)
        vals_l1 = doc_totals[mask1];
        vals_l0 = doc_totals[mask0]
        total_l0_count = len(vals_l0)

        # 3. 閾値リストの整理 (デフォルトでL1最大値を含める)
        max_l1_val = np.max(vals_l1) if len(vals_l1) > 0 else 0
        eval_thresholds = [max_l1_val]
        if thresholds is not None:
            if isinstance(thresholds, (list, np.ndarray)):
                eval_thresholds.extend(thresholds)
            else:
                eval_thresholds.append(thresholds)
        eval_thresholds = sorted(list(set(eval_thresholds)))

        # 4. 詳細分析とデータ集計
        all_analysis_results = []
        stats_messages = []
        idx0 = np.where(mask0)[0]

        print(f"\n--- 閾値分析レポート (L1最大値: {max_l1_val}) ---")

        for thr in eval_thresholds:
            # 閾値「より大きい」の正常データを特定
            over_l0_mask_in_l0 = (vals_l0 > thr)
            over_l0_indices = idx0[over_l0_mask_in_l0]

            over_l0_count = len(over_l0_indices)
            over_l0_ratio = (over_l0_count / total_l0_count) * 100 if total_l0_count > 0 else 0

            # --------------------------------------------------
            # 【ご要望部分】件数と割合のprint出力
            # --------------------------------------------------
            print(f"閾値: {thr:.1f}")
            print(f"  正常データ数: {over_l0_count}件 / {total_l0_count}件")
            print(f"  正常データの割合: {over_l0_ratio:.2f}%")

            msg = f"Thr {thr:.1f}: {over_l0_count}件 ({over_l0_ratio:.1f}%)"
            stats_messages.append(msg)

            # 各該当データの単語内訳を調査
            for global_idx in over_l0_indices:
                sub_data = self.M[global_idx, target_indices]
                if hasattr(sub_data, "toarray"):
                    row_data = sub_data.toarray().ravel()
                else:
                    row_data = np.asarray(sub_data).ravel()

                word_counts = [f"{target_words[i]}:{int(row_data[i])}" for i in range(len(row_data)) if row_data[i] > 0]

                all_analysis_results.append({
                    "Threshold": thr,
                    "Is_L1_Max": thr == max_l1_val,
                    "Global_Index": global_idx,
                    "Total_Count": doc_totals[global_idx],
                    "L0_Over_Count": over_l0_count,
                    "L0_Over_Ratio(%)": over_l0_ratio,
                    "Word_Breakdown": ", ".join(word_counts)
                })

        # CSV保存
        if all_analysis_results:
            df_export = pd.DataFrame(all_analysis_results)
            csv_filename = f"csv_{filename_suffix}_詐欺群の最大で線を引いた特徴量全ての合計のグラフ_.csv"
            df_export.to_csv(os.path.join(self.save_dir, csv_filename), index=False, encoding='utf-8-sig')
            print(f"\n >> 詳細データを '{csv_filename}' に保存しました。")

        # グラフ描画
        plt.figure(figsize=(14, 7))
        vals_all = np.concatenate([vals_l1, vals_l0])
        colors = (["blue"] * len(vals_l1)) + (["red"] * len(vals_l0))
        plt.bar(range(len(vals_all)), vals_all, color=colors)

        for i, thr in enumerate(eval_thresholds):
            color = 'green' if thr == max_l1_val else plt.cm.viridis(i / len(eval_thresholds))
            plt.axhline(y=thr, color=color, linestyle='--', linewidth=1.5, label=f"Thr: {thr:.1f}")

        plt.text(0.02, 0.98, "\n".join(stats_messages), transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=9)

        plt.ylabel("Total Frequency")
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_png: plt.savefig(os.path.join(self.save_dir,f"{filename_suffix}_指定した閾値以上の全ての特徴.png"), dpi=100)
        plt.show()
        plt.close()



# ==========================================================
# 4. 学習クラス (StrictSVMTrainer)
# ==========================================================
class StrictSVMTrainer:
    """
    独自のマージン条件付きSVM学習、および詳細なモデル分析を行うクラス
    """

    def __init__(self, n_splits=10, random_state=42, save_dir="data"):
        self.inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.margin_threshold = 0.9990
        self.save_dir=save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _check_strict_margin(self, model, X_train, y_train):
        """学習データ上で、すべてのデータ点のマージンが閾値以上かチェック"""
        decision_values = model.decision_function(X_train)
        y_dist = np.where(y_train == 0, -1, 1)
        # マージン $M = y \cdot f(x)$
        margin = y_dist * decision_values
        return np.all(margin >= self.margin_threshold)

    def run_custom_cv(self, X, y, kernel='linear', C_list=None, Gamma_list=None):
        """独自ロジックのダブルクロスバリデーション（既存機能）"""
        print(f"\n=== Custom CV Start (Kernel: {kernel}) ===")
        if C_list is None: C_list = [1000]
        if Gamma_list is None: Gamma_list = [None]
        results = {"accuracy": [], "recall": [], "precision": [], "f1": [], "best_params": []}
        valid_outer_loop = True

        for train_idx, test_idx in self.outer_cv.split(X, y):
            best_score, best_params, found_valid_param = 0, {}, False
            for c in C_list:
                for gamma in Gamma_list:
                    scores, is_param_valid = [], True
                    for in_tr_idx, in_te_idx in self.inner_cv.split(X[train_idx], y[train_idx]):
                        real_tr_idx, real_te_idx = train_idx[in_tr_idx], train_idx[in_te_idx]
                        kwargs = {'kernel': kernel, 'C': c, 'class_weight': 'balanced', 'tol': 1e-10}
                        if gamma is not None: kwargs['gamma'] = gamma
                        clf = svm.SVC(**kwargs)
                        clf.fit(X[real_tr_idx], y[real_tr_idx])
                        if self._check_strict_margin(clf, X[real_tr_idx], y[real_tr_idx]):
                            scores.append(clf.score(X[real_te_idx], y[real_te_idx]))
                        else:
                            is_param_valid = False;
                            break
                    if is_param_valid and len(scores) > 0:
                        avg_score = sum(scores) / len(scores)
                        if avg_score > best_score:
                            best_score, best_params, found_valid_param = avg_score, {'C': c, 'gamma': gamma}, True
            if not found_valid_param:
                valid_outer_loop = False;
                break
            kwargs_out = {'kernel': kernel, 'C': best_params['C'], 'class_weight': 'balanced', 'tol': 1e-10}
            if best_params['gamma'] is not None: kwargs_out['gamma'] = best_params['gamma']
            clf_final = svm.SVC(**kwargs_out)
            clf_final.fit(X[train_idx], y[train_idx])
            if self._check_strict_margin(clf_final, X[train_idx], y[train_idx]):
                y_pred = clf_final.predict(X[test_idx])
                results["accuracy"].append(accuracy_score(y[test_idx], y_pred))
                results["recall"].append(recall_score(y[test_idx], y_pred))
                results["precision"].append(precision_score(y[test_idx], y_pred))
                results["f1"].append(f1_score(y[test_idx], y_pred))
                results["best_params"].append(best_params)
            else:
                valid_outer_loop = False;
                break
        return results if valid_outer_loop else None

    def perform_error_analysis(self, X, y, feature_names, kernel='linear', C=1000, document_names=None):
        """マージン条件違反データの特定と寄与度分解"""
        print("\n===　誤判定データ及びマージン内のデータを出力。要因となる特徴の重みと値を出力する  ===")
        clf = svm.SVC(kernel=kernel, C=C, class_weight='balanced')
        clf.fit(X, y)
        scores, predictions = clf.decision_function(X), clf.predict(X)
        y_signed = np.where(y == 0, -1, 1)
        margins = y_signed * scores
        error_indices = np.where(margins < self.margin_threshold)[0]

        if len(error_indices) == 0:
            print("すべてのデータが条件をクリアしています。");
            return None

        weights, bias = clf.coef_[0], clf.intercept_[0]
        error_details = []
        for idx in error_indices:
            contributions = X[idx] * weights
            top_idx = np.argsort(np.abs(contributions))[::-1][:5]
            top_factors = [f"{feature_names[i]}(val={X[idx][i]:.2f}, w={weights[i]:.2f})" for i in top_idx]
            error_details.append({
                "Index": idx, "True": y[idx], "Pred": predictions[idx],
                "Document_Name": document_names[idx] if document_names is not None and idx < len(document_names) else "",
                "Score": scores[idx], "Margin": margins[idx], "Top_Factors": " | ".join(top_factors)
            })
        df = pd.DataFrame(error_details).sort_values(by="Margin")
        df.to_csv(os.path.join(self.save_dir,"誤判定データの確認.csv"), index=False, encoding='utf-8-sig')
        print(f"違反データ数: {len(error_indices)} / レポート保存完了")
        return df

    def analyze_global_weights(self, model, feature_names, top_n=15, save_png=True):
        """モデル全体の重要単語ランキングと可視化"""
        print("\n=== 各特徴量の重みを分布で確認する ===")
        weights = model.coef_[0]
        df_w = pd.DataFrame({"Word": feature_names, "Weight": weights}).sort_values(by="Weight", ascending=False)
        df_w.to_csv(os.path.join(self.save_dir,"特徴の重みについて.csv"), index=False, encoding='utf-8-sig')

        df_plot = pd.concat([df_w.head(top_n), df_w.tail(top_n)])
        plt.figure(figsize=(10, 8))
        plt.barh(df_plot['Word'], df_plot['Weight'], color=['blue' if x > 0 else 'red' for x in df_plot['Weight']])
        plt.title(f"Top {top_n} Positive & Negative Weights");
        plt.tight_layout();
        if save_png: plt.savefig(os.path.join(self.save_dir,f"weight_SVM.png"), dpi=100)
        plt.show()
        plt.close()
        return df_w

    def inspect_specific_data(self, model, X, y, feature_names, target_index):
        """特定1件のデータの判定根拠を深掘り"""
        print(f"\n=== あるデータ番号 {target_index}のより詳しい要因分析 ===")
        x_vec, weights, bias = X[target_index], model.coef_[0], model.intercept_[0]
        contributions = x_vec * weights
        df_ins = pd.DataFrame({"Word": feature_names, "Value": x_vec, "Weight": weights, "Contribution": contributions})
        df_ins = df_ins.assign(Abs=df_ins["Contribution"].abs()).sort_values(by="Abs", ascending=False).drop(
            columns="Abs")
        print(
            f"正解: {y[target_index]}, 予測: {model.predict([x_vec])[0]}, スコア: {model.decision_function([x_vec])[0]:.4f}")
        print(df_ins.head(10).to_string(index=False))
        df_ins.to_csv(os.path.join(self.save_dir,f"データ番号{target_index}の重みと各特徴量の値.csv"), index=False, encoding='utf-8-sig')


# ==========================================================
# 5. メイン実行ブロック
# ==========================================================
if __name__ == "__main__":
    # ----------------------------------------------------------
    # 表示設定の変更
    # ----------------------------------------------------------
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合

    # --------------------------------------------------
    # 設定パラメータ
    # --------------------------------------------------
    #ターゲット単語
    #辞書から引っ張ってくるときは以下のルールを守ること
    #-は消して繋げること　例) big-tree → bigtree
    #大文字は小文字に変換すること
    target_tokens = ["beacon node", "block builder", "proposer", "builder api specification", "committee",
                     "execution payload", "fork choice rule", "liveness", "middleware", "mevboost",
                     "plausible liveness", "probabilistic liveness", "blockbuilder separation", "pbs", "relay",
                     "searcher", "staker", "validator client", "validator pubkey", "51% attack", "account", "address",
                     "application binary interface", "abi", "antisybil", "assert", "base fee", "beacon chain",
                     "bigendian", "block", "block explorer", "block header", "block propagation", "block proposer",
                     "block reward", "block status", "block time", "block validation", "bootnode", "bridge", "bytecode",
                     "byzantium fork", "casper ffg", "checkpoint", "computational infeasibility", "consensus",
                     "consensus client", "consensus layer", "consensus rules", "constantinople fork",
                     "contract account", "contract creation transaction", "cryptography", "dag", "data availability",
                     "decentralization", "decentralized autonomous organization", "dao", "desci",
                     "decentralized exchange", "dex", "deposit contract", "defi", "difficulty", "digital signature",
                     "discovery", "distributed hash table", "dht", "double spend",
                     "elliptic curve digital signature algorithm", "ecdsa", "epoch", "equivocation", "eth1", "eth2",
                     "ethereum improvement proposal", "eip", "ethereum name service", "ens", "execution client",
                     "execution layer", "externally owned account", "eoa", "ethereum request for comments", "erc-20",
                     "erc-721", "erc-1155", "ethash", "ethereum virtual machine", "evm", "evm assembly language",
                     "faucet", "finality", "finney", "fork", "fork choice algorithm", "fraud proof", "gas", "gas limit",
                     "gas price", "genesis block", "gwei", "hard fork", "hash", "hash rate", "holographic consensus",
                     "homestead", "index", "integrated development environment", "ide",
                     "immutable deployed code problem", "internal transaction", "key derivation function", "kdf", "key",
                     "keccak-256", "layer 1", "layer 2", "library", "light client", "liquidity", "liquidity tokens",
                     "lmdghost", "mainnet", "max fee per gas", "merkle patricia tree", "mpt", "merkle root", "message",
                     "message call", "maximal extractable value", "mev", "mining", "miner", "mint", "multisig",
                     "network", "network hashrate", "nonfungible token", "nft", "node", "nonce", "offchain",
                     "ommer block", "ommer uncle block", "onchain", "optimistic rollup", "oracle", "peertopeer network",
                     "permissionless", "plasma", "private key", "poap", "proofofstake", "pos", "proofofwork", "pow",
                     "protodanksharding", "public goods", "public key", "receipt", "reentrancy attack", "reward",
                     "recursive length prefix", "rlp", "rollups", "remote procedure call", "rpc",
                     "secure hash algorithm", "sha", "recovery phrase", "sequencer", "serialization", "shard",
                     "shard chain", "sidechain", "signing", "singleton", "slasher", "slot", "snark", "soft fork",
                     "solidity", "solidity inline assembly", "stablecoin", "staking", "staking pool", "stark", "state",
                     "state channels", "supermajority", "syncing", "sync committee", "szabo",
                     "terminal total difficulty", "ttd", "testnet", "token factory", "transaction", "transaction fee",
                     "trust assumptions", "trustlessness", "turing complete", "validator", "validator lifecycle",
                     "validity proof", "validium", "vyper", "web3", "wei", "wrapped token", "zero address",
                     "zeroknowledge rollup"]
    # --- 重複削除処理を追加 ---
    # target_tokens 定義の直後に追加
    import collections

    # 1. 重複している単語とその回数をカウント
    counter = collections.Counter(target_tokens)
    duplicates = {word: count for word, count in counter.items() if count > 1}

    # 2. 結果を表示
    if duplicates:
        print("\n--- 重複している単語のリスト ---")
        for word, count in duplicates.items():
            print(f"単語: '{word}' (重複回数: {count})")
        print(f"重複単語の種類数: {len(duplicates)}")
    else:
        print("\n重複している単語はありません。")

    original_count = len(target_tokens)
    target_tokens = list(dict.fromkeys(target_tokens))
    new_count = len(target_tokens)

    if original_count != new_count:
        print(f"【情報】重複を除去しました: {original_count}語 -> {new_count}語")



    #その他パラメータ
    USE_REAL_SCAM = False
    USE_TF_IDF = False
    KEEP_DIGITS = True
    PERCENT_MODE = "word"
    # 全体152番目の文書を除外したいので 0-based index を使う
    REMOVE_INDEX = 151
    SAVE_DIR = str(get_output_dir(__file__, PROJECT_ROOT))

    # --------------------------------------------------
    # 処理実行
    # --------------------------------------------------
    #データの読み込み
    # 1. データのロード
    #dataの読み込みと指定単語の抽出および外れ値（データ152の処理）
    print("データの読み込み及び指定単語の抽出をします")
    dm = DataManager(
        real_scam=USE_REAL_SCAM,
        tf_idf_true=USE_TF_IDF,
        keep_digits=KEEP_DIGITS,
        percent_mode=PERCENT_MODE,
    )
    M_raw, labels_raw, feature_names = dm.load_data()

    # 2. 特徴量の抽出 (まだ外れ値は含まれるが、必要な列だけにする)
    # ここで found_words を取得しておく
    print("指定単語の抽出")
    X_selected, found_words = dm.extract_specific_features(target_tokens)

    # 3. 外れ値削除 (Rawデータから指定行を削除)
    print("指定データを削除します")
    X_final, y_final = dm.remove_outlier(X_selected, labels_raw, index_to_remove=REMOVE_INDEX)
    document_names_final = dm.remove_outlier_from_names(dm.document_filenames, index_to_remove=REMOVE_INDEX)
    print("")
    print("")

    # --------------------------------------------------
    # 全特徴量名（語彙リスト）の確認用CSV出力
    # --------------------------------------------------
    print("全特徴量名（語彙リスト）をCSVに保存します...")
    # feature_names（またはfound_wordsではなく、システムが抽出した全単語）をデータフレーム化
    df_all_features = pd.DataFrame(feature_names, columns=["feature_name"])

    # アルファベット順に並び替えておくと確認しやすくなります
    df_all_features = df_all_features.sort_values(by="feature_name")

    # CSVファイルとして保存
    all_features_csv = os.path.join(SAVE_DIR, "all_extracted_features_list.csv")
    df_all_features.to_csv(all_features_csv, index=False, encoding='utf-8-sig')

    print(f" >> 全語彙数: {len(feature_names)}")
    print(f" >> すべての特徴量名を '{all_features_csv}' に保存しました。")




    # ==========================================================
    # 正規化をしない場合の分析
    # ==========================================================
    print("\n--- 正規化前のデータに対する分析 ---")
    # classの定義
    # 削除後のデータ(X_final)を使ってAnalyzerを作成
    # feature_names には抽出済みの found_words を指定
    analyzer = FeatureAnalyzer(X_final, y_final, found_words, save_dir=SAVE_DIR)


    # 詐欺群の上位10%が入ることを許した時の各指定単語がそれぞれの文書割合とその差分を計算する（詐欺群は基本的に0.1がmaxとなる）
    #print("\n--- 指定単語それぞれについて詐欺群の出現回数上位10%の部分で切った時の出現割合とその差 ---")
    #df_scores = analyzer.calculate_90percentile_scores(found_words)
    #print(df_scores)

    # 各指定単語の可視化
    print("\n--- 各指定単語の出現回数の分布を表示 ---")
    analyzer.visualize_word_distribution(found_words, save_png= True, filename_suffix="出現回数")

    # 指定単語ごとの統計 (個別)
    print("\n--- 各指定単語の群ごとの出現回数と出現割合の統計 ---")
    df_individual = analyzer.analyze_target_words_statistics(found_words)
    print(df_individual.to_string())
    df_individual.to_csv(os.path.join(SAVE_DIR,"csv_出現回数_特徴個別の出現回数と出現割合.csv"), index=False, encoding='utf-8-sig')
    print(" >> 'csv_出現回数_特徴個別の出現回数と出現割合.csv' に保存しました。")


    # 指定単語全体の可視化
    print("\n--- 全ての指定単語の出現回数の分布を表示 ---")
    analyzer.visualize_group_total_distribution(found_words, save_png=True, filename_suffix="出現回数")

    # 指定単語グループ全体の統計 (合計)
    print("\n--- 全ての指定単語の出現回数と出現割合の統計 ---")
    df_group = analyzer.analyze_group_statistics(found_words)
    print(df_group.to_string())
    df_group.to_csv(os.path.join(SAVE_DIR, "csv_出現回数_全ての特徴の出現回数と出現割合.csv"), index=False, encoding='utf-8-sig')
    print(" >> 'csv_出現回数_全ての特徴の出現回数と出現割合' に保存しました。")

    print("\n--- 詐欺群の最大の出現回数を記載、その後それ以上の詐欺でない群がどの程度の割合含まれているかを確認する ---")
    #値の確認
    max_line = [440, 450, 500, 550, 600, 650, 700, 800, 1000]
    analyzer.visualize_group_total_with_max_sagiline(found_words, filename_suffix="出現回数", thresholds=max_line)
    print("")
    print("")




    # ==========================================================
    # Min-Max正規化を行った場合の分析
    # ==========================================================
    print("\n--- min-max正規化を行う ---")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_final)  # X_final(外れ値削除後)を使用

    print("\n--- min-max正規化後に行う分析 ---")
    # 正規化後のデータ X_normalized を使って新しいAnalyzerを作成
    analyzer_norm = FeatureAnalyzer(X_normalized, y_final, found_words, save_dir=SAVE_DIR)

    # 詐欺群の上位10%が入ることを許した時の各指定単語がそれぞれの文書割合とその差分を計算する（詐欺群は基本的に0.1がmaxとなる）
    #print("\n--- 指定単語それぞれについて詐欺群の出現回数上位10%の部分で切った時の出現割合とその差 ---")
    #df_scores = analyzer_norm.calculate_90percentile_scores(found_words)
    #print(df_scores)

    # 各指定単語の可視化
    print("\n--- 各指定単語の出現回数の分布を表示 ---")
    analyzer_norm.visualize_word_distribution(found_words, save_png= True, filename_suffix="正規化後")

    # 指定単語ごとの統計 (個別)
    print("\n--- 各指定単語の群ごとの出現回数と出現割合の統計 ---")
    df_individual = analyzer_norm.analyze_target_words_statistics(found_words)
    print(df_individual.to_string())
    df_individual.to_csv(os.path.join(SAVE_DIR, "csv_正規化後_特徴個別の出現回数と出現割合.csv"), index=False, encoding='utf-8-sig')
    print(" >> 'csv_正規化後_特徴個別の出現回数と出現割合' に保存しました。")


    # 指定単語全体の可視化
    print("\n--- 全ての指定単語の出現回数の分布を表示 ---")
    analyzer_norm.visualize_group_total_distribution(found_words, save_png=True, filename_suffix="正規化後")

    # 指定単語グループ全体の統計 (合計)
    print("\n--- 全ての指定単語の出現回数と出現割合の統計 ---")
    df_group = analyzer_norm.analyze_group_statistics(found_words)
    print(df_group.to_string())
    df_group.to_csv(os.path.join(SAVE_DIR, "csv_正規化後_全ての特徴の出現回数と出現割合.csv"), index=False, encoding='utf-8-sig')
    print(" >> 'csv_正規化後_全ての特徴の出現回数と出現割合.csv' に保存しました。")

    print("\n--- 詐欺群の最大の出現回数を記載、その後それ以上の詐欺でない群がどの程度の割合含まれているかを確認する ---")
    #値の確認
    max_line = [9.5, 10.0, 11.0, 12.0, 13.0, 15.0, 20.0]
    analyzer_norm.visualize_group_total_with_max_sagiline(found_words, filename_suffix="正規化後", thresholds=max_line)
    print("")
    print("")



    # ==========================================================
    # 7. 統合された分析パイプラインの実行
    # ==========================================================
    trainer = StrictSVMTrainer(save_dir=SAVE_DIR)

    # 誤判定データ及びマージン内のデータを出力。要因となる特徴の重みと値を出力する
    df_errors = trainer.perform_error_analysis(
        X_normalized,
        y_final,
        found_words,
        document_names=document_names_final,
    )
    print(df_errors)

    # エラー分析で使用する線形SVMの設定
    clf_final_model = svm.SVC(kernel='linear', C=1000, class_weight='balanced')
    clf_final_model.fit(X_normalized, y_final)

    # 各特徴の重みの分布を確認
    trainer.analyze_global_weights(clf_final_model, found_words, top_n=36, save_png=True)

    #　ピンポイントのデータの分析
    # すべての特徴の重みとあるデータの値、scoreからなぜを分析する
    trainer.inspect_specific_data(clf_final_model, X_normalized, y_final, found_words, target_index=5)








"""
    # 7. 学習実行 (正規化済みのデータを使用)
    trainer = StrictSVMTrainer()

    # (A) 線形SVM
    trainer.run_custom_cv(X_normalized, y_final, kernel='linear', C_list=[1000])

    # (B) RBFカーネルSVM
    C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    G_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    trainer.run_custom_cv(X_normalized, y_final, kernel='rbf', C_list=C_range, Gamma_list=G_range)

    # 8. 比較用 GridSearch
    param_grid_std = {'C': C_range, 'gamma': G_range}
    df_result = trainer.run_standard_grid_search(X_normalized, y_final, param_grid_std)

    # 予測結果の一部を表示
    print("\n--- Standard GridSearch Predictions (Head) ---")
    print(df_result.head())
"""
