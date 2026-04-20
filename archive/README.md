過去の研究コードを保管するフォルダです。

この配下は現在の主流コードではなく、研究の履歴を残すための領域です。
現時点で主に使うコードは `active/` にあります。

- `feature_history/bias_ranking/`
  - 単語の偏り確認や差分可視化
- `feature_history/term_selection/`
  - 偏った単語を特徴量として選ぶ処理
- `feature_history/unique_vocab/`
  - クラス固有語の抽出
- `feature_history/frequency_phrase_analysis/`
  - 頻度、フレーズ、閾値分析の試行版
- `feature_history/preprocessing_checks/`
  - stopword などの前処理比較
- `feature_history/term_analysis_inactive_20260420/`
  - しばらく使わない term analysis 系の補助スクリプト退避先
- `experiment_history/legacy_pipelines/`
  - 旧パイプライン実験
- `experiment_history/reusable_eval/`
  - 実験で再利用していた評価補助
- `experiment_history/undergraduate/`
  - 学部研究段階のコード
- `experiment_history/masters/`
  - 修士段階のコード
- `experiment_history/broad_trials/`
  - 幅広い分類器を混ぜて試した大型実験
- `experiment_history/term_analysis_inactive_20260420/`
  - しばらく使わない term analysis 系の実験コード退避先

現在の主利用コードは `active/term_analysis/doc_count_topn_margin_nested_cv_experiment.py` と `active/term_analysis/doc_count_score_threshold_margin_nested_cv_experiment.py` です。
再利用しやすい共通部品は `active/reusable_core/` に置いています。

この配下のスクリプトも、`project_runtime.py` により import 経路と出力先が壊れにくいようにしています。
ただし、古い実験の中には処理時間が非常に長いものがあります。
