# pythonProject4

詐欺的なホワイトペーパーと非詐欺ホワイトペーパーを対象に、PDF からテキストを抽出し、特徴量を作成し、単語分析や分類実験を行う研究用プロジェクトです。

現在の主利用コードは [technical_term_analysis.py](c:/Users/taken/PycharmProjects/pythonProject4/active/term_analysis/technical_term_analysis.py) と [scam_term_scout.py](c:/Users/taken/PycharmProjects/pythonProject4/active/term_analysis/scam_term_scout.py) です。  
過去に量産した特徴抽出・実験コードは `archive/` に整理して保管しています。

## 現在の使い方

現役の分析を実行する場合は、仮想環境の Python を使って次を実行します。

```powershell
.\.venv\Scripts\python.exe active\term_analysis\technical_term_analysis.py
.\.venv\Scripts\python.exe active\term_analysis\scam_term_scout.py
```

PyCharm を使う場合も interpreter は [\.venv](c:/Users/taken/PycharmProjects/pythonProject4/.venv) を指定してください。

## フォルダ構成

```text
pythonProject4/
  active/
    reusable_core/
    term_analysis/
  archive/
    feature_history/
    experiment_history/
  data/
    outputs/
  詐欺_先行研究/
  詐欺じゃない_先行研究/
  詐欺_実際の詐欺/
  詐欺じゃない_実際の詐欺/
  document_*.pkl
  project_runtime.py
```

### `active/`

現在も使うコードです。

- [technical_term_analysis.py](c:/Users/taken/PycharmProjects/pythonProject4/active/term_analysis/technical_term_analysis.py)
  - 現在のメイン分析スクリプト
  - 技術用語の抽出、可視化、正規化、SVM 分析、誤判定分析を実行
- [scam_term_scout.py](c:/Users/taken/PycharmProjects/pythonProject4/active/term_analysis/scam_term_scout.py)
  - ラベル1側とラベル0側の両方で多い単語候補を探索するスクリプト
  - 各 n-gram ごとに TF-IDF、出現回数、出現文書数、平均回数、出現率差を集計
- [dataset_loader.py](c:/Users/taken/PycharmProjects/pythonProject4/active/reusable_core/dataset_loader.py)
  - データセット読込の共通入口
  - PDF キャッシュ読込、ラベル作成、TF-IDF / 出現回数行列の生成
- [pdf_text_loader.py](c:/Users/taken/PycharmProjects/pythonProject4/active/reusable_core/pdf_text_loader.py)
  - PDF からテキストを抽出して前処理し、`pkl` にキャッシュ
- [text_vectorizer.py](c:/Users/taken/PycharmProjects/pythonProject4/active/reusable_core/text_vectorizer.py)
  - トークン化、stopword 除去、TF-IDF / 出現回数ベクトル化

### `archive/`

過去の研究コードです。現在は履歴保管が目的です。

- `feature_history/`
  - 単語偏り確認、特徴選択、固有語抽出、頻度・フレーズ分析など
- `experiment_history/`
  - 旧実験パイプライン、学部研究段階、修士研究段階、大型試行コードなど

詳細は [archive/README.md](c:/Users/taken/PycharmProjects/pythonProject4/archive/README.md) を参照してください。

### `data/`

出力データの保存先です。

- `data/outputs/<コードの相対パス>/`
  - 各スクリプトの `CSV` や `PNG` を自動で整理する場所
- 例
  - [technical_term_analysis の出力先](c:/Users/taken/PycharmProjects/pythonProject4/data/outputs/active/term_analysis/technical_term_analysis)
  - [scam_term_scout の出力先](c:/Users/taken/PycharmProjects/pythonProject4/data/outputs/active/term_analysis/scam_term_scout)

詳細は [data/README.md](c:/Users/taken/PycharmProjects/pythonProject4/data/README.md) を参照してください。

## データ

入力データはプロジェクト直下の次のフォルダを使います。

- [詐欺_先行研究](c:/Users/taken/PycharmProjects/pythonProject4/詐欺_先行研究)
- [詐欺じゃない_先行研究](c:/Users/taken/PycharmProjects/pythonProject4/詐欺じゃない_先行研究)
- [詐欺_実際の詐欺](c:/Users/taken/PycharmProjects/pythonProject4/詐欺_実際の詐欺)
- [詐欺じゃない_実際の詐欺](c:/Users/taken/PycharmProjects/pythonProject4/詐欺じゃない_実際の詐欺)

PDF から抽出したテキストは、プロジェクト直下の `document_*.pkl` にキャッシュされます。

## 実行時の設計

- [project_runtime.py](c:/Users/taken/PycharmProjects/pythonProject4/project_runtime.py)
  - プロジェクトルートの検出
  - `sys.path` の補助設定
  - `data/outputs/...` への出力先整理

この仕組みにより、古いスクリプトを深いフォルダに移しても、相対 import と相対出力が極力壊れないようにしています。

## 現状の確認結果

確認済みの内容は次です。

- 全 `.py` ファイルの構文チェックは通過
- import 解決は通過
- 主要モジュールの読み込み確認は通過
- 一部の旧実験スクリプトは実行時間が非常に長く、短時間では完走しない

これは「壊れていてすぐ落ちる」ものと「計算が重くて終わらない」ものを分けて確認した結果です。  
現在、明確な実行エラーとして見つかったものは修正済みです。

## 注意点

- 実行は必ず [\.venv](c:/Users/taken/PycharmProjects/pythonProject4/.venv) を使ってください
- 古い実験コードの一部は `plt.show()` を含みます
- 古い実験コードの多くは研究当時のまま残しており、保守性より履歴保管を優先しています
- `archive/` のコードは「再実験したいときに参照できる状態」を目標にしており、現役コードのような整理度ではありません

## 読む順番

初めて見る場合は次の順番が分かりやすいです。

1. [technical_term_analysis.py](c:/Users/taken/PycharmProjects/pythonProject4/active/term_analysis/technical_term_analysis.py)
2. [dataset_loader.py](c:/Users/taken/PycharmProjects/pythonProject4/active/reusable_core/dataset_loader.py)
3. [pdf_text_loader.py](c:/Users/taken/PycharmProjects/pythonProject4/active/reusable_core/pdf_text_loader.py)
4. [text_vectorizer.py](c:/Users/taken/PycharmProjects/pythonProject4/active/reusable_core/text_vectorizer.py)
5. [scam_term_scout.py](c:/Users/taken/PycharmProjects/pythonProject4/active/term_analysis/scam_term_scout.py)
6. 必要に応じて [archive/README.md](c:/Users/taken/PycharmProjects/pythonProject4/archive/README.md)

## メモ

このプロジェクトでは、ファイル名と出力フォルダ名から役割が分かることを重視して整理しています。  
今後新しい分析コードを追加する場合も、`active/` または `archive/` の適切な場所に置き、出力は `data/outputs/<そのコードのパス>/` に揃える方針です。
