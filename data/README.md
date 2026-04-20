出力データの保存先です。

- 実行スクリプトが相対パスで `CSV` や `PNG` を保存した場合は、
  `data/outputs/<コードの相対パス>/`
  に自動で振り分けられます。
- 例:
  - `active/term_analysis/technical_term_analysis.py`
  - `data/outputs/active/term_analysis/technical_term_analysis/`

このルールにより、どのコードが出力した成果物かをフォルダ名から追えるようにしています。

現在の主な出力先は次です。

- [technical_term_analysis の出力](c:/Users/taken/PycharmProjects/pythonProject4/data/outputs/active/term_analysis/technical_term_analysis)
- [scam_term_scout の出力](c:/Users/taken/PycharmProjects/pythonProject4/data/outputs/active/term_analysis/scam_term_scout)

今後新しいスクリプトを追加した場合も、同じルールで `data/outputs/` 配下に整理されます。
