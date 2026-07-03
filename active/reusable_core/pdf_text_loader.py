import re
import os
import pickle
import json


PREPROCESSING_VERSION = 2


def normalize_text(text, keep_digits=False, percent_mode="drop"):
    # 1. 単語をつなぐハイフンを空白へ置換する。
    # 例: "state-of-the-art" -> "state of the art"
    text = re.sub(r"[-‐‑‒–—―]+", " ", text)

    # 2. パーセント表記の扱いを切り替える
    if percent_mode == "word":
        text = text.replace('%', ' percent ')
    elif percent_mode != "drop":
        raise ValueError(f"Unsupported percent_mode: {percent_mode}")

    # 3. 残った改行コードをスペースに置換
    text = re.sub(r'[\r\n]+', ' ', text)

    # 4. 英数字の保持有無に応じて許可文字を切り替える
    if keep_digits:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. 連続する空白を1つのスペースに集約し、前後の空白を削除
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class Text_road_and_dell:

    def __init__(self, cash_file, folder_path, keep_digits=False, percent_mode="drop"):
        self.cash_file = cash_file
        self.folder_path = folder_path
        self.keep_digits = keep_digits
        self.percent_mode = percent_mode

    def _text_dell(self, text):
        return normalize_text(
            text,
            keep_digits=self.keep_digits,
            percent_mode=self.percent_mode,
        )

    def _source_files(self):
        files = []
        for filename in os.listdir(self.folder_path):
            lower_name = filename.lower()
            if lower_name.endswith(".pdf") or lower_name.endswith(".txt"):
                files.append(filename)
        return files

    def _source_metadata(self, files):
        metadata = []
        for filename in files:
            file_path = os.path.join(self.folder_path, filename)
            stat = os.stat(file_path)
            metadata.append(
                {
                    "name": filename,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
        return metadata

    def _cache_metadata(self, files):
        return {
            "preprocessing_version": PREPROCESSING_VERSION,
            "sources": self._source_metadata(files),
        }

    def _metadata_path(self):
        return f"{self.cash_file}.sources.json"

    def _cache_is_fresh(self, files):
        if not os.path.exists(self.cash_file):
            return False

        metadata_path = self._metadata_path()
        if not os.path.exists(metadata_path):
            # 前処理バージョンを確認できない古いキャッシュは再作成する。
            return False

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                cached_metadata = json.load(f)
        except Exception:
            return False

        return cached_metadata == self._cache_metadata(files)

    def _read_pdf_text(self, file_path):
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text("text") + " "
        doc.close()
        return extracted_text

    def _read_txt_text(self, file_path):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return f.read()

    def read_PDF(self):
        files = self._source_files()

        if self._cache_is_fresh(files):
            with open(self.cash_file, "rb") as f:
                documents = pickle.load(f)
            print("キャッシュファイルから読み込みました。")
        else:
            documents = []

            for i, filename in enumerate(files, 1):
                print(f"{i}番目のデータを読み込みます: {filename}")
                file_path = os.path.join(self.folder_path, filename)

                try:
                    if filename.lower().endswith(".pdf"):
                        extracted_text = self._read_pdf_text(file_path)
                    else:
                        extracted_text = self._read_txt_text(file_path)

                    # クリーニング処理
                    cleaned_text = self._text_dell(extracted_text)
                    documents.append(cleaned_text)

                except Exception as e:
                    print(f"エラーが発生しました ({filename}): {e}")

            # キャッシュ保存
            with open(self.cash_file, "wb") as f:
                pickle.dump(documents, f)
            with open(self._metadata_path(), "w", encoding="utf-8") as f:
                json.dump(self._cache_metadata(files), f, ensure_ascii=False, indent=2)
            print("新たに保存しました。")

        return documents
