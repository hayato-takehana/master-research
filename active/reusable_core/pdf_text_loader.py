import re
import os
import pickle
import fitz  # PyMuPDF


def normalize_text(text, keep_digits=False, percent_mode="drop"):
    # 1. 行末のハイフンと改行をセットで消去（単語の分断を最優先で結合）
    # 例: "re-\n entrancy" -> "reentrancy"
    text = re.sub(r'-\s*[\r\n]\s*', '', text)

    # 2. 文中に残っているすべてのハイフンを空文字に置換
    # 例: "re-entrancy" -> "reentrancy", "big-endian" -> "bigendian"
    text = text.replace('-', '')

    # 3. パーセント表記の扱いを切り替える
    if percent_mode == "word":
        text = text.replace('%', ' percent ')
    elif percent_mode != "drop":
        raise ValueError(f"Unsupported percent_mode: {percent_mode}")

    # 4. 残った改行コードをスペースに置換
    text = re.sub(r'[\r\n]+', ' ', text)

    # 5. 英数字の保持有無に応じて許可文字を切り替える
    if keep_digits:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 6. 連続する空白を1つのスペースに集約し、前後の空白を削除
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

    def read_PDF(self):
        if os.path.exists(self.cash_file):
            with open(self.cash_file, "rb") as f:
                documents = pickle.load(f)
            print("キャッシュファイルから読み込みました。")
        else:
            documents = []

            # フォルダ内のファイルをリストアップ
            files = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]

            for i, filename in enumerate(files, 1):
                print(f"{i}番目のデータを読み込みます: {filename}")
                file_path = os.path.join(self.folder_path, filename)

                try:
                    # PyMuPDF (fitz) を使用した読み込み
                    doc = fitz.open(file_path)
                    extracted_text = ""

                    for page in doc:
                        # get_text("text") でテキストを抽出
                        extracted_text += page.get_text("text") + " "

                    doc.close()

                    # クリーニング処理
                    cleaned_text = self._text_dell(extracted_text)
                    documents.append(cleaned_text)

                except Exception as e:
                    print(f"エラーが発生しました ({filename}): {e}")

            # キャッシュ保存
            with open(self.cash_file, "wb") as f:
                pickle.dump(documents, f)
            print("新たに保存しました。")

        return documents
