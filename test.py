from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pandas as pd
from rapidocr_onnxruntime import RapidOCR


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = find_project_root()

# Change this path if you want to test another image-based PDF.
PDF_PATH = PROJECT_ROOT / "詐欺_先行研究" / "162_1.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "ta" / "ocr_test"
RENDER_ZOOM = 2.0


def extract_embedded_text(pdf_path: Path) -> list[str]:
    document = fitz.open(pdf_path)
    return [page.get_text("text") for page in document]


def render_page_to_image(page: fitz.Page, output_path: Path) -> None:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM), alpha=False)
    pixmap.save(output_path)


def run_ocr(pdf_path: Path, output_dir: Path) -> tuple[list[dict], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    ocr = RapidOCR()
    document = fitz.open(pdf_path)
    rows: list[dict] = []
    page_texts: list[str] = []

    for page_index, page in enumerate(document, start=1):
        image_path = image_dir / f"{pdf_path.stem}_page{page_index:02d}.png"
        render_page_to_image(page, image_path)

        result, _ = ocr(str(image_path))
        page_lines: list[str] = []
        if result:
            for line_index, item in enumerate(result, start=1):
                _box, text, confidence = item
                text = str(text).strip()
                page_lines.append(text)
                rows.append(
                    {
                        "page": page_index,
                        "line": line_index,
                        "text": text,
                        "confidence": float(confidence),
                    }
                )
        page_texts.append("\n".join(page_lines))

    return rows, page_texts


def main() -> int:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        return 1

    embedded_page_texts = extract_embedded_text(PDF_PATH)
    embedded_text = "\n".join(embedded_page_texts)
    print(f"PDF: {PDF_PATH}")
    print(f"pages: {len(embedded_page_texts)}")
    print(f"embedded text characters: {len(embedded_text)}")

    rows, ocr_page_texts = run_ocr(PDF_PATH, OUTPUT_DIR)
    ocr_text = "\n\n".join(
        f"--- page {page_index} ---\n{page_text}"
        for page_index, page_text in enumerate(ocr_page_texts, start=1)
    )

    csv_path = OUTPUT_DIR / f"{PDF_PATH.stem}_ocr_lines.csv"
    text_path = OUTPUT_DIR / f"{PDF_PATH.stem}_ocr_text.txt"
    pd.DataFrame(rows, columns=["page", "line", "text", "confidence"]).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    text_path.write_text(ocr_text, encoding="utf-8")

    print(f"ocr lines: {len(rows)}")
    print(f"saved csv: {csv_path}")
    print(f"saved text: {text_path}")
    print("\n--- OCR sample ---")
    print("\n".join(ocr_text.splitlines()[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
