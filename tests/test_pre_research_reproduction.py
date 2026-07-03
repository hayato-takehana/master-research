import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_runtime import bootstrap_project_paths


bootstrap_project_paths(PROJECT_ROOT)

from pdf_text_loader import PREPROCESSING_VERSION, Text_road_and_dell, normalize_text
from active.term_analysis.pre_research_reproduction import PriorResearchTfidfTransformer


class PdfTextLoaderTests(unittest.TestCase):
    def test_normalize_text_replaces_hyphens_with_spaces(self):
        self.assertEqual(
            normalize_text("state-of-the-art re-\nentrancy"),
            "state of the art re entrancy",
        )

    def test_cache_metadata_includes_preprocessing_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "sample.txt"
            source_path.write_text("state-of-the-art", encoding="utf-8")
            cache_path = temp_path / "documents.pkl"
            with cache_path.open("wb") as cache_file:
                pickle.dump(["stale"], cache_file)

            loader = Text_road_and_dell(str(cache_path), str(temp_path))
            files = loader._source_files()
            metadata_path = Path(loader._metadata_path())
            self.assertFalse(loader._cache_is_fresh(files))

            metadata_path.write_text(
                json.dumps(loader._source_metadata(files)),
                encoding="utf-8",
            )
            self.assertFalse(loader._cache_is_fresh(files))

            metadata_path.write_text(
                json.dumps(loader._cache_metadata(files)),
                encoding="utf-8",
            )
            self.assertTrue(loader._cache_is_fresh(files))
            self.assertEqual(
                loader._cache_metadata(files)["preprocessing_version"],
                PREPROCESSING_VERSION,
            )


class PriorResearchTfidfTransformerTests(unittest.TestCase):
    def test_test_only_term_is_not_added_to_training_vocabulary(self):
        transformer = PriorResearchTfidfTransformer(
            min_len=3,
            min_df=1,
            use_stemming=False,
        )
        transformer.fit(["shared training", "shared feature"])

        transformed = transformer.transform(["shared testonly"])

        self.assertNotIn("testonly", transformer.feature_names_)
        self.assertEqual(transformed.shape[1], len(transformer.feature_names_))
        self.assertGreater(transformed.nnz, 0)


if __name__ == "__main__":
    unittest.main()
