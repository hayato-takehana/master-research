import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from active.reusable_core import dataset_loader


class DatasetLoaderTests(unittest.TestCase):
    def test_default_dataset_names_remain_backward_compatible(self):
        self.assertEqual(
            dataset_loader._resolve_dataset_names(real_scam=False),
            dataset_loader.PRE_RESEARCH_DATASET_NAMES,
        )
        self.assertEqual(
            dataset_loader._resolve_dataset_names(real_scam=True),
            dataset_loader.REAL_SCAM_DATASET_NAMES,
        )

    def test_delete_v1_dataset_paths_are_selectable(self):
        scam_path, non_scam_path = dataset_loader._dataset_paths(
            dataset_names=dataset_loader.PRE_RESEARCH_DELETE_V1_DATASET_NAMES,
        )

        self.assertEqual(
            scam_path,
            dataset_loader.PROJECT_ROOT / "詐欺_先行研究_delete_v1",
        )
        self.assertEqual(
            non_scam_path,
            dataset_loader.PROJECT_ROOT / "詐欺じゃない_先行研究_delete_v1",
        )

    def test_load_documents_uses_dataset_specific_paths_and_caches(self):
        with patch.object(dataset_loader, "Text_road_and_dell") as loader_class:
            loader_class.return_value.read_PDF.side_effect = [
                ["scam document"],
                ["non-scam document"],
            ]

            documents = dataset_loader.load_documents(
                dataset_names=(
                    "詐欺_先行研究_delete_v1",
                    "詐欺じゃない_先行研究_delete_v1",
                ),
            )

        self.assertEqual(documents, (["scam document"], ["non-scam document"]))
        scam_call, non_scam_call = loader_class.call_args_list
        self.assertTrue(
            scam_call.args[0].endswith(
                "document_詐欺_先行研究_delete_v1.pkl"
            )
        )
        self.assertTrue(
            non_scam_call.args[0].endswith(
                "document_詐欺じゃない_先行研究_delete_v1.pkl"
            )
        )
        self.assertTrue(
            scam_call.args[1].endswith("詐欺_先行研究_delete_v1")
        )
        self.assertTrue(
            non_scam_call.args[1].endswith(
                "詐欺じゃない_先行研究_delete_v1"
            )
        )

    def test_dataset_names_requires_exactly_two_names(self):
        with self.assertRaises(ValueError):
            dataset_loader._resolve_dataset_names(dataset_names=("only_one",))


if __name__ == "__main__":
    unittest.main()
