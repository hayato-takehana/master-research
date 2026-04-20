from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


IGNORED_DIR_NAMES = {
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    "data",
}

_OUTPUT_REDIRECT_INITIALIZED = False


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start).resolve() if start else Path(__file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            return candidate

    raise RuntimeError(f"Project root could not be found from: {current}")


def _iter_python_dirs(project_root: Path) -> Iterable[Path]:
    seen: set[Path] = set()

    for py_file in project_root.rglob("*.py"):
        parts = py_file.relative_to(project_root).parts
        if any(part in IGNORED_DIR_NAMES for part in parts):
            continue

        parent = py_file.parent
        if parent not in seen:
            seen.add(parent)
            yield parent


def bootstrap_project_paths(project_root: str | Path | None = None) -> Path:
    root = find_project_root(project_root)

    ordered_paths = [root, *sorted(_iter_python_dirs(root))]
    for path in ordered_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root


def get_output_dir(script_path: str | Path, project_root: str | Path | None = None) -> Path:
    root = find_project_root(project_root)
    script = Path(script_path).resolve()
    relative_no_suffix = script.relative_to(root).with_suffix("")
    output_dir = root / "data" / "outputs" / relative_no_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def redirect_relative_outputs(output_dir: str | Path, override: bool = False) -> None:
    global _OUTPUT_REDIRECT_INITIALIZED

    if _OUTPUT_REDIRECT_INITIALIZED and not override:
        return

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    original_to_csv = getattr(pd.DataFrame.to_csv, "__wrapped_original__", pd.DataFrame.to_csv)
    original_plt_savefig = getattr(plt.savefig, "__wrapped_original__", plt.savefig)
    original_fig_savefig = getattr(Figure.savefig, "__wrapped_original__", Figure.savefig)

    def _resolve_path(path_like):
        if isinstance(path_like, (str, Path)):
            path_obj = Path(path_like)
            if not path_obj.is_absolute():
                return output_dir / path_obj
        return path_like

    def patched_to_csv(self, path_or_buf=None, *args, **kwargs):
        return original_to_csv(self, _resolve_path(path_or_buf), *args, **kwargs)

    def patched_plt_savefig(fname, *args, **kwargs):
        return original_plt_savefig(_resolve_path(fname), *args, **kwargs)

    def patched_fig_savefig(self, fname, *args, **kwargs):
        return original_fig_savefig(self, _resolve_path(fname), *args, **kwargs)

    patched_to_csv.__wrapped_original__ = original_to_csv
    patched_plt_savefig.__wrapped_original__ = original_plt_savefig
    patched_fig_savefig.__wrapped_original__ = original_fig_savefig

    pd.DataFrame.to_csv = patched_to_csv
    plt.savefig = patched_plt_savefig
    Figure.savefig = patched_fig_savefig

    _OUTPUT_REDIRECT_INITIALIZED = True
