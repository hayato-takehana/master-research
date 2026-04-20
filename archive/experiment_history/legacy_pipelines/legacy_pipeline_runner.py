from pathlib import Path
import os
import sys


def _bootstrap_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "active").exists() and (candidate / "archive").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Project root could not be found.")


PROJECT_ROOT = _bootstrap_project_root()

from project_runtime import bootstrap_project_paths, get_output_dir, redirect_relative_outputs

bootstrap_project_paths(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
redirect_relative_outputs(get_output_dir(__file__, PROJECT_ROOT))

from real_scam_pipeline import real_scam
from prior_study_pipeline import pre_research

#実際の詐欺
#抽出方法の特徴の数、number, CSVファイルにするか、主成分分析による可視化をするか,SVMをするか
#number = 1:バイナリー変数 2: 出現回数 3:列での正規化 4:行での正規化
#real_scam(0, 10, 4, True, True, False)

#先行研究
#抽出方法の特徴の数、number, CSVファイルにするか、主成分分析による可視化をするか,SVMをするか
#number = 1:バイナリー変数 2:出現回数 3:列での正規化 4:行での正規化
m=200
pre_research(m, m, 2, False, False, False, True)

"""
kaisuu = [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for k in kaisuu:

    pre_research(k, k,2, False,False, False, True)
"""
