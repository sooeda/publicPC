import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  

from PairwiseComparison import Pairwise_comparison

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TT = 5
SS = 0.3

METHODS_CONFIG = {
    "mof":           {"center_method": "mof",      "kwargs": {}},
    "rowsum":        {"center_method": "rowsum",   "kwargs": {}},
    "tropical_best": {"center_method": "tropical", "kwargs": {"tropical_mode": "best"}},
    "tropical_worst":{"center_method": "tropical", "kwargs": {"tropical_mode": "worst"}},
    "aof":           {"center_method": "aof",      "kwargs": {}},
    "md":            {"center_method": "md",       "kwargs": {}},
    "sg":            {"center_method": "sg",       "kwargs": {}},
    "gof":           {"center_method": "gof",      "kwargs": {}},
}


def ranks_desc(weights: np.ndarray) -> np.ndarray:
    """1 = лучший (максимальный вес)."""
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(weights)[::-1]
    r = np.empty_like(order, dtype=int)
    for rank, idx in enumerate(order, start=1):
        r[idx] = rank
    return r


def run_one_method(json_path: Path, method_label: str, tt: float, ss: float) -> Dict[str, Any]:
    cfg = METHODS_CONFIG[method_label]
    gsd = Pairwise_comparison(
        debug_mode=False,
        center_method=cfg["center_method"],
        **cfg["kwargs"],
    )
    gsd.load_from_json(str(json_path))
    res = gsd.run_complete_analysis(t=tt, s=ss)
    return res


def main_one_file_corr_vs_mof(
    input_json: Path,
    out_csv: Path,
    tt: float = TT,
    ss: float = SS,
) -> pd.DataFrame:
    input_json = input_json.resolve()
    out_csv = out_csv.resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    per_method: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for method_label in METHODS_CONFIG.keys():
        try:
            per_method[method_label] = run_one_method(input_json, method_label, tt=tt, ss=ss)
        except Exception as e:
            rows.append({
                "method": method_label,
                "status": "error",
                "spearman_vs_mof": np.nan,
                "error": repr(e),
            })

    if "mof" not in per_method or "weights" not in per_method["mof"]:
        raise RuntimeError("MOF did not produce weights; cannot compute correlations.")

    mof_ranks = ranks_desc(np.asarray(per_method["mof"]["weights"], dtype=float))

    for method_label, res in per_method.items():
        w = np.asarray(res["weights"], dtype=float)
        r = ranks_desc(w)
        rho = float(spearmanr(mof_ranks, r)[0])
        rows.append({
            "method": method_label,
            "status": "ok",
            "spearman_vs_mof": rho,
            "error": "",
        })

    df = pd.DataFrame(rows).sort_values(["status", "method"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    return df


if __name__ == "__main__":
    INPUT_NAME = "article_example2.json" 
    input_fp = DATA_DIR / INPUT_NAME

    out_fp = OUT_DIR / f"{Path(INPUT_NAME).stem}__spearman_with_mof.csv"

    df = main_one_file_corr_vs_mof(
        input_json=input_fp,
        out_csv=out_fp,
        tt=TT,
        ss=SS,
    )

    print("Input :", input_fp.resolve())
    print("Output:", out_fp.resolve())
    print(df)
