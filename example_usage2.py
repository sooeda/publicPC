import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from PairwiseComparison import Pairwise_comparison


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TT = 5
SS = 0.3

METHODS_CONFIG = {
    "mof":      {"center_method": "mof",      "kwargs": {}},
    "rowsum":   {"center_method": "rowsum",   "kwargs": {}},
    #"tropical": {"center_method": "tropical", "kwargs": {}},
    "tropical_best":  {"center_method": "tropical", "kwargs": {"tropical_mode": "best"}},
    "tropical_worst": {"center_method": "tropical", "kwargs": {"tropical_mode": "worst"}},
    "aof":      {"center_method": "aof",      "kwargs": {}},
    "md":       {"center_method": "md",       "kwargs": {}},
    "sg":       {"center_method": "sg",       "kwargs": {}},
    "gof":      {"center_method": "gof",      "kwargs": {}},
}


def weights_to_ranks_desc(weights: np.ndarray) -> np.ndarray:

    sorted_idx = np.argsort(weights)[::-1]
    ranks = np.zeros_like(sorted_idx, dtype=int)
    for r, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = r
    return ranks


def run_methods_for_single_json(
    input_json_path: Path,
    output_json_path: Path,
    tt: float = TT,
    ss: float = SS,
) -> None:
    input_json_path = input_json_path.resolve()
    output_json_path = output_json_path.resolve()

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    results: List[Dict[str, Any]] = []

    for method_label, cfg in METHODS_CONFIG.items():
        gsd = Pairwise_comparison(
            debug_mode=False,
            center_method=cfg["center_method"],
            **cfg["kwargs"],
        )
        gsd.load_from_json(str(input_json_path))
        res = gsd.run_complete_analysis(t=tt, s=ss)

        center = getattr(gsd, "center_matrix", None)
        if center is None:
            raise RuntimeError(f"center_matrix is None for method={method_label}")

        weights = np.asarray(res["weights"], dtype=float)
        ranks = weights_to_ranks_desc(weights)
        results.append({
            "method": method_label,
            "center_matrix": np.asarray(center, dtype=float).tolist(),
            "weights": weights.tolist(),
            "ranks": ranks.tolist(),
        })

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    INPUT_NAME = "article_example1.json"

    input_fp = DATA_DIR / INPUT_NAME
    out_fp = OUT_DIR / f"{Path(INPUT_NAME).stem}_methods_center_weights_ranks.json"

    run_methods_for_single_json(input_fp, out_fp)
    print(f"Input : {input_fp.resolve()}")
    print(f"Output: {out_fp.resolve()}")
