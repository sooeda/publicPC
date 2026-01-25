import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr 

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
    #"tropical":       {"center_method": "tropical", "kwargs": {"tropical_mode": "crisp"}},
    "tropical_best":  {"center_method": "tropical", "kwargs": {"tropical_mode": "best"}},
    "tropical_worst": {"center_method": "tropical", "kwargs": {"tropical_mode": "worst"}},
    "aof":      {"center_method": "aof",      "kwargs": {}},
    "md":       {"center_method": "md",       "kwargs": {}},
    "sg":       {"center_method": "sg",       "kwargs": {}},
    "gof":      {"center_method": "gof",      "kwargs": {}},
}

#шаблон названия папки
FOLDER_RE = re.compile(r"^(?P<n>\d+)_experts_(?P<q>low|medium|high)_contrast$")



def get_ranks(weights: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(weights)[::-1]
    ranks = np.zeros_like(sorted_idx, dtype=float)
    for r, idx in enumerate(sorted_idx, 1):
        ranks[idx] = r
    return ranks

#поиск всех файлов
def iter_json_files(data_dir: Path) -> List[Tuple[int, str, Path]]:
    items: List[Tuple[int, str, Path]] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")
    for sub in data_dir.iterdir():
        if not sub.is_dir():
            continue
        m = FOLDER_RE.match(sub.name)
        if not m:
            continue
        n_experts = int(m.group("n"))
        quality = m.group("q")
        for fp in sorted(sub.glob("*.json")):
            items.append((n_experts, quality, fp))
    return items

# анализ 1 файла
def run_one_file(json_path: Path, method_label: str, tt: float, ss: float) -> Dict[str, Any]:
    cfg = METHODS_CONFIG[method_label]
    gsd = Pairwise_comparison(
        debug_mode=False,
        center_method=cfg["center_method"],
        **cfg["kwargs"],
    )
    gsd.load_from_json(str(json_path))
    res = gsd.run_complete_analysis(t=tt, s=ss)

    if getattr(gsd, "center_matrix", None) is not None:
        res["center_matrix"] = np.asarray(gsd.center_matrix, dtype=float).tolist()
    if getattr(gsd, "dm_ids", None) is not None:
        res["dm_ids"] = list(gsd.dm_ids)
    if getattr(gsd, "alpha_weights", None) is not None:
        res["alpha_weights"] = [float(x) for x in gsd.alpha_weights]
    if getattr(gsd, "criteria_names", None) is not None:
        res["criteria_names"] = list(gsd.criteria_names)

    return res

# Построение матрицы корреляции
def corr_matrix_for_file(per_method: Dict[str, Dict[str, Any]], method_order: List[str]) -> np.ndarray:
    ranks = {}
    for m in method_order:
        w = np.asarray(per_method[m]["weights"], dtype=float)
        ranks[m] = get_ranks(w)

    K = len(method_order)
    C = np.full((K, K), np.nan, dtype=float)
    for i, mi in enumerate(method_order):
        for j, mj in enumerate(method_order):
            C[i, j] = float(spearmanr(ranks[mi], ranks[mj])[0])
    return C


def main(data_dir: Path = DATA_DIR, tt: float = TT, ss: float = SS) -> None:
    files = iter_json_files(data_dir)
    if not files:
        raise RuntimeError(f"No scenario folders found in {data_dir.resolve()}")

    method_order = list(METHODS_CONFIG.keys())

    rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    all_matrices: List[Dict[str, Any]] = []

    # Анализ всех файлов
    for n_experts, quality, fp in files:
        per_method: Dict[str, Dict[str, Any]] = {}
        all_ok = True

        # Анализ всех методов
        for method_label in method_order:
            try:
                per_method[method_label] = run_one_file(fp, method_label=method_label, tt=tt, ss=ss)
            except Exception as e:
                all_ok = False
                rows.append({
                    "file": str(fp),
                    "folder": fp.parent.name,
                    "n_experts": n_experts,
                    "quality": quality,
                    "method": method_label,
                    "status": "error",
                    "error": str(e),
                })
                continue
        for method_label, res in per_method.items():
            all_matrices.append({
                "method": method_label,
                "n_experts": int(n_experts),
                "quality": quality,
                "center_matrix": res.get("center_matrix"),
            })

        mof_ok = ("mof" in per_method) and ("weights" in per_method["mof"])
        if mof_ok:
            mof_ranks = get_ranks(np.array(per_method["mof"]["weights"], dtype=float))

        #Сохранение результатов
        for method_label, res in per_method.items():
            w = np.array(res["weights"], dtype=float)
            rank_corr_vs_mof = np.nan
            if mof_ok:
                rank_corr_vs_mof = float(spearmanr(mof_ranks, get_ranks(w))[0])

            rows.append({
                "file": str(fp),
                "folder": fp.parent.name,
                "n_experts": n_experts,
                "quality": quality,
                "method": method_label,
                "status": "ok",
                "lambda_opt": float(res["lambda_opt"]),
                "U": float(res["U"]),
                "GSI": float(res["GSI"]),
                "rank_corr_vs_mof": rank_corr_vs_mof,
            })

        if all_ok and len(per_method) == len(method_order):
            C = corr_matrix_for_file(per_method, method_order)
            for i, mi in enumerate(method_order):
                for j, mj in enumerate(method_order):
                    pair_rows.append({
                        "file": str(fp),
                        "folder": fp.parent.name,
                        "n_experts": n_experts,
                        "quality": quality,
                        "method_i": mi,
                        "method_j": mj,
                        "spearman": float(C[i, j]),
                    })

    raw_df = pd.DataFrame(rows)
    raw_path = OUT_DIR / "raw_results.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"Saved raw results: {raw_path}")

    ok_df = raw_df[raw_df["status"] == "ok"].copy()

    summary = (
        ok_df
        .groupby(["n_experts", "quality", "method"], as_index=False)
        .agg(
            files_count=("file", "nunique"),
            lambda_opt_mean=("lambda_opt", "mean"),
            lambda_opt_std=("lambda_opt", "std"),
            U_mean=("U", "mean"),
            U_std=("U", "std"),
            GSI_mean=("GSI", "mean"),
            GSI_std=("GSI", "std"),
            rank_corr_mean=("rank_corr_vs_mof", "mean"),
            rank_corr_std=("rank_corr_vs_mof", "std"),
        )
        .sort_values(["quality", "n_experts", "method"])
    )

    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    matrices_path = OUT_DIR / "all_aggregated_matrices.json"
    with open(matrices_path, "w", encoding="utf-8") as f:
        json.dump(all_matrices, f, ensure_ascii=False, indent=2)
    print(f"Saved matrices JSON: {matrices_path}")

    pair_df = pd.DataFrame(pair_rows)
    pair_raw_path = OUT_DIR / "pairwise_method_corr_raw.csv"
    pair_df.to_csv(pair_raw_path, index=False, encoding="utf-8")
    print(f"Saved pairwise corr raw: {pair_raw_path}")

# Построение и сохранение тепловых карт
    if not pair_df.empty:
        heatmap_dir = OUT_DIR / "corr_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="ticks")
        for (nexp, q), g in pair_df.groupby(["n_experts", "quality"]):
            g2 = (
                g.dropna(subset=["spearman"])
                 .groupby(["method_i", "method_j"], as_index=False)
                 .agg(spearman_mean=("spearman", "mean"))
            )

            mat = (
                g2.pivot(index="method_i", columns="method_j", values="spearman_mean")
                  .reindex(index=method_order, columns=method_order)
            )
            mat_csv = heatmap_dir / f"corr_heatmap_n{nexp}_{q}.csv"
            mat.to_csv(mat_csv, encoding="utf-8")

            plt.figure(figsize=(0.7 * len(method_order) + 4, 0.7 * len(method_order) + 3))
            ax = sns.heatmap(
                mat,
                vmin=-1, vmax=1, center=0,
                cmap="RdBu_r",
                annot=True, fmt=".2f",
                square=True,
            )
            ax.set_title(f"coefficient Spearman | n={nexp} | {q}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            plt.tight_layout()

            img_path = heatmap_dir / f"corr_heatmap_n{nexp}_{q}.png"
            plt.savefig(img_path, dpi=200)
            plt.close()

        print(f"Saved heatmaps to: {heatmap_dir}")

    if not summary.empty:
        df_plot = summary.copy()
        df_plot["quality"] = pd.Categorical(df_plot["quality"], categories=["high", "medium", "low"], ordered=True)

        g = sns.relplot(
            data=df_plot,
            x="n_experts",
            y="rank_corr_mean",
            hue="method",
            style="method",
            markers=True,
            dashes=True,
            col="quality",
            kind="line",
            facet_kws=dict(sharey=True, sharex=True),
            height=4,
            aspect=1.15,
            alpha=0.9,
        )

        g.set_axis_labels("Число экспертов", "Spearman(ранги) с MOF")
        g.set_titles("{col_name} contrast")

        for ax in g.axes.flat:
            for line in ax.lines:
                line.set_linewidth(2.0)

        plt.tight_layout()
        plt.savefig(OUT_DIR / "rank_corr_vs_mof_by_contrast.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ корреляций методов")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".") / "data",
        help="Папка с данными (по умолчанию ./data)",
    )
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"Data directory is not a directory: {args.data_dir}")
    
    DATA_DIR = args.data_dir.resolve()
    OUT_DIR = DATA_DIR.parent / "out_results" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    main(DATA_DIR)

