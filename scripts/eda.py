"""Global EDA on the processed labeled set -> results/eda.json, results/tables/eda.md, sample figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from wm811k_audit.constants import CLASS_NAMES, PROCESSED_DIR, RESULTS_DIR
from wm811k_audit.data import load_processed
from wm811k_audit.diagnostics import global_eda

WAFER_CMAP = ListedColormap(["#f0efec", "#9ec5f4", "#0d366b"])  # no die / pass / fail


def write_markdown(e: dict, path: Path):
    lines = ["# WM-811K labeled set — EDA", "", f"- labeled wafers: {e['n_labeled']:,}",
             f"- lots: {e['lots']['n_lots']:,} (singleton lot ids: {e['lots']['n_singleton']})",
             f"- labeled wafers per lot, median: {e['lots']['labeled_per_lot_quantiles']['50']:.0f}", "",
             "## Class counts", "", "| class | count | share |", "|---|---:|---:|"]
    for c in CLASS_NAMES:
        n = e["class_counts"][c]
        lines.append(f"| {c} | {n:,} | {100 * n / e['n_labeled']:.2f}% |")
    o = e["orig_split"]
    lines += ["", "## Original Training/Test labels", "",
              f"- counts: {o['counts']}", f"- lots containing both Training and Test wafers: **{o['lots_with_both']}**",
              f"- Test wafers whose lot also has Training wafers: {o['test_wafers_with_lot_in_training']:.3f}",
              f"- lot-number range: {o['lot_num_range']}", f"- Training/Test runs along lot order: {o['runs_along_lot_order']}",
              "", "| split | " + " | ".join(CLASS_NAMES) + " |", "|---|" + "---:|" * len(CLASS_NAMES)]
    for s, row in o["crosstab"].items():
        lines.append(f"| {s} | " + " | ".join(f"{row[c]:,}" for c in CLASS_NAMES) + " |")
    L = e["lots"]
    lines += ["", "## Within-lot structure", "",
              f"- lots with >=2 defect wafers: {L['lots_with_2plus_defects']:,}; single-class among them: {L['frac_single_class_among_them']:.3f}",
              f"- defect wafers living in such lots: {L['defect_wafers_in_such_lots']:,} / {L['n_defect_wafers']:,}"]
    for key, title in (("duplicates", "Exact duplicates (raw maps)"), ("duplicates64", "Exact duplicates (after 64x64 resize)")):
        d = e[key]
        lines += ["", f"## {title}", "", f"- rows in duplicate groups: {d['n_rows_in_groups']:,} ({100 * d['frac_rows']:.2f}%), groups: {d['n_groups']:,}",
                  f"- groups spanning >1 lot: {d['frac_groups_multi_lot']:.3f}; groups spanning Training&Test: {d['groups_spanning_orig_split']:,}",
                  f"- rows by class: {d['rows_by_class']}"]
    S = e["shapes"]
    lines += ["", "## Map shapes", "", f"- unique shapes: {S['n_unique']}; H quantiles {S['h_quantiles']}; W quantiles {S['w_quantiles']}",
              f"- maps with H>64 or W>64 (downsampled): {100 * S['frac_over_64']:.2f}%",
              "- top shapes: " + ", ".join(f"{t['h']}x{t['w']} ({t['count']:,})" for t in S["top"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_samples(maps64, meta, path: Path, per_class: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(per_class, 9, figsize=(13.5, 4.8), facecolor="#fcfcfb")
    for j, c in enumerate(CLASS_NAMES):
        rows = meta.index[meta["failure_type"] == c].values
        pick = rng.choice(rows, size=min(per_class, len(rows)), replace=False)
        for i in range(per_class):
            ax = axes[i, j]
            ax.set_axis_off()
            if i < len(pick):
                ax.imshow(maps64[pick[i]], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
            if i == 0:
                ax.set_title(c, fontsize=9, color="#0b0b0b")
    fig.suptitle("WM-811K labeled wafer maps after 64x64 nearest-neighbour resize (gray: no die, light: pass, dark: fail)",
                 fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    maps64, meta = load_processed(PROCESSED_DIR)
    e = global_eda(meta)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    with open(RESULTS_DIR / "eda.json", "w", encoding="utf-8") as f:
        json.dump(e, f, indent=2, ensure_ascii=False)
    write_markdown(e, RESULTS_DIR / "tables" / "eda.md")
    plot_samples(maps64, meta, RESULTS_DIR / "figures" / "samples_per_class.png")
    print(json.dumps({k: e[k] for k in ("n_labeled", "orig_split", "lots", "duplicates")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
