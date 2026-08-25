"""Turn results.csv + per-run artifacts into the tables and figures the README uses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .constants import CLASS_NAMES, RESULTS_DIR

SUMMARY_METRICS = ["own_macro_f1", "own_defect_f1", "gold_defect_f1", "gold_full_macro_f1"]
AXES = {"split": ["A1", "A2", "A3"], "classes": ["B1", "B2"], "cap": ["C1", "C2", "C3"]}
AXIS_LABELS = {"split": {"A1": "original", "A2": "random", "A3": "lot-group"},
               "classes": {"B1": "9 classes", "B2": "8 defect classes"},
               "cap": {"C1": "no cap", "C2": "cap 5000/class", "C3": "balanced (min class)"}}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["model"] == "smallcnn"].copy() if "model" in df.columns else df


def cell_summary(df: pd.DataFrame, metrics=SUMMARY_METRICS) -> pd.DataFrame:
    g = df.groupby("cell_id")
    out = pd.DataFrame({"split": g["split"].first(), "classes": g["classes"].first(), "cap": g["cap"].first(),
                        "n_seeds": g["seed"].nunique()})
    for m in metrics:
        if m in df.columns:
            out[f"{m}_mean"] = g[m].mean()
            out[f"{m}_std"] = g[m].std(ddof=0)
    return out


def main_effects(summary: pd.DataFrame, metric: str) -> dict:
    col = f"{metric}_mean"
    s = summary.dropna(subset=[col])
    s = s[s["split"].isin(AXES["split"])]
    levels = {ax: {lv: float(s.loc[s[ax] == lv, col].mean()) for lv in lvs if (s[ax] == lv).any()} for ax, lvs in AXES.items()}
    rng = {ax: (max(v.values()) - min(v.values())) if v else float("nan") for ax, v in levels.items()}
    # additive model y ~ 1 + A + B + C on cell means; residual = interaction
    X = [np.ones(len(s))]
    for ax, lvs in AXES.items():
        for lv in lvs[1:]:
            X.append((s[ax] == lv).astype(float).values)
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, s[col].values, rcond=None)
    resid = s[col].values - X @ beta
    return dict(metric=metric, levels=levels, range=rng, interaction_rms=float(np.sqrt(np.mean(resid ** 2))),
                interaction_max_abs=float(np.max(np.abs(resid))) if len(resid) else float("nan"))


def core_pairs(df: pd.DataFrame) -> dict:
    s = cell_summary(df)

    def cell(cid):
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s.loc[cid].items()} if cid in s.index else None

    split = {cid: cell(cid) for cid in ("A2-B1-C1", "A3-B1-C1", "A1-B1-C1", "A4-B1-C1") if cid in s.index}
    if "A2-B1-C1" in s.index and "A3-B1-C1" in s.index:
        split["gap_own_macro_f1"] = float(s.loc["A2-B1-C1", "own_macro_f1_mean"] - s.loc["A3-B1-C1", "own_macro_f1_mean"])
        split["gap_gold_defect_f1"] = float(s.loc["A2-B1-C1", "gold_defect_f1_mean"] - s.loc["A3-B1-C1", "gold_defect_f1_mean"])
        split["A2_within_model_gap"] = float(s.loc["A2-B1-C1", "own_defect_f1_mean"] - s.loc["A2-B1-C1", "gold_defect_f1_mean"])
        split["A3_within_model_gap"] = float(s.loc["A3-B1-C1", "own_defect_f1_mean"] - s.loc["A3-B1-C1", "gold_defect_f1_mean"])
    cap = {cid: cell(cid) for cid in ("A3-B1-C1", "A3-B1-C2", "A3-B1-C3") if cid in s.index}
    if "A3-B1-C1" in s.index and "A3-B1-C3" in s.index:
        cap["gap_own_macro_f1"] = float(s.loc["A3-B1-C3", "own_macro_f1_mean"] - s.loc["A3-B1-C1", "own_macro_f1_mean"])
        cap["gap_gold_defect_f1"] = float(s.loc["A3-B1-C3", "gold_defect_f1_mean"] - s.loc["A3-B1-C1", "gold_defect_f1_mean"])
    return dict(split=split, cap=cap)


def _fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def write_tables(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = cell_summary(df)
    s.to_csv(out_dir / "cells.csv")
    lines = ["| cell | split | classes | cap | n_train | as-reported macro-F1 (own test) | gold defect-F1 (common) | gold 9-class macro-F1 | lot_share | dup_rate |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    ntr = df.groupby("cell_id")["n_train"].mean()
    ls = df.groupby("cell_id")["lot_share_rate"].mean()
    dr = df.groupby("cell_id")["dup_rate"].mean()
    for cid, r in s.iterrows():
        gf = _fmt(r["gold_full_macro_f1_mean"], r["gold_full_macro_f1_std"]) if not np.isnan(r["gold_full_macro_f1_mean"]) else "—"
        lines.append(f"| {cid} | {AXIS_LABELS['split'].get(r['split'], r['split'])} | {AXIS_LABELS['classes'][r['classes']]} | "
                     f"{AXIS_LABELS['cap'][r['cap']]} | {ntr[cid]:,.0f} | {_fmt(r['own_macro_f1_mean'], r['own_macro_f1_std'])} | "
                     f"{_fmt(r['gold_defect_f1_mean'], r['gold_defect_f1_std'])} | {gf} | {ls[cid]:.2f} | {dr[cid]:.3f} |")
    (out_dir / "cells.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    me_lines = ["| metric | axis | level means | range (max−min) | interaction RMS | interaction max |", "|---|---|---|---:|---:|---:|"]
    effects = {}
    for metric in ("own_macro_f1", "gold_defect_f1"):
        me = main_effects(s, metric)
        effects[metric] = me
        for ax in AXES:
            lv = ", ".join(f"{AXIS_LABELS[ax][k]}={v:.3f}" for k, v in me["levels"][ax].items())
            me_lines.append(f"| {metric} | {ax} | {lv} | {me['range'][ax]:.3f} | {me['interaction_rms']:.3f} | {me['interaction_max_abs']:.3f} |")
    (out_dir / "main_effects.md").write_text("\n".join(me_lines) + "\n", encoding="utf-8")

    cp = core_pairs(df)
    cp_lines = ["## Core pair 1 — random vs lot-group split (9 classes, no cap)", ""]
    for cid in ("A2-B1-C1", "A3-B1-C1", "A1-B1-C1", "A4-B1-C1"):
        if cid in cp["split"]:
            c = cp["split"][cid]
            cp_lines.append(f"- {cid}: own macro-F1 {_fmt(c['own_macro_f1_mean'], c['own_macro_f1_std'])}, own defect-F1 {_fmt(c['own_defect_f1_mean'], c['own_defect_f1_std'])}, "
                            f"gold defect-F1 {_fmt(c['gold_defect_f1_mean'], c['gold_defect_f1_std'])}")
    for k in ("gap_own_macro_f1", "gap_gold_defect_f1", "A2_within_model_gap", "A3_within_model_gap"):
        if k in cp["split"]:
            cp_lines.append(f"- {k}: {cp['split'][k]:+.3f}")
    cp_lines += ["", "## Core pair 2 — full vs balanced subset (lot-group split, 9 classes)", ""]
    for cid in ("A3-B1-C1", "A3-B1-C2", "A3-B1-C3"):
        if cid in cp["cap"]:
            c = cp["cap"][cid]
            cp_lines.append(f"- {cid}: own macro-F1 {_fmt(c['own_macro_f1_mean'], c['own_macro_f1_std'])}, gold defect-F1 {_fmt(c['gold_defect_f1_mean'], c['gold_defect_f1_std'])}")
    for k in ("gap_own_macro_f1", "gap_gold_defect_f1"):
        if k in cp["cap"]:
            cp_lines.append(f"- {k}: {cp['cap'][k]:+.3f}")
    (out_dir / "core_pairs.md").write_text("\n".join(cp_lines) + "\n", encoding="utf-8")

    noise = s[["own_macro_f1_std", "gold_defect_f1_std"]].describe().loc[["mean", "max"]]
    (out_dir / "seed_noise.md").write_text("| | own macro-F1 seed std | gold defect-F1 seed std |\n|---|---:|---:|\n" +
                                          "\n".join(f"| {i} | {r['own_macro_f1_std']:.4f} | {r['gold_defect_f1_std']:.4f} |" for i, r in noise.iterrows()) + "\n",
                                          encoding="utf-8")
    return dict(summary=s, effects=effects, core=cp)


def _bar_pair(ax, labels, means, stds, colors, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, width=0.55, color=colors, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", visible=False)


def plot_core_pair_split(cp: dict, out: Path):
    cells = [c for c in ("A1-B1-C1", "A2-B1-C1", "A3-B1-C1") if c in cp["split"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    names = {"A1-B1-C1": "original", "A2-B1-C1": "random", "A3-B1-C1": "lot-group"}
    cols = {"A1-B1-C1": AQUA, "A2-B1-C1": BLUE, "A3-B1-C1": ORANGE}
    _bar_pair(axes[0], [names[c] for c in cells], [cp["split"][c]["own_macro_f1_mean"] for c in cells],
              [cp["split"][c]["own_macro_f1_std"] for c in cells], [cols[c] for c in cells],
              "As reported: macro-F1 on each protocol's own test set", "macro-F1 (9 classes)")
    _bar_pair(axes[1], [names[c] for c in cells], [cp["split"][c]["gold_defect_f1_mean"] for c in cells],
              [cp["split"][c]["gold_defect_f1_std"] for c in cells], [cols[c] for c in cells],
              "Same models on the common lot-disjoint gold set", "defect-F1 (8 classes)")
    fig.suptitle("Split protocol (9 classes, no cap) — bars: mean of 3 seeds, whiskers: seed std", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_core_pair_cap(cp: dict, out: Path):
    cells = [c for c in ("A3-B1-C1", "A3-B1-C2", "A3-B1-C3") if c in cp["cap"]]
    names = {"A3-B1-C1": "no cap", "A3-B1-C2": "cap 5000", "A3-B1-C3": "balanced"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _bar_pair(axes[0], [names[c] for c in cells], [cp["cap"][c]["own_macro_f1_mean"] for c in cells],
              [cp["cap"][c]["own_macro_f1_std"] for c in cells], [BLUE] * len(cells),
              "As reported: macro-F1 on each protocol's own test set", "macro-F1 (9 classes)")
    _bar_pair(axes[1], [names[c] for c in cells], [cp["cap"][c]["gold_defect_f1_mean"] for c in cells],
              [cp["cap"][c]["gold_defect_f1_std"] for c in cells], [ORANGE] * len(cells),
              "Same models on the common lot-disjoint gold set", "defect-F1 (8 classes)")
    fig.suptitle("Sample-selection protocol (lot-group split, 9 classes)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_main_effects(effects: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, axis in zip(axes, AXES):
        for metric, color, label in (("own_macro_f1", BLUE, "as-reported (own test)"), ("gold_defect_f1", ORANGE, "on gold (common)")):
            lv = effects[metric]["levels"][axis]
            xs = np.arange(len(lv))
            ax.plot(xs, list(lv.values()), marker="o", ms=6, lw=2, color=color, label=label)
        ax.set_xticks(np.arange(len(lv)), [AXIS_LABELS[axis][k] for k in lv], rotation=0)
        ax.set_title(f"axis: {axis}", fontsize=10, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("level mean over the other axes")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Main effect of each protocol axis (18 cells, 3 seeds each)", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(runs_dir: Path, cell_id: str, out: Path):
    cms = []
    for d in sorted(runs_dir.glob(f"{cell_id}-s*")):
        m = json.loads((d / "metrics.json").read_text())
        if m.get("gold_full"):
            cms.append(np.array(m["gold_full"]["confusion"], dtype=float))
    if not cms:
        return
    cm = np.sum(cms, axis=0)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_title(f"{cell_id} on gold (row-normalised %, 3 seeds summed; n={int(cm.sum()):,})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_nn_hamming(runs_dir: Path, out: Path):
    series = []
    for cid, color, label in (("A2-B1-C1", BLUE, "random split"), ("A3-B1-C1", ORANGE, "lot-group split")):
        d = runs_dir / f"{cid}-s0" / "nn_hamming.npy"
        if d.exists():
            series.append((np.load(d), color, label))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    hi = max(int(np.percentile(h, 99)) for h, _, _ in series)
    bins = np.linspace(0, max(hi, 1), 60)
    for h, color, label in series:
        ax.hist(h, bins=bins, histtype="step", lw=2, color=color, label=f"{label} (median {np.median(h):.0f})", density=True)
    ax.set_xlabel("Hamming distance from each test wafer to its nearest training wafer (64x64 dies)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("How close is the nearest training example? (seed 0, 9 classes, no cap)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)
    root = Path(args.results)
    df = load_results(root / "results.csv")
    tables = write_tables(df, root / "tables")
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    plot_core_pair_split(tables["core"], figs / "core_pair_split.png")
    plot_core_pair_cap(tables["core"], figs / "core_pair_cap.png")
    plot_main_effects(tables["effects"], figs / "main_effects.png")
    plot_confusion(root / "runs", "A3-B1-C1", figs / "confusion_gold_A3-B1-C1.png")
    plot_nn_hamming(root / "runs", figs / "nn_hamming_A2_vs_A3.png")
    print((root / "tables" / "core_pairs.md").read_text())


if __name__ == "__main__":
    main()
