#!/usr/bin/env python
"""Build Table-II summary tables and figures from existing eval CSV (no training)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WAVELET_LABELS = {
    "three_wavelets_1_subband": "3W-1S",
    "one_wavelet_4_subbands": "1W-4S",
    "one_wavelet_1_subband": "1W-1S",
}

SIZE_ORDER = ["32x32", "60x60", "116x116", "228x228", "452x452"]
WAVELET_ORDER = [
    "three_wavelets_1_subband",
    "one_wavelet_4_subbands",
    "one_wavelet_1_subband",
]


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["level_num"] = df["level"].astype(str).str.replace("Lv", "", regex=False).astype(int)
    df["size_num"] = df["image_size"].str.split("x").str[0].astype(int)
    df["wavelet_short"] = df["wavelet_type"].map(WAVELET_LABELS)
    df["accuracy"] = df["accuracy"].astype(float)
    df["f_measure_macro"] = df["f_measure_macro"].astype(float)
    df["f_measure_abnormal"] = df["f_measure_abnormal"].astype(float)
    df["wavelet_type"] = pd.Categorical(df["wavelet_type"], categories=WAVELET_ORDER, ordered=True)
    df["image_size"] = pd.Categorical(df["image_size"], categories=SIZE_ORDER, ordered=True)
    return df.sort_values(["image_size", "wavelet_type", "level_num"]).reset_index(drop=True)


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    slim = df[
        [
            "wavelet_type",
            "wavelet_short",
            "image_size",
            "level",
            "accuracy",
            "f_measure_macro",
            "f_measure_abnormal",
        ]
    ].copy()
    slim.to_csv(out_dir / "table2_metrics.csv", index=False, float_format="%.4f")

    # Pivot: accuracy by size x wavelet (best level per cell)
    best = (
        slim.sort_values("accuracy", ascending=False)
        .groupby(["image_size", "wavelet_short"], as_index=False, observed=True)
        .first()
    )
    pivot_acc = best.pivot(index="image_size", columns="wavelet_short", values="accuracy")
    pivot_acc = pivot_acc.reindex(index=SIZE_ORDER, columns=["3W-1S", "1W-4S", "1W-1S"])
    pivot_acc.to_csv(out_dir / "table2_best_accuracy_by_size.csv", float_format="%.4f")

    pivot_fabn = best.pivot(index="image_size", columns="wavelet_short", values="f_measure_abnormal")
    pivot_fabn = pivot_fabn.reindex(index=SIZE_ORDER, columns=["3W-1S", "1W-4S", "1W-1S"])
    pivot_fabn.to_csv(out_dir / "table2_best_f_abnormal_by_size.csv", float_format="%.4f")

    top = slim.sort_values("accuracy", ascending=False).head(10)
    bottom = slim.sort_values("accuracy", ascending=True).head(5)

    lines = [
        "# Table-II Local Results (from existing eval, no retraining)",
        "",
        f"Runs: **{len(slim)}** / 45",
        "",
        f"- Mean accuracy: **{slim['accuracy'].mean():.4f}**",
        f"- Mean F(abnormal): **{slim['f_measure_abnormal'].mean():.4f}**",
        f"- Best: `{top.iloc[0]['wavelet_short']}` / `{top.iloc[0]['image_size']}` / `{top.iloc[0]['level']}` "
        f"(acc={top.iloc[0]['accuracy']:.4f})",
        f"- Worst: `{bottom.iloc[0]['wavelet_short']}` / `{bottom.iloc[0]['image_size']}` / `{bottom.iloc[0]['level']}` "
        f"(acc={bottom.iloc[0]['accuracy']:.4f})",
        "",
        "## Best accuracy per size × wavelet (best level kept)",
        "",
        pivot_acc.to_markdown(floatfmt=".4f"),
        "",
        "## Best F(abnormal) per size × wavelet",
        "",
        pivot_fabn.to_markdown(floatfmt=".4f"),
        "",
        "## Top 10 by accuracy",
        "",
        top[["wavelet_short", "image_size", "level", "accuracy", "f_measure_abnormal"]]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Full metrics",
        "",
        slim[["wavelet_short", "image_size", "level", "accuracy", "f_measure_macro", "f_measure_abnormal"]]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_dir / "table2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    best = (
        df.sort_values("accuracy", ascending=False)
        .groupby(["image_size", "wavelet_short"], as_index=False, observed=True)
        .first()
    )
    mat = best.pivot(index="image_size", columns="wavelet_short", values="accuracy")
    mat = mat.reindex(index=SIZE_ORDER, columns=["3W-1S", "1W-4S", "1W-1S"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis", vmin=0.92, vmax=1.0, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns.tolist())
    ax.set_yticks(range(mat.shape[0]), mat.index.tolist())
    ax.set_xlabel("Wavelet setting")
    ax.set_ylabel("Image size")
    ax.set_title("Best accuracy (best level per cell)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_bars_by_size(df: pd.DataFrame, out_path: Path) -> None:
    best = (
        df.sort_values("accuracy", ascending=False)
        .groupby(["image_size", "wavelet_short"], as_index=False, observed=True)
        .first()
    )
    sizes = SIZE_ORDER
    wavelets = ["3W-1S", "1W-4S", "1W-1S"]
    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, w in enumerate(wavelets):
        vals = []
        for s in sizes:
            row = best[(best["image_size"] == s) & (best["wavelet_short"] == w)]
            vals.append(float(row["accuracy"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=w)

    ax.set_xticks(x, sizes)
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Image size")
    ax.set_title("Best accuracy by size and wavelet")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_level_curves(df: pd.DataFrame, out_path: Path) -> None:
    # For sizes that have multiple levels: 116/228/452
    focus_sizes = ["116x116", "228x228", "452x452"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, size in zip(axes, focus_sizes):
        sub = df[df["image_size"] == size]
        for w, short in WAVELET_LABELS.items():
            part = sub[sub["wavelet_type"] == w].sort_values("level_num")
            if part.empty:
                continue
            ax.plot(part["level_num"], part["accuracy"], marker="o", label=short)
        ax.set_title(size)
        ax.set_xlabel("Wavelet level")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xticks(sorted(sub["level_num"].unique()))
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.92, 1.0)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Accuracy vs decomposition level", y=1.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metric_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for short in ["3W-1S", "1W-4S", "1W-1S"]:
        part = df[df["wavelet_short"] == short]
        ax.scatter(part["accuracy"], part["f_measure_abnormal"], s=36, alpha=0.85, label=short)
    ax.plot([0.92, 1.0], [0.92, 1.0], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(0.92, 1.0)
    ax.set_ylim(0.92, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("F-measure (Abnormal)")
    ax.set_title("Accuracy vs F(abnormal), all 45 runs")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="results/table2_results.csv",
        help="Path to aggregated Table-II metrics CSV",
    )
    parser.add_argument(
        "--out-dir",
        default="results/summary",
        help="Output directory for tables and figures",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    write_tables(df, out_dir)
    plot_heatmap(df, out_dir / "fig_table2_best_accuracy_heatmap.png")
    plot_bars_by_size(df, out_dir / "fig_table2_best_accuracy_bars.png")
    plot_level_curves(df, out_dir / "fig_table2_accuracy_vs_level.png")
    plot_metric_scatter(df, out_dir / "fig_table2_acc_vs_fabnormal.png")

    print(f"Wrote tables and figures to {out_dir.resolve()}")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
