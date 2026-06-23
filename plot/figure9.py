"""Plot NSE curves for five selected target basins.

The script first verifies that all requested target basins have complete
method/k results on the current machine. If any selected basin is incomplete,
it prints the missing combinations and exits without drawing.

Example:
    python scripts/plot_selected_basin_nse_panels.py \
      --results-root /home/pengfeiqu/torchhydro/experiments/results/camels_test \
      --out /home/pengfeiqu/torchhydro/selected_basin_nse_panels.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import pandas as pd


# =========================
# 加载 Times New Roman 字体
# =========================
FONT_DIR = Path.home() / ".local/share/fonts/times_new_roman"

font_files = list(FONT_DIR.glob("*.ttf"))

if not font_files:
    print(f"[WARN] No Times New Roman font files found in {FONT_DIR}")
else:
    for font_file in font_files:
        font_manager.fontManager.addfont(str(font_file))

    print("[OK] Loaded font files:")
    for font_file in font_files:
        print(f"  {font_file}")


# =========================
# 全局画图风格：Times New Roman + 所有字 18
# =========================
rcParams.update(
    {
        "text.usetex": False,
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
        "figure.titlesize": 18,
        "axes.unicode_minus": False,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


K_VALUES = [100, 200, 300, 400, 500, 600, 670]

METHODS = ["method1", "method2", "method3", "method4"]

METHOD_LABELS = {
    "method1": "Camels Attr.",
    "method2": "AlphaEarth",
    "method3": "Camels Emb.",
    "method4": "Random",
}

METHOD_COLORS = {
    "method1": "#1f77b4",
    "method2": "#e41a1c",
    "method3": "#2ca02c",
    "method4": "#6a3d9a",
}

METHOD_MARKERS = {
    "method1": "o",
    "method2": "s",
    "method3": "^",
    "method4": "d",
}

# 默认绘制的 5 个目标流域
DEFAULT_TARGETS = ["1013500", "14138900", "6339500", "9065500", "10249300"]

# 只有这个流域的 y 轴范围设置为 -1 到 1
SPECIAL_YLIM_TARGET = "10249300"

FOLDER_RE = re.compile(
    r"simplelstm_.*?_ens_(?P<target>\d{8})_method(?P<method>\d+)_k(?P<k>\d+).*_test$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NSE vs k for selected basins.")

    parser.add_argument(
        "--results-root",
        action="append",
        default=[],
        help="Directory containing simplelstm_*_test result folders. Can be repeated.",
    )

    parser.add_argument(
        "--detail-csv",
        action="append",
        default=[],
        help="Existing detail CSV with basin_id/target, method, k, NSE columns. Can be repeated.",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target basin ids. Leading zeros are optional.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        choices=METHODS,
        help="Methods that must be present and plotted.",
    )

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=K_VALUES,
        help="Training basin counts that must be present and plotted.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output figure path.",
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Output merged detail CSV path. Default: <out stem>_detail.csv",
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate target/method/k rows instead of dropping duplicates.",
    )

    return parser.parse_args()


def zfill_basin_id(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text.zfill(8)


def display_basin_id(value: str) -> str:
    return str(int(value)) if str(value).isdigit() else value


def method_name(method_num: str | int) -> str:
    return f"method{int(method_num)}"


def rows_from_results_root(root: Path) -> list[dict]:
    rows: list[dict] = []

    for folder in sorted(root.glob("simplelstm_*_test")):
        if not folder.is_dir():
            continue

        match = FOLDER_RE.match(folder.name)

        if not match:
            continue

        metric_file = folder / "metric_streamflow.csv"

        if not metric_file.exists():
            continue

        method = method_name(match.group("method"))
        k = int(match.group("k"))
        target = zfill_basin_id(match.group("target"))

        metric = pd.read_csv(metric_file)

        if metric.shape[1] < 2:
            continue

        basin_col = metric.columns[0]
        nse_col = "NSE" if "NSE" in metric.columns else metric.columns[1]

        for _, metric_row in metric.iterrows():
            nse = pd.to_numeric(metric_row[nse_col], errors="coerce")

            if pd.isna(nse):
                continue

            basin_id = zfill_basin_id(metric_row[basin_col])

            rows.append(
                {
                    "source": str(root),
                    "folder": folder.name,
                    "target": target,
                    "basin_id": basin_id,
                    "method": method,
                    "k": k,
                    "NSE": float(nse),
                }
            )

    return rows


def load_detail_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "NSE" not in df.columns and "nse" in df.columns:
        df = df.rename(columns={"nse": "NSE"})

    required = {"method", "k", "NSE"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()

    if "target" not in df.columns:
        if "basin_id" not in df.columns:
            raise ValueError(f"{path} needs either target or basin_id column")
        df["target"] = df["basin_id"]

    if "basin_id" not in df.columns:
        df["basin_id"] = df["target"]

    df["target"] = df["target"].map(zfill_basin_id)
    df["basin_id"] = df["basin_id"].map(zfill_basin_id)
    df["method"] = df["method"].astype(str)
    df["k"] = df["k"].astype(int)
    df["NSE"] = pd.to_numeric(df["NSE"], errors="coerce")
    df["source"] = str(path)

    return df.dropna(subset=["NSE"])


def load_results(results_roots: list[str], detail_csvs: list[str]) -> pd.DataFrame:
    frames = []

    for root_str in results_roots:
        root = Path(root_str)
        rows = rows_from_results_root(root)

        if rows:
            frames.append(pd.DataFrame(rows))
        else:
            print(f"[WARN] no metric rows found under {root}")

    for csv_str in detail_csvs:
        frames.append(load_detail_csv(Path(csv_str)))

    if not frames:
        raise SystemExit("No input rows found. Provide --results-root or --detail-csv.")

    return pd.concat(frames, ignore_index=True)


def missing_selected_results(
    detail: pd.DataFrame,
    targets: list[str],
    methods: list[str],
    ks: list[int],
) -> list[tuple[str, str, int]]:
    available = set(
        detail[["target", "method", "k"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    missing = []

    for target in targets:
        for method in methods:
            for k in ks:
                if (target, method, k) not in available:
                    missing.append((target, method, k))

    return missing


def set_nse_ylim_for_target_axis(ax, target: str) -> None:
    """
    设置每个目标流域子图的 NSE y 轴范围。

    只有 10249300 使用 -1 到 1；
    其他流域使用 0 到 1。
    """
    special_target = zfill_basin_id(SPECIAL_YLIM_TARGET)

    if target == special_target:
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def make_five_panel_axes() -> tuple[plt.Figure, list[plt.Axes]]:
    """
    创建 5 个子图：
    上排 3 个，下排 2 个居中。
    不再创建第 6 个汇总子图。
    """
    fig = plt.figure(figsize=(18, 9.2))

    # 2 行 6 列：
    # 上排每个图占 2 列；
    # 下排两个图居中，各占 2 列。
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1, 1],
        hspace=0.42,
        wspace=0.65,
    )

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    return fig, axes


def plot_panels(
    detail: pd.DataFrame,
    targets: list[str],
    methods: list[str],
    ks: list[int],
    out: Path,
) -> None:
    fig, axes = make_five_panel_axes()

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for panel_idx, target in enumerate(targets):
        ax = axes[panel_idx]
        target_df = detail[detail["target"] == target]

        for method in methods:
            part = (
                target_df[target_df["method"] == method]
                .sort_values("k")
                .drop_duplicates(subset=["k"], keep="first")
            )

            if part.empty:
                continue

            ax.plot(
                part["k"],
                part["NSE"],
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method),
                linewidth=1.4,
                markersize=4,
                label=METHOD_LABELS.get(method, method),
            )

        ax.set_title(
            f"Basin {display_basin_id(target)}",
            fontsize=18,
            fontname="Times New Roman",
        )
        ax.set_xlabel(
            panel_labels[panel_idx],
            fontsize=18,
            fontname="Times New Roman",
        )
        ax.set_ylabel(
            "NSE",
            fontsize=18,
            fontname="Times New Roman",
        )
        ax.set_xticks(ks)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        set_nse_ylim_for_target_axis(ax, target)

    # 总图例
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=18,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=300, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    args = parse_args()

    targets = [zfill_basin_id(x) for x in args.targets]
    methods = args.methods
    ks = args.ks

    detail = load_results(args.results_root, args.detail_csv)

    detail = detail[detail["method"].isin(methods) & detail["k"].isin(ks)].copy()

    before = len(detail)

    if not args.keep_duplicates:
        detail = detail.drop_duplicates(
            subset=["target", "method", "k"],
            keep="first",
        )

    dropped = before - len(detail)

    missing = missing_selected_results(detail, targets, methods, ks)

    if missing:
        print(
            "[ERROR] Selected basins are incomplete on this server. Figure not generated."
        )

        missing_df = pd.DataFrame(
            missing,
            columns=["target", "method", "k"],
        )

        print(missing_df.to_string(index=False))

        raise SystemExit(2)

    out = Path(args.out)

    detail_csv = Path(args.csv) if args.csv else out.with_name(f"{out.stem}_detail.csv")

    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_csv, index=False)

    plot_panels(
        detail=detail,
        targets=targets,
        methods=methods,
        ks=ks,
        out=out,
    )

    print(f"[DONE] selected basins complete: {', '.join(targets)}")
    print(f"[DONE] rows: {len(detail)}")
    print(f"[DONE] dropped duplicates: {dropped}")
    print(f"[DONE] wrote detail -> {detail_csv}")
    print(f"[DONE] wrote plot -> {out}")


if __name__ == "__main__":
    main()
