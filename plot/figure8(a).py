"""from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =========================
# 全局画图风格
# =========================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
})


METHODS = ["method1", "method2", "method3", "method4"]
KS = [100, 200, 300, 400, 500, 600, 670]

METHOD_LABELS = {
    "method1": "Camels Attr.",
    "method2": "AlphaEarth",
    "method3": "Camels Emb.",
    "method4": "Random",
}

METHOD_COLORS = {
    "method1": "#1f78b4",
    "method2": "#e31a1c",
    "method3": "#33a02c",
    "method4": "#6a3d9a",
}

METHOD_MARKERS = {
    "method1": "o",
    "method2": "s",
    "method3": "^",
    "method4": "d",
}


# =========================
# 路径设置
# =========================
SUMMARY_DIR = Path("final_camels_test_nse_summary")
EXTRA_RESULT_DIR = Path("/home/pengfeiqu/torchhydro/result")


def read_metric_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        lower_cols = {str(c).lower(): c for c in df.columns}

        if "nse" in lower_cols:
            return float(df.iloc[0][lower_cols["nse"]])

        df = pd.read_csv(csv_path, header=None)

    except Exception:
        df = pd.read_csv(csv_path, header=None)

    if str(df.iloc[0, 0]).lower() in ["basin_id", "basin", "gauge_id"]:
        df = df.iloc[1:].reset_index(drop=True)

    return float(df.iloc[0, 1])


def parse_extra_csv_name(filename: str):
    m = re.match(r"^(1|2|3|random)_(\d+)_metric_streamflow\.csv$", filename)

    if m is None:
        return None

    method_raw = m.group(1)
    k = int(m.group(2))

    if method_raw == "random":
        method = "method4"
    else:
        method = f"method{method_raw}"

    return method, k


def load_existing_all_rows(summary_dir: Path):
    all_rows_path = summary_dir / "final_all_rows.csv"

    if not all_rows_path.exists():
        raise FileNotFoundError(f"Cannot find {all_rows_path}")

    df = pd.read_csv(all_rows_path, dtype={"basin_id": str})
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    df["k"] = df["k"].astype(int)
    df["nse"] = df["nse"].astype(float)

    return df


def collect_extra_basins(extra_result_dir: Path, existing_df: pd.DataFrame):
    existing_basins = set(existing_df["basin_id"].astype(str).str.zfill(8))

    new_rows = []
    skipped_basins = []
    added_basins = []

    for basin_dir in sorted(extra_result_dir.iterdir()):
        if not basin_dir.is_dir():
            continue

        basin_id = str(basin_dir.name).zfill(8)

        if basin_id in existing_basins:
            skipped_basins.append(basin_id)
            continue

        basin_rows = []

        for csv_path in sorted(basin_dir.glob("*_metric_streamflow.csv")):
            parsed = parse_extra_csv_name(csv_path.name)

            if parsed is None:
                continue

            method, k = parsed

            if method not in METHODS or k not in KS:
                continue

            nse = read_metric_csv(csv_path)

            basin_rows.append({
                "server": "extra",
                "basin_id": basin_id,
                "method": method,
                "k": k,
                "nse": nse,
                "source": "extra_result",
            })

        if basin_rows:
            new_rows.extend(basin_rows)
            added_basins.append(basin_id)
        else:
            print(f"[WARN] No valid csv found in {basin_dir}")

    new_df = pd.DataFrame(new_rows)

    print(f"Skipped existing basins: {skipped_basins}")
    print(f"Added new basins: {added_basins}")
    print(f"New rows added: {len(new_df)}")

    return new_df


def regenerate_summary_and_plots(all_df: pd.DataFrame, summary_dir: Path):
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_df["basin_id"] = all_df["basin_id"].astype(str).str.zfill(8)
    all_df["k"] = all_df["k"].astype(int)
    all_df["nse"] = all_df["nse"].astype(float)

    all_df = all_df.sort_values(
        ["basin_id", "method", "k", "server"]
    ).reset_index(drop=True)

    all_df.to_csv(summary_dir / "final_all_rows.csv", index=False)

    basin_median = (
        all_df
        .groupby(["basin_id", "method", "k"], as_index=False)
        .agg(
            median_nse=("nse", "median"),
            count=("nse", "count"),
        )
    )

    basin_median = basin_median.sort_values(
        ["basin_id", "method", "k"]
    ).reset_index(drop=True)

    basin_median.to_csv(
        summary_dir / "basin_method_k_median.csv",
        index=False,
    )

    overall_median = (
        basin_median
        .groupby(["method", "k"], as_index=False)
        .agg(
            median_nse_all_basins=("median_nse", "median"),
            basin_count=("basin_id", "nunique"),
        )
    )

    overall_median = overall_median.sort_values(
        ["method", "k"]
    ).reset_index(drop=True)

    overall_median.to_csv(
        summary_dir / "overall_method_k_median.csv",
        index=False,
    )

    plot_overall_median(overall_median, summary_dir)
    plot_each_basin_median(basin_median, summary_dir)

    print("[OK] Regenerated all csv, png and pdf files.")


def plot_overall_median(overall_median: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        sub = overall_median[
            overall_median["method"] == method
        ].sort_values("k")

        if sub.empty:
            continue

        ax.plot(
            sub["k"],
            sub["median_nse_all_basins"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.2,
            markersize=6,
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel("Number of training basins, k")
    ax.set_ylabel("Median NSE across all basins")
    ax.set_title("Overall median NSE for different methods and k")
    ax.set_xticks(KS)
    ax.set_ylim(0, 0.9)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "overall_median_nse.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "overall_median_nse.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_each_basin_median(basin_median: pd.DataFrame, out_dir: Path):
    pdf_path = out_dir / "per_basin_median_nse.pdf"

    with PdfPages(pdf_path) as pdf:
        for basin_id, g in basin_median.groupby("basin_id"):
            fig, ax = plt.subplots(figsize=(8, 5))

            for method in METHODS:
                sub = g[g["method"] == method].sort_values("k")

                if sub.empty:
                    continue

                ax.plot(
                    sub["k"],
                    sub["median_nse"],
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    linewidth=2.2,
                    markersize=6,
                    label=METHOD_LABELS[method],
                )

            ax.set_xlabel("Number of training basins, k")
            ax.set_ylabel("Median NSE")
            ax.set_title(f"Basin {basin_id}: median NSE")
            ax.set_xticks(KS)
            ax.set_ylim(0, 0.9)
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.legend(frameon=False)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[OK] Saved {pdf_path}")


def main():
    existing_df = load_existing_all_rows(SUMMARY_DIR)
    extra_df = collect_extra_basins(EXTRA_RESULT_DIR, existing_df)

    if extra_df.empty:
        print("[INFO] No new basins added. Regenerating summaries from existing data.")
        updated_df = existing_df.copy()
    else:
        updated_df = pd.concat(
            [existing_df, extra_df],
            ignore_index=True,
        )

    regenerate_summary_and_plots(updated_df, SUMMARY_DIR)


if __name__ == "__main__":
    main()"""

from pathlib import Path
import re
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib.backends.backend_pdf import PdfPages


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
# 全局画图风格
# =========================
rcParams.update(
    {
        "text.usetex": False,
        # 使用刚才加载的 Times New Roman
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        # 所有文字统一 18
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
        "figure.titlesize": 18,
        "axes.unicode_minus": False,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        # 保存 PDF/PS 时嵌入 TrueType 字体
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


METHODS = ["method1", "method2", "method3", "method4"]
KS = [100, 200, 300, 400, 500, 600, 670]

METHOD_LABELS = {
    "method1": "Camels Attr.",
    "method2": "AlphaEarth",
    "method3": "Camels Emb.",
    "method4": "Random",
}

METHOD_COLORS = {
    "method1": "#1f78b4",
    "method2": "#e31a1c",
    "method3": "#33a02c",
    "method4": "#6a3d9a",
}

METHOD_MARKERS = {
    "method1": "o",
    "method2": "s",
    "method3": "^",
    "method4": "d",
}


# =========================
# 路径设置
# =========================
SUMMARY_DIR = Path("outputs")
EXTRA_RESULT_DIR = Path("/home/pengfeiqu/torchhydro/result")


# =========================
# 自动 y 轴范围
# =========================
def get_auto_ylim(values, margin_ratio=0.08):
    values = pd.Series(values).dropna()

    if values.empty:
        return 0, 1

    ymin = values.min()
    ymax = values.max()

    if ymin == ymax:
        margin = 0.1
    else:
        margin = (ymax - ymin) * margin_ratio

    return ymin - margin, ymax + margin


def read_metric_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        lower_cols = {str(c).lower(): c for c in df.columns}

        if "nse" in lower_cols:
            return float(df.iloc[0][lower_cols["nse"]])

        df = pd.read_csv(csv_path, header=None)

    except Exception:
        df = pd.read_csv(csv_path, header=None)

    if str(df.iloc[0, 0]).lower() in ["basin_id", "basin", "gauge_id"]:
        df = df.iloc[1:].reset_index(drop=True)

    return float(df.iloc[0, 1])


def parse_extra_csv_name(filename: str):
    m = re.match(r"^(1|2|3|random)_(\d+)_metric_streamflow\.csv$", filename)

    if m is None:
        return None

    method_raw = m.group(1)
    k = int(m.group(2))

    if method_raw == "random":
        method = "method4"
    else:
        method = f"method{method_raw}"

    return method, k


def load_existing_all_rows(summary_dir: Path):
    all_rows_path = summary_dir / "final_all_rows.csv"

    if not all_rows_path.exists():
        raise FileNotFoundError(f"Cannot find {all_rows_path}")

    df = pd.read_csv(all_rows_path, dtype={"basin_id": str})
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    df["k"] = df["k"].astype(int)
    df["nse"] = df["nse"].astype(float)

    return df


def collect_extra_basins(extra_result_dir: Path, existing_df: pd.DataFrame):
    existing_basins = set(existing_df["basin_id"].astype(str).str.zfill(8))

    new_rows = []
    skipped_basins = []
    added_basins = []

    for basin_dir in sorted(extra_result_dir.iterdir()):
        if not basin_dir.is_dir():
            continue

        basin_id = str(basin_dir.name).zfill(8)

        if basin_id in existing_basins:
            skipped_basins.append(basin_id)
            continue

        basin_rows = []

        for csv_path in sorted(basin_dir.glob("*_metric_streamflow.csv")):
            parsed = parse_extra_csv_name(csv_path.name)

            if parsed is None:
                continue

            method, k = parsed

            if method not in METHODS or k not in KS:
                continue

            nse = read_metric_csv(csv_path)

            basin_rows.append(
                {
                    "server": "extra",
                    "basin_id": basin_id,
                    "method": method,
                    "k": k,
                    "nse": nse,
                    "source": "extra_result",
                }
            )

        if basin_rows:
            new_rows.extend(basin_rows)
            added_basins.append(basin_id)
        else:
            print(f"[WARN] No valid csv found in {basin_dir}")

    new_df = pd.DataFrame(new_rows)

    print(f"Skipped existing basins: {skipped_basins}")
    print(f"Added new basins: {added_basins}")
    print(f"New rows added: {len(new_df)}")

    return new_df


def regenerate_summary_and_plots(all_df: pd.DataFrame, summary_dir: Path):
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_df["basin_id"] = all_df["basin_id"].astype(str).str.zfill(8)
    all_df["k"] = all_df["k"].astype(int)
    all_df["nse"] = all_df["nse"].astype(float)

    all_df = all_df.sort_values(["basin_id", "method", "k", "server"]).reset_index(
        drop=True
    )

    all_df.to_csv(summary_dir / "final_all_rows.csv", index=False)

    basin_median = all_df.groupby(["basin_id", "method", "k"], as_index=False).agg(
        median_nse=("nse", "median"),
        count=("nse", "count"),
    )

    basin_median = basin_median.sort_values(["basin_id", "method", "k"]).reset_index(
        drop=True
    )

    basin_median.to_csv(
        summary_dir / "basin_method_k_median.csv",
        index=False,
    )

    overall_median = basin_median.groupby(["method", "k"], as_index=False).agg(
        median_nse_all_basins=("median_nse", "median"),
        basin_count=("basin_id", "nunique"),
    )

    overall_median = overall_median.sort_values(["method", "k"]).reset_index(drop=True)

    overall_median.to_csv(
        summary_dir / "overall_method_k_median.csv",
        index=False,
    )

    plot_overall_median(overall_median, summary_dir)
    plot_each_basin_median(basin_median, summary_dir)

    print("[OK] Regenerated all csv, png and pdf files.")


def plot_overall_median(overall_median: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHODS:
        sub = overall_median[overall_median["method"] == method].sort_values("k")

        if sub.empty:
            continue

        ax.plot(
            sub["k"],
            sub["median_nse_all_basins"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=2.2,
            markersize=6,
            label=METHOD_LABELS[method],
        )

    ymin, ymax = get_auto_ylim(overall_median["median_nse_all_basins"])

    ax.set_xlabel("Number of training basins, k", fontsize=18)
    ax.set_ylabel("Median NSE across all basins", fontsize=18)
    ax.set_title("Overall median NSE for different methods and k", fontsize=18)

    ax.set_xticks(KS)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_ylim(ymin, ymax)

    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=12,
    )

    fig.tight_layout()

    fig.savefig(
        out_dir / "overall_median_nse.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir / "overall_median_nse.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_each_basin_median(basin_median: pd.DataFrame, out_dir: Path):
    pdf_path = out_dir / "per_basin_median_nse_3x3.pdf"

    basin_ids = sorted(basin_median["basin_id"].unique())
    basins_per_page = 9
    n_pages = math.ceil(len(basin_ids) / basins_per_page)

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            page_basins = basin_ids[
                page * basins_per_page : (page + 1) * basins_per_page
            ]

            fig, axes = plt.subplots(
                3,
                3,
                figsize=(18, 14),
                sharex=False,
                sharey=False,
            )

            axes = axes.flatten()

            for ax, basin_id in zip(axes, page_basins):
                g = basin_median[basin_median["basin_id"] == basin_id]

                for method in METHODS:
                    sub = g[g["method"] == method].sort_values("k")

                    if sub.empty:
                        continue

                    ax.plot(
                        sub["k"],
                        sub["median_nse"],
                        marker=METHOD_MARKERS[method],
                        color=METHOD_COLORS[method],
                        linewidth=2.0,
                        markersize=5,
                        label=METHOD_LABELS[method],
                    )

                ymin, ymax = get_auto_ylim(g["median_nse"])

                ax.set_title(f"Basin {basin_id}", fontsize=18)
                ax.set_xticks(KS)
                ax.tick_params(axis="both", labelsize=18)
                ax.set_ylim(ymin, ymax)

                ax.grid(True, linestyle=":", alpha=0.35)

                ax.legend(
                    loc="lower right",
                    frameon=False,
                    fontsize=10,
                )

            for ax in axes[len(page_basins) :]:
                ax.axis("off")

            fig.supxlabel(
                "Number of training basins, k",
                fontsize=18,
            )

            fig.supylabel(
                "Median NSE",
                fontsize=18,
            )

            fig.suptitle(
                f"Per-basin median NSE, page {page + 1}/{n_pages}",
                fontsize=18,
                y=0.995,
            )

            fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(fig)

    print(f"[OK] Saved {pdf_path}")


def main():
    existing_df = load_existing_all_rows(SUMMARY_DIR)
    extra_df = collect_extra_basins(EXTRA_RESULT_DIR, existing_df)

    if extra_df.empty:
        print("[INFO] No new basins added. Regenerating summaries from existing data.")
        updated_df = existing_df.copy()
    else:
        updated_df = pd.concat(
            [existing_df, extra_df],
            ignore_index=True,
        )

    regenerate_summary_and_plots(updated_df, SUMMARY_DIR)


if __name__ == "__main__":
    main()
