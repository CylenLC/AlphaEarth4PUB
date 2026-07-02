import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path
from matplotlib import rcParams, font_manager


# ============================
# Times New Roman 字体设置
# ============================
# 兼容 Mac 和服务器两种情况
font_candidates = [
    Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf"),
]

# 如果服务器上也有这个字体目录，也一起加载
server_font_dir = Path.home() / ".local/share/fonts/times_new_roman"
if server_font_dir.exists():
    font_candidates.extend(list(server_font_dir.glob("*.ttf")))

for font_file in font_candidates:
    if font_file.exists():
        font_manager.fontManager.addfont(str(font_file))
        print(f"[OK] Loaded font: {font_file}")

rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def process_files(folder_path):
    all_data = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith("metric_streamflow.csv"):
            continue

        parts = filename.split("_")

        if len(parts) < 3 or "seed" not in parts[1]:
            print("Unrecognized filename:", filename)
            continue

        method = parts[0]
        seed = parts[1].replace("seed", "")
        fold = parts[2]

        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        nse_values = df.iloc[:, 3].to_numpy(dtype=float)
        nse_values = nse_values[~np.isnan(nse_values)]

        all_data.setdefault(method, {}).setdefault(seed, {})[fold] = nse_values

    return all_data


def gather_values(data, method_name):
    vals = []

    for seed in data[method_name]:
        for fold in data[method_name][seed]:
            vals.extend(data[method_name][seed][fold])

    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return vals


def bootstrap_cdf(vals, n_boot=100, sample_frac=0.8, x_grid=None, random_state=42):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]

    if x_grid is None:
        x_grid = np.linspace(0, 1, 500)

    rng = np.random.default_rng(random_state)

    n = len(vals)
    sample_size = max(1, int(sample_frac * n))

    boot_cdfs = []

    for _ in range(n_boot):
        sample = rng.choice(vals, size=sample_size, replace=True)
        sample_sorted = np.sort(sample)

        cdf_vals = np.searchsorted(sample_sorted, x_grid, side="right") / len(
            sample_sorted
        )
        boot_cdfs.append(cdf_vals)

    boot_cdfs = np.asarray(boot_cdfs)

    median_cdf = np.percentile(boot_cdfs, 50, axis=0)
    lower_cdf = np.percentile(boot_cdfs, 5, axis=0)
    upper_cdf = np.percentile(boot_cdfs, 95, axis=0)

    return x_grid, median_cdf, lower_cdf, upper_cdf


def empirical_cdf(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]

    vals_sorted = np.sort(vals)
    cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)

    return vals_sorted, cdf


def plot_cdf_with_uncertainty(
    is_data, oos_data, method_names, n_boot=100, sample_frac=0.8, random_state=42
):
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    x_grid = np.linspace(0, 1, 500)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycler = cycle(colors)

    # IS 图中画 OOS 的灰色背景 CDF
    for method_name in method_names:
        if method_name not in oos_data:
            continue

        oos_vals = gather_values(oos_data, method_name)

        if len(oos_vals) == 0:
            continue

        oos_sorted, oos_cdf = empirical_cdf(oos_vals)

        ax1.plot(
            oos_sorted,
            oos_cdf,
            color="lightgray",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )

    # OOS 图中画 IS 的灰色背景 CDF
    for method_name in method_names:
        if method_name not in is_data:
            continue

        is_vals = gather_values(is_data, method_name)

        if len(is_vals) == 0:
            continue

        is_sorted, is_cdf = empirical_cdf(is_vals)

        ax2.plot(
            is_sorted, is_cdf, color="lightgray", linestyle="--", linewidth=1, zorder=1
        )

    # 彩色主曲线：bootstrap median CDF + 5%-95% uncertainty band
    for idx, method_name in enumerate(method_names):
        if method_name not in is_data or method_name not in oos_data:
            print(f"Skip {method_name}: not found in IS or OOS data.")
            continue

        color = next(color_cycler)

        is_vals = gather_values(is_data, method_name)
        oos_vals = gather_values(oos_data, method_name)

        if len(is_vals) == 0 or len(oos_vals) == 0:
            print(f"Skip {method_name}: empty values.")
            continue

        x_is, is_cdf_med, is_cdf_low, is_cdf_high = bootstrap_cdf(
            is_vals,
            n_boot=n_boot,
            sample_frac=sample_frac,
            x_grid=x_grid,
            random_state=random_state + idx,
        )

        x_oos, oos_cdf_med, oos_cdf_low, oos_cdf_high = bootstrap_cdf(
            oos_vals,
            n_boot=n_boot,
            sample_frac=sample_frac,
            x_grid=x_grid,
            random_state=random_state + 1000 + idx,
        )

        is_median = np.nanmedian(is_vals)
        oos_median = np.nanmedian(oos_vals)

        ax1.fill_between(
            x_is,
            is_cdf_low,
            is_cdf_high,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

        ax1.plot(
            x_is,
            is_cdf_med,
            label=f"{method_name.capitalize()} NSE={is_median:.3f}",
            color=color,
            linewidth=2,
            zorder=3,
        )

        ax2.fill_between(
            x_oos,
            oos_cdf_low,
            oos_cdf_high,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

        ax2.plot(
            x_oos,
            oos_cdf_med,
            label=f"{method_name.capitalize()} NSE={oos_median:.3f}",
            color=color,
            linewidth=2,
            zorder=3,
        )

        ax1.plot([0, is_median], [0.5, 0.5], color=color, linestyle=":", linewidth=1.2)
        ax1.plot(
            [is_median, is_median], [0, 0.5], color=color, linestyle=":", linewidth=1.2
        )

        ax2.plot([0, oos_median], [0.5, 0.5], color=color, linestyle=":", linewidth=1.2)
        ax2.plot(
            [oos_median, oos_median],
            [0, 0.5],
            color=color,
            linestyle=":",
            linewidth=1.2,
        )

    ax1.legend(loc="upper left", frameon=False)
    ax2.legend(loc="upper left", frameon=False)

    ax1.set_title("In-Sample (IS) CDF")
    ax2.set_title("Out-of-Sample (OOS) CDF")

    ax1.set_xlabel("NSE")
    ax2.set_xlabel("NSE")
    ax1.set_ylabel("CDF")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    is_folder_path = "/Users/cylenlc//k_folds_result/IS/"
    oos_folder_path = "/Users/cylenlc//k_folds_result/OOS/"

    is_data = process_files(is_folder_path)
    oos_data = process_files(oos_folder_path)

    method_names = ["alpha", "camels"]

    plot_cdf_with_uncertainty(
        is_data, oos_data, method_names, n_boot=5000, sample_frac=0.8, random_state=42
    )
