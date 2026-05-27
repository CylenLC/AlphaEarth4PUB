import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from scipy.stats import chi2_contingency


# ── 工具函数 ───────────────────────────────────────────────
def normalize_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace(".0", "", regex=False)

    def _strip(x):
        return str(int(x)) if x.isdigit() else x.lstrip("0") or "0"

    return s.map(_strip)


# ══════════════════════════════════════════════════════════
# 1. 读取 Köppen 标签（同时保留大区和细分类型）
# ══════════════════════════════════════════════════════════
cec = gpd.read_file(
    "/Users/cylenlc/Downloads/climatezones_shapefile/NA_ClimateZones/data/North_America_Climate_Zones.shp"
)
basins = gpd.read_file("data/basin_set_full_res/HCDN_nhru_final_671.shp")
basins = basins.to_crs(cec.crs)

basins["geometry"] = basins["geometry"].apply(
    lambda g: max(g.geoms, key=lambda x: x.area) if g.geom_type == "MultiPolygon" else g
)

basins_centroid = basins.copy()
basins_centroid["geometry"] = basins_centroid["geometry"].centroid

joined = gpd.sjoin(
    basins_centroid[["hru_id", "geometry"]],
    cec[["Code", "Climate", "geometry"]],
    how="left",
    predicate="within",
)

zone_map = {
    "Af": "temperate",
    "Am": "temperate",
    "Aw": "temperate",
    "BWh": "arid",
    "BWk": "arid",
    "BSh": "semi_arid",
    "BSk": "semi_arid",
    "Csa": "temperate",
    "Csb": "temperate",
    "Csc": "temperate",
    "Cwa": "temperate",
    "Cwb": "temperate",
    "Cwc": "temperate",
    "Cfa": "temperate",
    "Cfb": "temperate",
    "Cfc": "temperate",
    "Dsa": "continental",
    "Dsb": "continental",
    "Dsc": "continental",
    "Dsd": "continental",
    "Dwa": "continental",
    "Dwb": "continental",
    "Dwc": "continental",
    "Dfa": "continental",
    "Dfb": "continental",
    "Dfc": "continental",
    "Dfd": "continental",
    "ET": "alpine",
    "EF": "alpine",
}

joined["zone"] = joined["Code"].map(zone_map).fillna("unknown")
joined["gauge_id"] = normalize_id(joined["hru_id"])

df_koppen = (
    joined[["gauge_id", "Code", "zone"]]
    .rename(columns={"Code": "koppen_code"})
    .reset_index(drop=True)
)

# 过滤unknown
df_valid = df_koppen[df_koppen["zone"] != "unknown"].copy()

# 统计
print("Köppen 5大区分布：")
print(df_valid["zone"].value_counts())
print("\nKöppen 细分类型分布：")
print(df_valid["koppen_code"].value_counts())

n_zones = df_valid["zone"].nunique()
n_subtypes = df_valid["koppen_code"].nunique()
print(f"\n5大区数量 K = {n_zones}")
print(f"细分类型数量 K = {n_subtypes}")


# ══════════════════════════════════════════════════════════
# 2. 读取属性CSV并做KMeans聚类（两种K值）
# ══════════════════════════════════════════════════════════
csv_files = {
    "CAMELS Attr.": "data/attributes/basin_attributes_method_1.csv",
    "CAMELS Emb.": "data/attributes/BasinEmbeddings_2017_2024_avg_method_2.csv",
    "AlphaEarth": "data/attributes/fc_static_output_method_3.csv",
}


def run_kmeans(csv_path, k, random_state=42):
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    df[id_col] = normalize_id(df[id_col])
    attr_cols = [c for c in df.columns if c != id_col]
    X = df[attr_cols].values
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    result = df[[id_col]].copy()
    result.columns = ["gauge_id"]
    result["cluster"] = labels
    return result


# 分别用K=5（大区）和K=细分类型数运行聚类
results_zone = {}  # K = 5大区
results_subtype = {}  # K = 细分类型数

for method, path in csv_files.items():
    results_zone[method] = run_kmeans(path, k=n_zones)
    results_subtype[method] = run_kmeans(path, k=n_subtypes)
    print(f"{method} 聚类完成（K={n_zones} 和 K={n_subtypes}）")


# ══════════════════════════════════════════════════════════
# 3. 定量评估函数
# ══════════════════════════════════════════════════════════
def evaluate(df_valid, label_col, results_dict, level_name):
    print(f"\n{'='*60}")
    print(f"定量评估：{level_name}")
    print(f"{'='*60}")
    rows = []
    for method, df_result in results_dict.items():
        merged = pd.merge(df_valid, df_result, on="gauge_id", how="inner")
        if merged.empty:
            print(f"{method}: ID匹配失败")
            continue
        true_labels = merged[label_col].values
        cluster_labels = merged["cluster"].values
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        ct = pd.crosstab(merged[label_col], merged["cluster"])
        chi2, p_val, dof, _ = chi2_contingency(ct)
        print(f"\n【{method}】")
        print(f"  ARI : {ari:.4f}")
        print(f"  NMI : {nmi:.4f}")
        print(f"  Chi²: {chi2:.2f}, p={p_val:.2e}, dof={dof}")
        rows.append(
            {"Method": method, "ARI": ari, "NMI": nmi, "Chi2": chi2, "p": p_val}
        )
    return pd.DataFrame(rows)


df_eval_zone = evaluate(df_valid, "zone", results_zone, "5大区（K=5）")
df_eval_subtype = evaluate(
    df_valid, "koppen_code", results_subtype, f"细分类型（K={n_subtypes}）"
)


# ══════════════════════════════════════════════════════════
# 4. ARI / NMI 对比柱状图（大区 vs 细分）
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

metrics = ["ARI", "NMI"]
titles = ["Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)"]
x = np.arange(len(csv_files))
width = 0.35
methods = list(csv_files.keys())

for ax, metric, title in zip(axes, metrics, titles):
    vals_zone = df_eval_zone[metric].values
    vals_subtype = df_eval_subtype[metric].values

    bars1 = ax.bar(
        x - width / 2,
        vals_zone,
        width,
        label=f"5 Major Zones (K={n_zones})",
        color="#4C72B0",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        vals_subtype,
        width,
        label=f"Subtypes (K={n_subtypes})",
        color="#DD8452",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, max(max(vals_zone), max(vals_subtype)) * 1.3)
    ax.legend(fontsize=10)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Cluster Alignment with Köppen Zones vs. Subtypes", fontsize=14)
plt.tight_layout()
plt.savefig("results/ari_nmi_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("已保存：results/ari_nmi_comparison.png")


# ══════════════════════════════════════════════════════════
# 5. 热力图（大区 和 细分类型各一组）
# ══════════════════════════════════════════════════════════
def plot_heatmap(
    results_dict, df_valid, label_col, title_suffix, fname, figsize=(20, 6)
):
    zone_order = sorted(df_valid[label_col].unique())
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (method, df_result) in zip(axes, results_dict.items()):
        merged = pd.merge(df_valid, df_result, on="gauge_id", how="inner")
        if merged.empty:
            ax.set_title(f"{method}\n(ID匹配失败)")
            continue
        ct = pd.crosstab(merged[label_col], merged["cluster"])
        ct = ct.reindex(zone_order, fill_value=0)
        ct_norm = ct.div(ct.sum(axis=1), axis=0)

        sns.heatmap(
            ct_norm,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Proportion"},
        )
        ax.set_title(method, fontsize=13)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Köppen Zone", fontsize=11)
        ax.tick_params(axis="both", rotation=0, labelsize=8)

    plt.suptitle(
        f"Cluster vs. Köppen {title_suffix} (Row-normalized)", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"已保存：{fname}")


# 大区热力图
plot_heatmap(
    results_zone,
    df_valid,
    "zone",
    "5 Major Zones",
    "results/heatmap_major_zones.png",
    figsize=(20, 6),
)

# 细分类型热力图（行数多，图高一些）
plot_heatmap(
    results_subtype,
    df_valid,
    "koppen_code",
    "Subtypes",
    "results/heatmap_subtypes.png",
    figsize=(24, 14),
)


# ══════════════════════════════════════════════════════════
# 6. 地图对比（大区 和 细分类型）
# ══════════════════════════════════════════════════════════
lon_col = "lon_cen" if "lon_cen" in basins.columns else "lon"
lat_col = "lat_cen" if "lat_cen" in basins.columns else "lat"
basins["gauge_id"] = normalize_id(basins["hru_id"])
df_coords = basins[["gauge_id", lon_col, lat_col]].copy()

proj = ccrs.LambertConformal()
extent = (-125, -66.5, 24, 49)

zone_colors = {
    "temperate": "#2196F3",
    "continental": "#FF9800",
    "semi_arid": "#FFEB3B",
    "arid": "#F44336",
    "alpine": "#9C27B0",
}


def plot_map_comparison(
    results_dict, df_valid, label_col, label_colors, title_suffix, fname_prefix
):
    for method, df_result in results_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(22, 8), subplot_kw={"projection": proj})

        # 左图数据（Köppen标签）
        data_left = pd.merge(df_coords, df_valid, on="gauge_id", how="inner")
        # 右图数据（聚类结果）
        data_right = pd.merge(df_coords, df_result, on="gauge_id", how="inner")

        # 左图颜色：若传入None则自动生成 ← 修复关键
        left_vals = sorted(data_left[label_col].unique())
        if label_colors is not None:
            left_color_map = label_colors
        else:
            cmap_left = plt.get_cmap("tab20")
            left_color_map = {v: cmap_left(i % 20) for i, v in enumerate(left_vals)}

        # 右图颜色：单独为cluster数字生成颜色
        cluster_vals = sorted(data_right["cluster"].unique())
        cmap_tab = plt.get_cmap("tab10")
        right_color_map = {v: cmap_tab(i % 10) for i, v in enumerate(cluster_vals)}

        for ax, (data, title_, col, color_map) in zip(
            axes,
            [
                (data_left, f"Köppen {title_suffix}", label_col, left_color_map),
                (data_right, f"KMeans — {method}", "cluster", right_color_map),
            ],
        ):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linewidth=1)
            ax.set_facecolor("none")
            for spine in ax.spines.values():
                spine.set_visible(False)

            for val in sorted(data[col].unique()):
                sub = data[data[col] == val]
                ax.plot(
                    sub[lon_col].values,
                    sub[lat_col].values,
                    "o",
                    markersize=5,
                    color=color_map.get(val, "gray"),
                    transform=ccrs.PlateCarree(),
                    label=str(val),
                )

            ax.legend(loc="lower left", fontsize=7, frameon=True, title=col, ncol=2)
            ax.set_title(title_, fontsize=12)

        plt.suptitle(f"Köppen {title_suffix} vs. KMeans ({method})", fontsize=13)
        plt.tight_layout()
        safe_name = method.replace(" ", "_").replace(".", "")
        fname = f"results/{fname_prefix}_{safe_name}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"已保存：{fname}")


# 大区地图
plot_map_comparison(
    results_zone, df_valid, "zone", zone_colors, "5 Major Zones", "map_major"
)

# 细分类型地图
plot_map_comparison(
    results_subtype, df_valid, "koppen_code", None, "Subtypes", "map_subtype"
)

print("\n全部分析完成。")
