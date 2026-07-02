import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats


# ══════════════════════════════════════════════════════════
# 1. 读取数据
# ══════════════════════════════════════════════════════════
df = pd.read_csv("CAMELS_NDVI_stability.csv")
df["hru_id"] = df["hru_id"].astype(str).str.strip()

print("数据概览：")
print(df[["NDVI_early_mean", "NDVI_recent_mean", "delta_NDVI_mean"]].describe())


# ══════════════════════════════════════════════════════════
# 2. 基本统计分析
# ══════════════════════════════════════════════════════════
mean_delta = df["delta_NDVI_mean"].mean()
std_delta = df["delta_NDVI_mean"].std()
pct_stable = (df["delta_NDVI_mean"].abs() < 0.05).sum() / len(df) * 100
pct_increase = (df["delta_NDVI_mean"] > 0.05).sum() / len(df) * 100
pct_decrease = (df["delta_NDVI_mean"] < -0.05).sum() / len(df) * 100

print(f"\nδNDVI统计：")
print(f"  均值：{mean_delta:.4f}")
print(f"  标准差：{std_delta:.4f}")
print(f"  变化幅度 < 0.05（稳定）：{pct_stable:.1f}%")
print(f"  变化幅度 > 0.05（增加）：{pct_increase:.1f}%")
print(f"  变化幅度 < -0.05（减少）：{pct_decrease:.1f}%")

# t检验：delta NDVI是否显著不为零
t_stat, p_val = stats.ttest_1samp(df["delta_NDVI_mean"].dropna(), 0)
print(f"\n单样本t检验（H0: δNDVI = 0）：")
print(f"  t = {t_stat:.4f}, p = {p_val:.4e}")


# ══════════════════════════════════════════════════════════
# 3. 按变化程度分组
# ══════════════════════════════════════════════════════════
df["change_group"] = pd.cut(
    df["delta_NDVI_mean"],
    bins=[-np.inf, -0.1, -0.05, 0.05, 0.1, np.inf],
    labels=[
        "Large decrease\n(<-0.10)",
        "Moderate decrease\n(-0.10 to -0.05)",
        "Stable\n(-0.05 to 0.05)",
        "Moderate increase\n(0.05 to 0.10)",
        "Large increase\n(>0.10)",
    ],
)

print("\n各变化组流域数量：")
print(df["change_group"].value_counts().sort_index())


# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════

# ── 图1：delta NDVI直方图 ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(
    df["delta_NDVI_mean"],
    bins=40,
    color="#4C72B0",
    edgecolor="white",
    linewidth=0.5,
    alpha=0.85,
)
ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="No change")
ax.axvline(x=0.05, color="red", linestyle=":", linewidth=1.5, label="±0.05 threshold")
ax.axvline(x=-0.05, color="red", linestyle=":", linewidth=1.5)
ax.axvline(
    x=mean_delta,
    color="orange",
    linestyle="-",
    linewidth=1.5,
    label=f"Mean = {mean_delta:.3f}",
)
ax.set_xlabel("δNDVI (Recent - Early)", fontsize=12)
ax.set_ylabel("Number of Basins", fontsize=12)
ax.set_title("Distribution of NDVI Change\nacross CAMELS-US Basins", fontsize=13)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── 图2：早期 vs 近期 NDVI散点图 ──────────────────────
ax = axes[1]
sc = ax.scatter(
    df["NDVI_early_mean"],
    df["NDVI_recent_mean"],
    c=df["delta_NDVI_mean"],
    cmap="RdYlGn",
    vmin=-0.2,
    vmax=0.2,
    s=20,
    alpha=0.7,
)
lims = [
    min(df["NDVI_early_mean"].min(), df["NDVI_recent_mean"].min()) - 0.05,
    max(df["NDVI_early_mean"].max(), df["NDVI_recent_mean"].max()) + 0.05,
]
ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1 line")
ax.set_xlabel("NDVI (1984–1990)", fontsize=12)
ax.set_ylabel("NDVI (2017–2024)", fontsize=12)
ax.set_title("Early vs. Recent NDVI\nfor CAMELS-US Basins", fontsize=13)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("δNDVI", fontsize=11)

plt.tight_layout()
plt.savefig("results/ndvi_change_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
print("已保存：results/ndvi_change_distribution.png")


# ── 图3：空间分布地图 ──────────────────────────────────
basins_shp = gpd.read_file("data/basin_set_full_res/HCDN_nhru_final_671.shp")
basins_shp["hru_id"] = basins_shp["hru_id"].astype(str).str.strip()

lon_col = "lon_cen" if "lon_cen" in basins_shp.columns else "lon"
lat_col = "lat_cen" if "lat_cen" in basins_shp.columns else "lat"

df_map = pd.merge(
    basins_shp[["hru_id", lon_col, lat_col]],
    df[["hru_id", "delta_NDVI_mean"]],
    on="hru_id",
    how="inner",
)

proj = ccrs.LambertConformal()
extent = (-125, -66.5, 24, 49)

fig = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=proj)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.set_facecolor("none")
for spine in ax.spines.values():
    spine.set_visible(False)

norm = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
sc = ax.scatter(
    df_map[lon_col].values,
    df_map[lat_col].values,
    c=df_map["delta_NDVI_mean"].values,
    cmap="RdYlGn",
    norm=norm,
    s=20,
    transform=ccrs.PlateCarree(),
    zorder=5,
)
cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("δNDVI (2017–2024 minus 1984–1990)", fontsize=11)
plt.title("Spatial Distribution of NDVI Change across CAMELS-US Basins", fontsize=13)
plt.savefig("results/ndvi_change_map.png", dpi=300, bbox_inches="tight")
plt.show()
print("已保存：results/ndvi_change_map.png")


# ══════════════════════════════════════════════════════════
# 5. 按变化程度分组后的模型性能对比
# （需要读取模型性能结果文件）
# ══════════════════════════════════════════════════════════
# 读取模型性能结果（替换为你的实际文件路径）
try:
    df_perf = pd.read_csv("results/model_performance.csv")
    df_perf["hru_id"] = df_perf["hru_id"].astype(str).str.strip()

    df_merged = pd.merge(df, df_perf, on="hru_id", how="inner")

    # 按变化程度分为高变化组和低变化组
    threshold = 0.05
    df_merged["stability"] = np.where(
        df_merged["delta_NDVI_mean"].abs() < threshold, "Stable", "Changed"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["NSE", "KGE"]):
        if metric not in df_merged.columns:
            continue
        groups = [
            df_merged[df_merged["stability"] == g][metric].dropna()
            for g in ["Stable", "Changed"]
        ]
        ax.boxplot(
            groups,
            labels=["Stable\n(|δNDVI| < 0.05)", "Changed\n(|δNDVI| ≥ 0.05)"],
            patch_artist=True,
            boxprops=dict(facecolor="#4C72B0", alpha=0.7),
            medianprops=dict(color="black", linewidth=2),
        )

        t_stat, p_val = stats.mannwhitneyu(
            groups[0], groups[1], alternative="two-sided"
        )
        ax.set_title(
            f"{metric} by Basin Stability\n(Mann-Whitney U, p={p_val:.3f})", fontsize=12
        )
        ax.set_ylabel(metric, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/performance_by_stability.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("已保存：results/performance_by_stability.png")

except FileNotFoundError:
    print("未找到模型性能文件，跳过分组性能对比分析")


# ══════════════════════════════════════════════════════════
# 6. 保存分组结果
# ══════════════════════════════════════════════════════════
df[
    ["hru_id", "NDVI_early_mean", "NDVI_recent_mean", "delta_NDVI_mean", "change_group"]
].to_csv("results/ndvi_stability_results.csv", index=False)
print("\n已保存：results/ndvi_stability_results.csv")
print("\n分析完成。")
