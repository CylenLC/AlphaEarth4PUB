import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import colorsys


# =========================
# 1. 路径
# =========================
base_dir = Path("/Users/cylenlc/AlphaEarthViz")

shp_path = base_dir / "data/basin_set_full_res/HCDN_nhru_final_671.shp"

cec_path = Path(
    "/Users/cylenlc/Downloads/climatezones_shapefile/"
    "NA_ClimateZones/data/North_America_Climate_Zones.shp"
)

csv_files = [base_dir / "selected_basins_stratified.csv"] + [
    base_dir / f"selected_basins_stratified_v{i}.csv" for i in range(2, 9)
]


# =========================
# 2. ID 统一函数
# =========================
def normalize_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace(".0", "", regex=False)

    def _strip(x):
        return str(int(x)) if x.isdigit() else x.lstrip("0") or "0"

    return s.map(_strip)


# =========================
# 3. 读取并合并 8 个 CSV
# =========================
dfs = [pd.read_csv(f) for f in csv_files]
df_csv = pd.concat(dfs, ignore_index=True)

df_csv["gauge_id"] = normalize_id(df_csv["gauge_id"])

selected_info = (
    df_csv[["gauge_id", "koppen_code", "koppen_name", "zone"]]
    .drop_duplicates("gauge_id")
    .reset_index(drop=True)
)

print("Total rows in all CSVs:", len(df_csv))
print("Unique gauge_id in CSVs:", selected_info["gauge_id"].nunique())


# =========================
# 4. 读取 671 流域 shp 和 Köppen 气候区
# =========================
basins_raw = gpd.read_file(shp_path)
basins_raw["gauge_id"] = normalize_id(basins_raw["hru_id"])

cec = gpd.read_file(cec_path)

# 统一坐标系到 Köppen shp
basins_koppen = basins_raw.to_crs(cec.crs).copy()

# MultiPolygon 只保留最大 polygon
basins_koppen["geometry"] = basins_koppen["geometry"].apply(
    lambda g: max(g.geoms, key=lambda x: x.area) if g.geom_type == "MultiPolygon" else g
)

# 用质心匹配 Köppen 区域
basins_centroid = basins_koppen.copy()
basins_centroid["geometry"] = basins_centroid.geometry.centroid

joined = gpd.sjoin(
    basins_centroid[["gauge_id", "geometry"]],
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

df_koppen = (
    joined[["gauge_id", "Code", "Climate"]]
    .rename(
        columns={
            "Code": "koppen_code",
            "Climate": "koppen_name",
        }
    )
    .reset_index(drop=True)
)

df_koppen["zone"] = df_koppen["koppen_code"].map(zone_map).fillna("unknown")

df_koppen_valid = (
    df_koppen[df_koppen["zone"] != "unknown"]
    .drop_duplicates("gauge_id")
    .reset_index(drop=True)
)

print("\nAll 671 basins Köppen zone count:")
print(df_koppen_valid["zone"].value_counts())


# =========================
# 5. 如果不足 150，随机补足
# =========================
target_n = 150
current_n = selected_info["gauge_id"].nunique()
need_n = target_n - current_n

print(f"\nCurrent selected unique basins: {current_n}")
print(f"Need to add: {need_n}")

if need_n > 0:
    already_selected = set(selected_info["gauge_id"])

    candidates = df_koppen_valid[
        ~df_koppen_valid["gauge_id"].isin(already_selected)
    ].copy()

    # 排序后再抽样，保证不同环境下随机结果更稳定
    candidates = candidates.sort_values("gauge_id").reset_index(drop=True)

    print("Candidate basins available:", len(candidates))

    if len(candidates) < need_n:
        raise ValueError(
            f"候选流域数量不足：需要 {need_n} 个，但只有 {len(candidates)} 个。"
        )

    seed = 1004

    added_info = candidates.sample(n=need_n, random_state=seed)[
        ["gauge_id", "koppen_code", "koppen_name", "zone"]
    ].reset_index(drop=True)

    print("\nRandomly added basins:")
    print(added_info.to_string(index=False))

    selected_info_final = pd.concat(
        [selected_info, added_info],
        ignore_index=True,
    )

else:
    added_info = pd.DataFrame(
        columns=["gauge_id", "koppen_code", "koppen_name", "zone"]
    )

    selected_info_final = selected_info.copy()

    print("\nNo additional basins needed.")


selected_info_final = selected_info_final.drop_duplicates("gauge_id").reset_index(
    drop=True
)

print("\nFinal selected basins:", selected_info_final["gauge_id"].nunique())
print("\nFinal zone count:")
print(selected_info_final["zone"].value_counts())

# 保存补充的流域
added_path = base_dir / "added_basins_to_150.csv"
added_info.to_csv(added_path, index=False)

# 保存最终 150 个流域
final_path = base_dir / "selected_basins_150_final.csv"
selected_info_final.to_csv(final_path, index=False)

print("\nSaved added basins to:")
print(added_path)

print("\nSaved final 150 basins to:")
print(final_path)


# =========================
# 6. 准备画图坐标
# =========================
basins_raw["gauge_id"] = normalize_id(basins_raw["hru_id"])

if "lon_cen" in basins_raw.columns:
    lon_col = "lon_cen"
elif "lon" in basins_raw.columns:
    lon_col = "lon"
else:
    lon_col = None

if "lat_cen" in basins_raw.columns:
    lat_col = "lat_cen"
elif "lat" in basins_raw.columns:
    lat_col = "lat"
else:
    lat_col = None

if lon_col is not None and lat_col is not None:
    basin_coords = basins_raw[["gauge_id", lon_col, lat_col]].copy()
    basin_coords = basin_coords.rename(
        columns={
            lon_col: "lon",
            lat_col: "lat",
        }
    )
else:
    basins_wgs84 = basins_raw.to_crs(epsg=4326).copy()
    basins_wgs84["centroid"] = basins_wgs84.geometry.centroid
    basins_wgs84["lon"] = basins_wgs84["centroid"].x
    basins_wgs84["lat"] = basins_wgs84["centroid"].y

    basin_coords = basins_wgs84[["gauge_id", "lon", "lat"]].copy()

selected_points = pd.merge(
    basin_coords,
    selected_info_final,
    on="gauge_id",
    how="inner",
)

print("\nBasins found for plotting:", len(selected_points))

missing_for_plot = set(selected_info_final["gauge_id"]) - set(
    selected_points["gauge_id"]
)

if missing_for_plot:
    print("\nWarning: These selected basins were not found for plotting:")
    print(sorted(missing_for_plot))


# =========================
# 7. 绘图准备
# =========================
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10.5,
        "legend.title_fontsize": 12,
    }
)

proj = ccrs.LambertConformal()
extent = (-125, -66.5, 24, 49)

cec_plot = cec.to_crs(epsg=4326).copy()
cec_plot["zone"] = cec_plot["Code"].map(zone_map).fillna("unknown")

# 裁剪到美国本土范围
cec_plot = cec_plot.cx[extent[0] : extent[1], extent[2] : extent[3]]

# 大区边界
cec_zone_boundary = (
    cec_plot[cec_plot["zone"] != "unknown"].dissolve(by="zone").reset_index()
)

zone_order = ["arid", "semi_arid", "temperate", "continental", "alpine"]

zone_labels = {
    "temperate": "Temperate",
    "continental": "Continental",
    "semi_arid": "Semi-arid",
    "arid": "Arid",
    "alpine": "Alpine",
}


# =========================
# 8. 协调色系：左图和右图对应位置颜色大体相似
# =========================
# 左图 5 个 major climate zones 的基础颜色
zone_colors = {
    "arid": "#E64B35",  # red
    "semi_arid": "#F1C232",  # yellow
    "temperate": "#4DBBD5",  # blue
    "continental": "#00A087",  # green
    "alpine": "#8E44AD",  # purple
}


def make_shade(base_color, i, n):
    """
    根据 major zone 的基础颜色生成 subtype 的同色系深浅变化。
    这样右图 subtype 会和左图对应位置颜色大体一致。
    """
    rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)

    if n == 1:
        new_l = l
    else:
        # 控制同一大区内不同 subtype 的明暗变化
        # 数值范围越大，subtype 间差异越明显
        lightness_values = [0.34 + 0.42 * j / (n - 1) for j in range(n)]
        new_l = lightness_values[i]

    new_s = min(1.0, s * 1.15)

    return colorsys.hls_to_rgb(h, new_l, new_s)


# selected basins 中出现的 subtype，用于右图图例
koppen_codes = sorted(selected_points["koppen_code"].dropna().unique())

# 气候分区 shp 中，美国本土范围内出现的所有 subtype，用于右图背景蒙版
all_subtype_codes = sorted(
    cec_plot.loc[cec_plot["zone"] != "unknown", "Code"].dropna().unique()
)

print("\nSelected basin Köppen subtypes:")
print(koppen_codes)
print("Number of selected basin subtypes:", len(koppen_codes))

print("\nAll Köppen subtypes in plotted CONUS extent:")
print(all_subtype_codes)
print("Number of background subtypes:", len(all_subtype_codes))


# 为右图 subtype 生成颜色：
# 同一个 major zone 内的 subtype 使用同一基础色的不同深浅
subtype_mask_colors = {}

for zone in zone_order:
    zone_subtypes = sorted(
        [code for code in all_subtype_codes if zone_map.get(code, "unknown") == zone]
    )

    n_codes = len(zone_subtypes)

    for i, code in enumerate(zone_subtypes):
        subtype_mask_colors[code] = make_shade(
            base_color=zone_colors[zone],
            i=i,
            n=n_codes,
        )


# 点本身不使用颜色，但右图图例仍需要 subtype 色块
koppen_colors = {code: subtype_mask_colors.get(code, "gray") for code in koppen_codes}


# =========================
# 9. 绘图函数
# =========================
def add_climate_mask(ax, alpha=0.35, mask_level="zone"):
    """
    mask_level="zone":
        左图，5 个 major climate zones。

    mask_level="subtype":
        右图，所有 Köppen subtype。
    """

    if mask_level == "zone":
        for zone in zone_order:
            sub = cec_zone_boundary[cec_zone_boundary["zone"] == zone]

            if sub.empty:
                continue

            ax.add_geometries(
                sub.geometry,
                crs=ccrs.PlateCarree(),
                facecolor=zone_colors[zone],
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )

        # 左图保留 major zone 边界
        ax.add_geometries(
            cec_zone_boundary.geometry,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="dimgray",
            linewidth=0.45,
            alpha=0.55,
            zorder=3,
        )

    elif mask_level == "subtype":
        for code in all_subtype_codes:
            sub = cec_plot[cec_plot["Code"] == code]

            if sub.empty:
                continue

            ax.add_geometries(
                sub.geometry,
                crs=ccrs.PlateCarree(),
                facecolor=subtype_mask_colors[code],
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )

        # 右图只画 subtype 边界，不画 major zone 粗边界
        ax.add_geometries(
            cec_plot.geometry,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.55,
            zorder=2,
        )

    else:
        raise ValueError("mask_level must be either 'zone' or 'subtype'.")


def add_base_map(ax, with_climate_mask=True, mask_level="zone", mask_alpha=0.35):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if with_climate_mask:
        add_climate_mask(
            ax,
            alpha=mask_alpha,
            mask_level=mask_level,
        )

    ax.add_feature(
        cfeature.STATES,
        linewidth=0.45,
        edgecolor="gray",
        zorder=4,
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.9,
        zorder=4,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.7,
        zorder=4,
    )

    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_hollow_basin_points(ax, points_df, markersize=3.8):
    """
    绘制空心流域点：中间透明，边框黑色。
    """
    ax.plot(
        points_df["lon"].values,
        points_df["lat"].values,
        "o",
        markersize=markersize,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=0.4,
        alpha=1.0,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


# =========================
# 10. 左右子图
# =========================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7.5),
    subplot_kw={"projection": proj},
)


# -------------------------
# 左图：Major climate zones
# -------------------------
ax = axes[0]

add_base_map(
    ax,
    with_climate_mask=True,
    mask_level="zone",
    mask_alpha=0.35,
)

plot_hollow_basin_points(ax, selected_points, markersize=3.8)

# 左图图例：用色块表示 major climate zone
zone_counts = selected_points["zone"].value_counts().to_dict()

zone_handles = [
    mpatches.Patch(
        facecolor=zone_colors[zone],
        edgecolor="black",
        alpha=0.75,
        label=f"{zone_labels[zone]} (n={zone_counts.get(zone, 0)})",
    )
    for zone in zone_order
    if zone_counts.get(zone, 0) > 0
]

legend_a = ax.legend(
    handles=zone_handles,
    loc="lower left",
    bbox_to_anchor=(-0.18, 0.02),
    borderaxespad=0.0,
    fontsize=10.5,
    title="Major climate zone",
    title_fontsize=12,
    frameon=True,
    ncol=1,
)

legend_a.get_frame().set_edgecolor("black")
legend_a.get_frame().set_linewidth(0.6)

ax.text(
    0.5,
    -0.08,
    "(a) Major climate zones",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=15,
    fontname="Times New Roman",
)


# -------------------------
# 右图：Köppen climate subtypes
# -------------------------
ax = axes[1]

add_base_map(
    ax,
    with_climate_mask=True,
    mask_level="subtype",
    mask_alpha=0.45,
)

plot_hollow_basin_points(ax, selected_points, markersize=3.8)

# 右图图例：用色块表示 selected basins 中出现的 subtype
koppen_counts = selected_points["koppen_code"].value_counts().to_dict()

koppen_handles = [
    mpatches.Patch(
        facecolor=koppen_colors[code],
        edgecolor="black",
        alpha=0.75,
        label=f"{code} (n={koppen_counts.get(code, 0)})",
    )
    for code in koppen_codes
]

legend_b = ax.legend(
    handles=koppen_handles,
    loc="lower left",
    bbox_to_anchor=(-0.28, 0.02),
    borderaxespad=0.0,
    fontsize=9.5,
    title="Köppen subtype",
    title_fontsize=11.5,
    frameon=True,
    ncol=2,
)

legend_b.get_frame().set_edgecolor("black")
legend_b.get_frame().set_linewidth(0.6)

ax.text(
    0.5,
    -0.08,
    "(b) Köppen climate subtypes",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=15,
    fontname="Times New Roman",
)


# =========================
# 11. 总标题与保存
# =========================
plt.suptitle(
    "Spatial Distribution of Selected Basins Across Major and Köppen Climate Zones",
    fontsize=15,
    y=0.98,
)

plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])

combined_png_path = (
    base_dir / "selected_basins_150_coordinated_zone_and_subtype_colors.png"
)

combined_pdf_path = (
    base_dir / "selected_basins_150_coordinated_zone_and_subtype_colors.pdf"
)

plt.savefig(combined_png_path, dpi=600, bbox_inches="tight")
plt.savefig(combined_pdf_path, bbox_inches="tight")
plt.show()

print("\nSaved combined map:")
print(combined_png_path)
print(combined_pdf_path)
