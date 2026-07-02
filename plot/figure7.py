"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. 读取数据 ────────────────────────────────────────────
attr = pd.read_csv('/Users/cylenlc/AlphaEarthViz/data/向量csv/basin_attributes_method_1.csv', index_col=0)
emb  = pd.read_csv('/Users/cylenlc/AlphaEarthViz/data/向量csv/BasinEmbeddings_2017_2024_avg_method_2.csv', index_col=0)
aef  = pd.read_csv('/Users/cylenlc/AlphaEarthViz/data/向量csv/fc_static_output_method_3.csv', index_col=0)

# 统一index格式，去除前导零以保证匹配
attr.index = attr.index.astype(str).str.lstrip('0')
emb.index  = emb.index.astype(str).str.lstrip('0')
aef.index  = aef.index.astype(str).str.lstrip('0')

# ── 2. 指定目标流域 ────────────────────────────────────────
target_id = '1013500'  # 01013500去除前导零

# ── 3. 计算余弦相似度 ──────────────────────────────────────
def compute_similarity(df, target_id):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(X, index=df.index, columns=df.columns)

    target_vec = df_scaled.loc[target_id].values.reshape(1, -1)
    all_vecs   = df_scaled.values
    sim        = cosine_similarity(target_vec, all_vecs)[0]

    return pd.Series(sim, index=df.index)

sim_attr = compute_similarity(attr, target_id)
sim_emb  = compute_similarity(emb,  target_id)
sim_aef  = compute_similarity(aef,  target_id)

# ── 4. 按gauge_id数字顺序排列 ──────────────────────────────
order = sorted(sim_attr.index, key=lambda x: int(x))

sim_attr = sim_attr[order]
sim_emb  = sim_emb[order]
sim_aef  = sim_aef[order]

# ── 5. 绘制条纹图 ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 5))

datasets = [
    (sim_attr, 'CAMELS Attr.'),
    (sim_emb,  'CAMELS Emb.'),
    (sim_aef,  'AlphaEarth'),
]

for ax, (sim, title) in zip(axes, datasets):
    data = sim.values.reshape(1, -1)
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='none')
    ax.set_title(title, fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])

# colorbar
cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Similarity', fontsize=11)
cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])

plt.tight_layout(rect=[0, 0.06, 1, 1])
#plt.savefig('stripe_plot_01013500.png', dpi=300, bbox_inches='tight')
plt.show()
#print("已保存图片：stripe_plot_01013500.png")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

id1 = '1013500'
id2 = '1031500'

def pair_similarity(df, name):

    # ===== 原始 cosine =====
    v1 = df.loc[id1].values.reshape(1, -1)
    v2 = df.loc[id2].values.reshape(1, -1)

    raw_sim = cosine_similarity(v1, v2)[0,0]

    # ===== z-score 后 cosine =====
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    df_scaled = pd.DataFrame(
        X,
        index=df.index,
        columns=df.columns
    )

    v1s = df_scaled.loc[id1].values.reshape(1, -1)
    v2s = df_scaled.loc[id2].values.reshape(1, -1)

    scaled_sim = cosine_similarity(v1s, v2s)[0,0]

    print(f'\n{name}')
    print(f'Raw cosine      : {raw_sim:.6f}')
    print(f'Scaled cosine   : {scaled_sim:.6f}')

pair_similarity(attr, 'Method 1 (CAMELS Attr)')
pair_similarity(emb,  'Method 2 (CAMELS Emb)')"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================
# 字体设置
# ============================
rcParams["font.family"] = "Times New Roman"
rcParams["axes.unicode_minus"] = False

# ============================
# 文件路径
# ============================
file1 = r"data/相似度矩阵/basin_cosine_similarity_method_1.csv"
file2 = r"data/相似度矩阵/basin_cosine_similarity_method_2.csv"
file3 = r"data/相似度矩阵/basin_cosine_similarity_method_3.csv"
shp_file = r"data/basin_set_full_res/HCDN_nhru_final_671.shp"

# ============================
# 读取数据
# ============================
df1 = pd.read_csv(file1, index_col=0)
df2 = pd.read_csv(file2, index_col=0)
df3 = pd.read_csv(file3, index_col=0)

basins = gpd.read_file(shp_file)


# ============================
# 统一 basin_id 格式
# ============================
def normalize_basin_ids(df):
    df.index = df.index.astype(str).str.zfill(8)
    df.columns = df.columns.astype(str).str.zfill(8)
    return df


df1 = normalize_basin_ids(df1)
df2 = normalize_basin_ids(df2)
df3 = normalize_basin_ids(df3)

basins["hru_id"] = basins["hru_id"].astype(str).str.zfill(8)

# ============================
# CRS 与流域质心点
# ============================
basins = basins.to_crs(epsg=4326)

# 用投影坐标算质心，避免警告
basins_proj = basins.to_crs(epsg=5070)

basin_points = basins_proj.copy()
basin_points["geometry"] = basin_points.geometry.centroid
basin_points = basin_points.to_crs(epsg=4326)

# ============================
# 美国本土范围
# ============================
xmin, ymin, xmax, ymax = -125, 24, -66, 50

# ============================
# 配色：与条纹图一致
# ============================
cmap = mcolors.LinearSegmentedColormap.from_list("blue_red", ["blue", "white", "red"])


def plot_base_map(ax):
    """
    白色背景 + 美国州界
    """
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())

    ax.set_facecolor("white")

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)

    ax.add_feature(cfeature.STATES, linewidth=0.45, edgecolor="#cfcfcf", zorder=1)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.55, edgecolor="#bdbdbd", zorder=1)

    ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#bdbdbd", zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_similarity_point_map(
    basin_points,
    similarity_df,
    basin_id,
    ax,
    title,
    vmin=-1,
    vmax=1,
):
    """
    绘制单个方法的流域点相似度空间分布图
    """
    basin_id = str(basin_id).zfill(8)

    if basin_id not in similarity_df.index:
        raise ValueError(f"{basin_id} not found in similarity matrix.")

    sim_values = similarity_df.loc[basin_id]

    sim_df = pd.DataFrame(
        {
            "hru_id": sim_values.index.astype(str).str.zfill(8),
            "similarity": sim_values.values,
        }
    )

    plot_gdf = basin_points.merge(sim_df, on="hru_id", how="left")

    # 底图
    plot_base_map(ax)

    # 流域点
    plot_gdf.plot(
        ax=ax,
        column="similarity",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        markersize=22,
        edgecolor="none",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=3,
        missing_kwds={"color": "lightgrey"},
    )

    # 高亮 target basin
    target = plot_gdf[plot_gdf["hru_id"] == basin_id]

    if not target.empty:
        target.plot(
            ax=ax,
            color="yellow",
            edgecolor="black",
            markersize=140,
            marker="*",
            linewidth=0.8,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    else:
        print(f"Warning: target basin {basin_id} not found.")

    ax.set_title(title, fontsize=16)


def plot_similarity_spatial_comparison_points(
    basin_points,
    df1,
    df2,
    df3,
    basin_id,
    save_path=None,
):
    """
    三种方法空间相似度点图对比
    """

    basin_id = str(basin_id).zfill(8)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # ============================
    # 三种方法
    # ============================
    plot_similarity_point_map(
        basin_points,
        df1,
        basin_id,
        axes[0],
        "Camels Attr.",
    )

    plot_similarity_point_map(
        basin_points,
        df3,
        basin_id,
        axes[1],
        "Camels Emb.",
    )

    plot_similarity_point_map(
        basin_points,
        df2,
        basin_id,
        axes[2],
        "AlphaEarth",
    )

    # ============================
    # Colorbar
    # ============================
    norm = plt.Normalize(vmin=-1, vmax=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])

    # 手动指定 colorbar 位置
    cbar_ax = fig.add_axes([0.18, 0.15, 0.64, 0.035])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    cbar.set_label("Similarity", fontsize=16)

    cbar.ax.tick_params(labelsize=13)

    # ============================
    # 标题
    # ============================
    fig.suptitle(
        f"Spatial Distribution of Basin Similarity: Target Basin {basin_id}",
        fontsize=18,
        y=0.98,
    )

    # ============================
    # 布局
    # ============================
    plt.subplots_adjust(
        left=0.03,
        right=0.98,
        top=0.88,
        bottom=0.20,
        wspace=0.08,
    )

    # ============================
    # 保存
    # ============================
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()


# ============================
# 示例绘图
# ============================
target_basin = df1.index[0]

plot_similarity_spatial_comparison_points(
    basin_points,
    df1,
    df2,
    df3,
    target_basin,
    save_path="figures/similarity_spatial_points.png",
)
