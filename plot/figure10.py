import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def _normalize_id_series(s: pd.Series) -> pd.Series:
    """
    统一ID格式：去空白、去'.0'、去前导0（等价于 int -> str）。
    若包含非数字字符，则仅去前导0。
    """
    s = s.astype(str).str.strip().str.replace(".0", "", regex=False)

    def _strip(x):
        return str(int(x)) if x.isdigit() else x.lstrip("0") or "0"

    return s.map(_strip)


def _auto_choose_k(X: np.ndarray, k_min=4, k_max=12, random_state=42) -> int:
    """
    用 silhouette score 自动选 K；若样本很小可调低 k_max。
    """
    best_k, best_score = None, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k if best_k is not None else max(2, k_min)


def cluster_and_plot_attributes_on_map(
    shp_file: str,
    attr_csv: str,
    id_col_shp: str = "hru_id",
    id_col_attr: str | None = None,  # 默认为 attr_csv 第一列
    selected_attr_cols: list[str] | None = None,  # 指定要用的属性列；默认用除ID外所有列
    n_clusters: int | None = None,  # None 时自动选K
    k_min: int = 4,
    k_max: int = 12,
    extent: tuple = (-125, -66.5, 24, 49),  # USA范围 (minlon,maxlon,minlat,maxlat)
    save_path: str | None = None,
    random_state: int = 42,
    restandardize: bool = False,  # 你的属性已标准化；若想再做一次标准化，可置 True
):
    """
    读取 17维属性 CSV（第一列为流域ID，后面为属性列），对属性向量做 KMeans 聚类，
    并将结果绘在地图上：同一类同色点。

    返回：包含 cluster 标签的数据表（便于后续分析/导出）。
    """
    # ===== 读 shapefile（拿经纬度） =====
    basins = gpd.read_file(shp_file)
    if id_col_shp not in basins.columns:
        raise ValueError(
            f"'{id_col_shp}' not in shapefile columns: {basins.columns.tolist()}"
        )
    # 试探中心点字段名
    lon_col = (
        "lon_cen"
        if "lon_cen" in basins.columns
        else ("lon" if "lon" in basins.columns else None)
    )
    lat_col = (
        "lat_cen"
        if "lat_cen" in basins.columns
        else ("lat" if "lat" in basins.columns else None)
    )
    if lon_col is None or lat_col is None:
        raise ValueError(
            "未在 shapefile 中找到中心点经纬度列（尝试了 lon_cen/lat_cen 或 lon/lat）。"
        )

    basins[id_col_shp] = _normalize_id_series(basins[id_col_shp].copy())
    basins = basins[[id_col_shp, lon_col, lat_col]].dropna().copy()

    # ===== 读 属性 CSV =====
    df_attr = pd.read_csv(attr_csv)
    if id_col_attr is None:
        id_col_attr = df_attr.columns[0]
    df_attr[id_col_attr] = _normalize_id_series(df_attr[id_col_attr].copy())

    # 选择属性列
    if selected_attr_cols is None:
        attr_cols = [c for c in df_attr.columns if c != id_col_attr]
    else:
        missing = [c for c in selected_attr_cols if c not in df_attr.columns]
        if missing:
            raise ValueError(f"属性列不存在: {missing}")
        attr_cols = selected_attr_cols

    if len(attr_cols) == 0:
        raise ValueError("未找到可用的属性列。")

    # ===== 对齐ID并构造聚类输入 =====
    merged = pd.merge(
        basins,
        df_attr[[id_col_attr] + attr_cols],
        left_on=id_col_shp,
        right_on=id_col_attr,
        how="inner",
    )
    if merged.empty:
        raise ValueError("基于流域ID内连接后为空，请检查ID是否一致（前导0问题）。")

    X = merged[attr_cols].values

    # 你的属性已标准化；这里通常不需要再标准化。
    # 如需稳健，可将 restandardize=True，再做一次标准化。
    if restandardize:
        X = StandardScaler().fit_transform(X)

    # ===== 选择K并聚类 =====
    if n_clusters is None:
        n_clusters = _auto_choose_k(
            X, k_min=k_min, k_max=k_max, random_state=random_state
        )
        print(f"[Info] 自动选择的 K = {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    merged["cluster"] = labels

    # ===== 绘图 =====
    proj = ccrs.LambertConformal()
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("none")

    # 颜色盘（tab20 支持最多20类；更多则自动采样 hsv）
    if n_clusters <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n_clusters)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]

    legend_handles = []
    for k in range(n_clusters):
        sub = merged[merged["cluster"] == k]
        ax.plot(
            sub[lon_col].values,
            sub[lat_col].values,
            "o",
            markersize=8,
            linestyle="None",
            color=colors[k],
            transform=ccrs.PlateCarree(),
        )
        legend_handles.append(
            mpatches.Patch(color=colors[k], label=f"Cluster {k} (n={len(sub)})")
        )

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(n_clusters, 6),
        fontsize=9,
        frameon=False,
    )
    plt.title(
        f"Unsupervised Clusters from {len(attr_cols)}-D Attributes (K={n_clusters})",
        fontsize=14,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"[Saved] {save_path}")
    plt.show()

    # 返回含 cluster 的数据；便于后续与嵌入聚类对比/统计
    return merged[[id_col_shp, lon_col, lat_col, "cluster"] + attr_cols]


shp_file = "data/basin_set_full_res/HCDN_nhru_final_671.shp"
# attr_csv  = "data/向量csv/basin_attributes_method_1.csv"  # 第一列 basin_id，其余17列为属性（已标准化）
attr_csv = "data/向量csv/fc_static_output_method_3.csv"
df_attr_labels = cluster_and_plot_attributes_on_map(
    shp_file=shp_file,
    attr_csv=attr_csv,
    n_clusters=None,  # 自动选K（默认4~12）
    k_min=4,
    k_max=12,
    save_path="results/attr_clusters_map.png",
    restandardize=False,  # 你的数据已标准化，保持 False
)
# df_attr_labels 里包含：hru_id, lon_cen/lat_cen, cluster，以及17个属性列
