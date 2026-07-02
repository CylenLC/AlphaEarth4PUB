# ====== cluster_result_analysis.py ======
import os, warnings
from typing import List, Optional, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr, f_oneway, kruskal, pearsonr, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def normalize_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace(".0", "", regex=False)

    def _strip(x):
        return str(int(x)) if x.isdigit() else (x.lstrip("0") or "0")

    return s.map(_strip)


def fdr_bh(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros_like(p, dtype=bool)
    cutoff = ranked[np.where(passed)[0].max()]
    return p <= cutoff


def correlation_ratio(categories, values) -> float:
    categories = pd.Series(categories).astype("category")
    cat = categories.cat.codes.values
    y = pd.to_numeric(values, errors="coerce").to_numpy()
    mask = ~np.isnan(y)
    y, cat = y[mask], cat[mask]
    if y.size == 0 or np.unique(cat).size < 2:
        return np.nan
    grand = y.mean()
    ssb = 0.0
    for k in np.unique(cat):
        yk = y[cat == k]
        ssb += len(yk) * (yk.mean() - grand) ** 2
    sst = ((y - grand) ** 2).sum()
    return ssb / sst if sst > 0 else np.nan


def _infer_emb_cols(df: pd.DataFrame, n_bands=64, prefix="A") -> List[str]:
    cols = []
    for i in range(n_bands):
        for c in (f"{prefix}{i:02d}", f"{prefix}{i}"):
            if c in df.columns:
                cols.append(c)
                break
        else:
            raise ValueError(f"缺少嵌入列 {prefix}{i:02d}/{prefix}{i}")
    return cols


def plot_corr_heatmap(
    corr_long_df: pd.DataFrame,
    out_path: str,
    top_n_dims: Optional[int] = 30,
    figsize=(12, 10),
):
    if corr_long_df.empty:
        print("[Info] 无相关结果，跳过热力图")
        return
    var_order = corr_long_df["var"].unique()

    pivot_abs = corr_long_df.pivot(index="emb", columns="var", values="rho").abs()

    # 重新排序列，按照出现顺序而不是字母排序
    pivot_abs = pivot_abs[var_order]
    if top_n_dims and pivot_abs.shape[0] > top_n_dims:
        order = pivot_abs.max(axis=1).sort_values(ascending=False).index[:top_n_dims]
        pivot_abs = pivot_abs.loc[order]
    plt.figure(figsize=figsize)
    im = plt.imshow(pivot_abs.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(pivot_abs.shape[0]), pivot_abs.index)
    plt.xticks(range(pivot_abs.shape[1]), pivot_abs.columns, rotation=60, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def plot_boxplots_by_cluster(df: pd.DataFrame, num_cols: List[str], out_dir: str):
    if "cluster" not in df.columns or not num_cols:
        print("[Info] 无 cluster 或连续属性，跳过箱线图")
        return
    ensure_dir(out_dir)
    clusters = sorted(df["cluster"].dropna().unique())
    for var in num_cols:
        data = [
            pd.to_numeric(df.loc[df["cluster"] == k, var], errors="coerce")
            .dropna()
            .values
            for k in clusters
        ]
        if sum(len(d) > 0 for d in data) < 2:
            continue
        plt.figure(figsize=(7, 5))
        plt.boxplot(data, labels=[f"C{k}" for k in clusters], showfliers=False)
        plt.xlabel("Cluster")
        plt.ylabel(var)
        plt.title(f"{var} by Cluster")
        plt.tight_layout()
        pth = os.path.join(out_dir, f"{var}_by_cluster.png")
        plt.savefig(pth, dpi=300)
        plt.close()
    print(f"[Saved] 箱线图 → {out_dir}")


def analyze_with_clusters(
    df_with_labels: pd.DataFrame,  # 你的聚类返回的 DataFrame
    attr_csv: str,  # 17维属性CSV（第一列为basin_id）
    out_dir: str = "results/cluster_analysis",
    heatmap_top_n_dims: int = 64,
    do_pca: bool = True,
) -> pd.DataFrame:
    """
    传入聚类结果 + 属性CSV，产出：
    - 簇质量: silhouette / Davies–Bouldin
    - 每簇样本量、属性均值：cluster_summary.csv
    - 连续属性簇间检验：ANOVA / Kruskal → cluster_attr_tests.csv
    - 类别属性簇×卡方：cluster_categorical_tests.csv
    - 嵌入维度与连续属性相关：embedding_vs_numeric_correlation.csv + 热力图
    - 嵌入PCA及载荷（可选）
    - 每个属性按簇箱线图（PNG若干）
    返回：并好属性后的总表（含 cluster）
    """
    ensure_dir(out_dir)
    # 1) 对齐属性
    df_attr = pd.read_csv(attr_csv)
    id_attr = df_attr.columns[0]
    df_attr[id_attr] = normalize_id(df_attr[id_attr])
    id_col = df_with_labels.columns[0]  # 假定第一列是 hru_id
    df_with_labels[id_col] = normalize_id(df_with_labels[id_col])
    df = pd.merge(df_with_labels, df_attr, left_on=id_col, right_on=id_attr, how="left")

    # 2) 基本列集
    emb_cols = _infer_emb_cols(df)
    attr_cols = [c for c in df_attr.columns if c != id_attr]
    cat_cols = [
        c for c in attr_cols if (df[c].dtype == "object" or df[c].nunique() <= 5)
    ]

    num_cols = [c for c in attr_cols if c not in cat_cols]
    print("数值型属性:", num_cols)
    print("类别属性:", cat_cols)
    # 3) 簇质量
    if "cluster" in df.columns and df["cluster"].nunique() > 1:
        Xz = StandardScaler().fit_transform(df[emb_cols].values)
        try:
            sil = silhouette_score(Xz, df["cluster"].values)
            db = davies_bouldin_score(Xz, df["cluster"].values)
        except Exception:
            sil, db = np.nan, np.nan
        print(f"[Cluster quality] silhouette={sil:.3f}, Davies–Bouldin={db:.3f}")
        # 保存簇中心（便于理解哪个维度差异大）
        centroids = df.groupby("cluster")[emb_cols].mean()
        centroids.to_csv(os.path.join(out_dir, "cluster_centroids_embedding.csv"))
        # 计算每簇相对全局z差的Top维度
        gmu, gsd = df[emb_cols].mean(), df[emb_cols].std(ddof=0)
        top_rows = []
        for k, row in centroids.iterrows():
            z = (row - gmu) / (gsd.replace(0, np.nan))
            top = z.abs().sort_values(ascending=False).head(10)
            top_rows.append(
                pd.DataFrame({"cluster": k, "emb": top.index, "abs_z": top.values})
            )
        pd.concat(top_rows, ignore_index=True).to_csv(
            os.path.join(out_dir, "cluster_topdims_by_abs_z.csv"), index=False
        )

    # 4) 簇×属性 摘要 + 显著性
    if "cluster" in df.columns and df["cluster"].nunique() > 1:
        # 摘要
        summary = []
        for k, grp in df.groupby("cluster"):
            row = {"cluster": int(k), "size": len(grp)}
            for c in num_cols:
                row[f"{c}_mean"] = pd.to_numeric(grp[c], errors="coerce").mean()
            summary.append(row)
        pd.DataFrame(summary).to_csv(
            os.path.join(out_dir, "cluster_summary.csv"), index=False
        )

        # 连续属性检验
        rows = []
        for c in num_cols:
            groups = [
                pd.to_numeric(g[c], errors="coerce").dropna().values
                for _, g in df.groupby("cluster")
            ]
            if all(len(g) > 1 for g in groups) and len(groups) >= 2:
                try:
                    F_p = f_oneway(*groups).pvalue
                except Exception:
                    F_p = np.nan
                try:
                    K_p = kruskal(*groups).pvalue
                except Exception:
                    K_p = np.nan
                rows.append({"var": c, "anova_p": F_p, "kruskal_p": K_p})
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "cluster_attr_tests.csv"), index=False
        )

        # 类别属性卡方
        cat_rows = []
        for c in cat_cols:
            if df[c].nunique() > 1:
                tbl = pd.crosstab(df[c], df["cluster"])
                if tbl.shape[0] > 1:
                    chi2, p, dof, exp = chi2_contingency(tbl)
                    cat_rows.append({"var": c, "chi2_p": p})
        if cat_rows:
            pd.DataFrame(cat_rows).to_csv(
                os.path.join(out_dir, "cluster_categorical_tests.csv"), index=False
            )

    # 5) 嵌入维度 × 连续属性 相关 + FDR + 热力图
    rows, pvals = [], []
    for axx in emb_cols:
        x = pd.to_numeric(df[axx], errors="coerce")
        for var in num_cols:
            y = pd.to_numeric(df[var], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 10:
                rho, p = spearmanr(x[mask], y[mask])
                rows.append({"emb": axx, "var": var, "rho": rho, "p": p})
                pvals.append(p)
    corr_df = pd.DataFrame(rows)
    if not corr_df.empty:
        corr_df["fdr_sig"] = fdr_bh(corr_df["p"].values, alpha=0.05)
        corr_df.to_csv(
            os.path.join(out_dir, "embedding_vs_numeric_correlation.csv"), index=False
        )
        plot_corr_heatmap(
            corr_df,
            os.path.join(out_dir, "heatmap_abs_rho.png"),
            top_n_dims=heatmap_top_n_dims,
            figsize=(12, 10),
        )
    """rows, pvals = [], []

    for axx in emb_cols:
        x = pd.to_numeric(df[axx], errors="coerce")

        for var in num_cols:
            y = pd.to_numeric(df[var], errors="coerce")
            mask = x.notna() & y.notna()

            if mask.sum() >= 10:
                # Compute mutual information
                mi = mutual_info_regression(
                    x[mask].values.reshape(-1, 1),
                    y[mask].values,
                    discrete_features=False
                )[0]

                # MI has no p-value; use NaN to keep DataFrame schema consistent
                rows.append({
                    "emb": axx,
                    "var": var,
                    "mi": mi,
                    "p": np.nan
                })
                pvals.append(np.nan)

    corr_df = pd.DataFrame(rows)

    # FDR cannot be computed without p-values, skip or keep placeholder
    if not corr_df.empty:
        corr_df["fdr_sig"] = False  # placeholder; MI has no p-value

        corr_df.to_csv(
            os.path.join(out_dir, "embedding_vs_numeric_mutual_information.csv"),
            index=False
        )

        # For heatmap, replace |rho| with MI values directly
        plot_corr_heatmap(
            corr_df.rename(columns={"mi": "rho"}),  # use same API, treat MI as "rho"
            os.path.join(out_dir, "heatmap_mutual_information.png"),
            top_n_dims=heatmap_top_n_dims,
            figsize=(12, 10)
        )"""

    # 6) 每个属性的簇箱线图
    if "cluster" in df.columns and df["cluster"].nunique() > 1 and num_cols:
        plot_boxplots_by_cluster(
            df, num_cols, out_dir=os.path.join(out_dir, "boxplots_by_cluster")
        )

    # 7) （可选）PCA帮助解释嵌入主轴
    if do_pca:
        Xz = StandardScaler().fit_transform(df[emb_cols].values)
        pca = PCA(n_components=5, random_state=42).fit(Xz)
        scores = pca.transform(Xz)
        pc_cols = [f"PC{i+1}" for i in range(5)]
        pd.DataFrame(scores, columns=pc_cols).assign(
            basin_id=df[id_attr].values
        ).to_csv(os.path.join(out_dir, "embedding_pca_scores.csv"), index=False)
        loadings = pd.DataFrame(pca.components_.T, index=emb_cols, columns=pc_cols)
        loadings.to_csv(os.path.join(out_dir, "pca_loadings.csv"))
        # PC 与连续属性相关
        pc_rows = []
        for i, pc in enumerate(pc_cols):
            for var in num_cols:
                r, p = spearmanr(scores[:, i], pd.to_numeric(df[var], errors="coerce"))
                pc_rows.append({"pc": pc, "var": var, "rho": r, "p": p})
        pc_df = pd.DataFrame(pc_rows)
        if not pc_df.empty:
            pc_df["fdr_sig"] = fdr_bh(pc_df["p"].values, alpha=0.05)
            pc_df.to_csv(
                os.path.join(out_dir, "pca_vs_numeric_correlation.csv"), index=False
            )

    print(f"[Done] 分析完成，结果在：{out_dir}")
    return df
