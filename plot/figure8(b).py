import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ============================
# 字体设置
# ============================
rcParams["font.family"] = "Times New Roman"
rcParams["axes.unicode_minus"] = False

# ============================
# 数据
# ============================
data = [
    ["method1", 100, 25, 147, 17.006803],
    ["method1", 200, 13, 147, 8.843537],
    ["method1", 300, 26, 147, 17.687075],
    ["method1", 400, 15, 147, 10.204082],
    ["method1", 500, 25, 147, 17.006803],
    ["method1", 600, 14, 147, 9.523810],
    ["method1", 670, 29, 147, 19.727891],
    ["method2", 100, 23, 147, 15.646259],
    ["method2", 200, 11, 147, 7.482993],
    ["method2", 300, 19, 147, 12.925170],
    ["method2", 400, 31, 147, 21.088435],
    ["method2", 500, 15, 147, 10.204082],
    ["method2", 600, 26, 147, 17.687075],
    ["method2", 670, 22, 147, 14.965986],
    ["method3", 100, 33, 147, 22.448980],
    ["method3", 200, 12, 147, 8.163265],
    ["method3", 300, 16, 147, 10.884354],
    ["method3", 400, 23, 147, 15.646259],
    ["method3", 500, 14, 147, 9.523810],
    ["method3", 600, 25, 147, 17.006803],
    ["method3", 670, 24, 147, 16.326531],
    ["method4", 100, 12, 147, 8.163265],
    ["method4", 200, 23, 147, 15.646259],
    ["method4", 300, 16, 147, 10.884354],
    ["method4", 400, 28, 147, 19.047619],
    ["method4", 500, 25, 147, 17.006803],
    ["method4", 600, 22, 147, 14.965986],
    ["method4", 670, 21, 147, 14.285714],
]

df = pd.DataFrame(
    data, columns=["method", "k", "best_basin_count", "total_basin_count", "percentage"]
)

# ============================
# 方法名替换，按你的论文方法名称修改
# ============================
method_name_map = {
    "method1": "CAMELS Attr.",
    "method2": "AEF",
    "method3": "CAMELS Emb.",
    "method4": "Random",
}

df["method_label"] = df["method"].map(method_name_map)

# ============================
# 透视表：行是 method，列是 k，值是 percentage
# ============================
heatmap_df = df.pivot(index="method_label", columns="k", values="percentage")

# 按指定顺序排列
method_order = ["CAMELS Attr.", "AEF", "CAMELS Emb.", "Random"]
k_order = [100, 200, 300, 400, 500, 600, 670]

heatmap_df = heatmap_df.loc[method_order, k_order]

# ============================
# 绘图
# ============================
fig, ax = plt.subplots(figsize=(10, 3.8))

im = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd")

# x/y 轴
ax.set_xticks(np.arange(len(k_order)))
ax.set_xticklabels(k_order, fontsize=18)

ax.set_yticks(np.arange(len(method_order)))
ax.set_yticklabels(method_order, fontsize=18)

ax.set_xlabel("Number of training basins, k", fontsize=18)
ax.set_ylabel("Method", fontsize=18)

# 数值标注
for i in range(heatmap_df.shape[0]):
    for j in range(heatmap_df.shape[1]):
        value = heatmap_df.iloc[i, j]
        ax.text(
            j, i, f"{value:.1f}%", ha="center", va="center", fontsize=18, color="black"
        )

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Percentage of target basins (%)", fontsize=18)
cbar.ax.tick_params(labelsize=18)

ax.set_title(
    "Distribution of optimal donor-basin pool size across 150 target basins",
    fontsize=18,
    pad=12,
)

plt.tight_layout()
plt.savefig(
    "/Users/cylenlc/work/AlphaEarth4PUB/plot/optimal_k_percentage_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
