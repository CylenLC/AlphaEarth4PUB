from pathlib import Path
import os

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "plot/outputs"

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/alphaearth4pub_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/alphaearth4pub_cache")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CLIMATE_SHP = Path(
    "/Users/cylenlc/Downloads/climatezones_shapefile/NA_ClimateZones/data/"
    "North_America_Climate_Zones.shp"
)
BASIN_SHP = ROOT / "data/basin_set_full_res/HCDN_nhru_final_671.shp"
RANK_CSV = ROOT / "data/alphaearth_vs_camels_basin_rank.csv"

OUT_FIG = OUT_DIR / "ranked_basins_climate_zones.png"
OUT_TABLE = OUT_DIR / "ranked_basins_climate_zones.csv"

# 1-based CSV row number, including the header row.
# START_ROW = 5 and END_ROW = 25 means physical lines 5 through 25 of the CSV.
START_ROW = 5
END_ROW = 25
ID_COL = 0


def normalize_id(values: pd.Series) -> pd.Series:
    """Normalize basin IDs for matching IDs with and without leading zeros."""
    values = values.astype(str).str.strip().str.replace(".0", "", regex=False)

    def _normalize_one(value: str) -> str:
        if value == "" or value.lower() == "nan":
            return ""
        return str(int(value)) if value.isdigit() else value.lstrip("0") or "0"

    return values.map(_normalize_one)


def display_id(values: pd.Series) -> pd.Series:
    """Format numeric USGS gauge IDs as 8-digit strings for output."""
    normalized = normalize_id(values)
    return normalized.map(lambda value: value.zfill(8) if value.isdigit() else value)


def read_selected_basin_ids() -> list[str]:
    df = pd.read_csv(RANK_CSV, dtype=str)
    first_data_index = max(START_ROW - 2, 0)
    last_data_index = max(END_ROW - 1, first_data_index + 1)
    ids = normalize_id(df.iloc[first_data_index:last_data_index, ID_COL]).dropna()
    ids = [value for value in ids if value]
    return list(dict.fromkeys(ids))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_ids = read_selected_basin_ids()
    if not selected_ids:
        raise ValueError(f"No basin IDs found in {RANK_CSV} from row {START_ROW}.")

    climate = gpd.read_file(CLIMATE_SHP)
    basins = gpd.read_file(BASIN_SHP)

    basins["basin_id_norm"] = normalize_id(basins["hru_id"])
    basins["basin_id"] = display_id(basins["hru_id"])

    selected = basins[basins["basin_id_norm"].isin(selected_ids)].copy()
    if selected.empty:
        raise ValueError("No selected basin IDs matched the basin shapefile.")

    order = {basin_id: i for i, basin_id in enumerate(selected_ids)}
    selected["rank_order"] = selected["basin_id_norm"].map(order)
    selected = selected.sort_values("rank_order")

    selected_points = gpd.GeoDataFrame(
        selected.drop(columns="geometry"),
        geometry=gpd.points_from_xy(selected["lon_cen"], selected["lat_cen"]),
        crs="EPSG:4326",
    ).to_crs(climate.crs)

    joined = gpd.sjoin(
        selected_points,
        climate[["Code", "Climate", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    missing_ids = sorted(set(selected_ids) - set(selected["basin_id_norm"]))
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs were not found in basin shapefile:")
        print(", ".join(missing_ids))

    output_cols = [
        "rank_order",
        "basin_id",
        "hru_id",
        "lon_cen",
        "lat_cen",
        "Code",
        "Climate",
    ]
    joined[output_cols].to_csv(OUT_TABLE, index=False)

    climate_wgs84 = climate.to_crs("EPSG:4326")
    points_wgs84 = joined.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(11, 8))
    climate_wgs84.plot(
        ax=ax,
        color="#f2f2f2",
        edgecolor="#c7c7c7",
        linewidth=0.35,
    )

    codes = [code for code in sorted(points_wgs84["Code"].dropna().unique())]
    cmap = plt.get_cmap("tab20", max(len(codes), 1))
    color_by_code = {code: cmap(i) for i, code in enumerate(codes)}

    for code, group in points_wgs84.groupby("Code", dropna=False):
        label_code = "Unknown" if pd.isna(code) else code
        color = "#222222" if pd.isna(code) else color_by_code[code]
        group.plot(
            ax=ax,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.35,
            markersize=42,
            label=label_code,
            zorder=3,
        )

    minx, miny, maxx, maxy = points_wgs84.total_bounds
    pad_x = max((maxx - minx) * 0.15, 2)
    pad_y = max((maxy - miny) * 0.15, 2)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_title(
        "Climate zones and locations of selected ranked basins",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    legend_handles = []
    for code in codes:
        climate_name = (
            points_wgs84.loc[points_wgs84["Code"] == code, "Climate"]
            .dropna()
            .iloc[0]
        )
        count = int((points_wgs84["Code"] == code).sum())
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color_by_code[code],
                markeredgecolor="black",
                markersize=8,
                label=f"{code}: {climate_name} ({count})",
            )
        )

    unknown_count = int(points_wgs84["Code"].isna().sum())
    if unknown_count:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#222222",
                markeredgecolor="black",
                markersize=8,
                label=f"Unknown ({unknown_count})",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Climate zone",
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")

    print(f"Selected basin count: {len(points_wgs84)}")
    print("\nClimate zone counts:")
    print(points_wgs84["Code"].fillna("Unknown").value_counts().sort_index())
    print(f"\nSaved figure: {OUT_FIG}")
    print(f"Saved table:  {OUT_TABLE}")


if __name__ == "__main__":
    main()
