// ==================== 工具函数 ====================
function calculate_basin_embeddings_batch(embedding_data, basin_shps, shps_batch_name, the_year, basin_id_key_in_shp, batch_size) {
    var year = ee.Number(the_year);
    var start_date = ee.Date.fromYMD(year, 1, 1);
    var end_date = start_date.advance(1, 'year');

    // 加载并镶嵌年度数据
    var annual_embedding = embedding_data
        .filterDate(start_date, end_date)
        .mosaic();

    // 生成波段名数组
    var band_names_js = [];
    for (var i = 0; i < 64; i++) {
        band_names_js.push("A" + (i < 10 ? "0" + i : i)); // ["A00","A01",...,"A63"]
    }

    // 流域总数
    var total_count = basin_shps.size();
    print("Total basin count: ", total_count);

    // 计算批次数
    var total_batches = ee.Number(total_count).divide(batch_size).ceil();
    print("Total batches: ", total_batches);

    // 循环分批导出
    total_batches.evaluate(function (n) {
        for (var b = 0; b < n; b++) {
            var start_index = b * batch_size;
            var end_index = start_index + batch_size;

            var basin_batch = basin_shps.toList(batch_size, start_index);
            basin_batch = ee.FeatureCollection(basin_batch);

            var basin_embeddings = annual_embedding
                .select(band_names_js)
                .reduceRegions({
                    collection: basin_batch,
                    reducer: ee.Reducer.mean(),
                    scale: 1000,
                    tileScale: 8,
                    maxPixelsPerRegion: 1e16
                })
                .map(function (feature) {
                    return feature.set({
                        "year": year,
                        "hru_id": feature.get(basin_id_key_in_shp)
                    });
                });

            Export.table.toDrive({
                collection: basin_embeddings,
                description: "SatEmb_" + shps_batch_name + "_" + the_year + "_batch" + b,
                folder: "SatelliteEmbedding",
                fileNamePrefix: "BasinEmbeddings_" + shps_batch_name + "_" + the_year + "_batch" + b,
                selectors: ["hru_id", "year"].concat(band_names_js)
            });
        }
    });
}

// ==================== 调用示例 ====================
var batch_name = "basins";
var basinid_key_in_shp = "CatchID";
var embedding_dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

// 每批 50 个流域（可以改大或改小）
calculate_basin_embeddings_batch(embedding_dataset, basins, batch_name, 2019, basinid_key_in_shp, 50);