var early_start = '1984-01-01';
var early_end = '1991-01-01';
var recent_start = '2017-01-01';
var recent_end = '2025-01-01';

// Collection 2 Level-2 Surface Reflectance scale factor
function applyScaleFactors(image) {
    var opticalBands = image.select('SR_B.*')
        .multiply(0.0000275)
        .add(-0.2);

    return image.addBands(opticalBands, null, true)
        .copyProperties(image, ['system:time_start']);
}

// 更完整的云、阴影、雪、饱和像元掩膜
function maskClouds(image) {
    var qa = image.select('QA_PIXEL');

    var mask = qa.bitwiseAnd(1 << 0).eq(0)   // Fill
        .and(qa.bitwiseAnd(1 << 1).eq(0))      // Dilated cloud
        .and(qa.bitwiseAnd(1 << 3).eq(0))      // Cloud
        .and(qa.bitwiseAnd(1 << 4).eq(0))      // Cloud shadow
        .and(qa.bitwiseAnd(1 << 5).eq(0));     // Snow

    var satMask = image.select('QA_RADSAT').eq(0);

    return image.updateMask(mask)
        .updateMask(satMask)
        .copyProperties(image, ['system:time_start']);
}

// L5: 先重命名为统一波段
function renameL5(image) {
    return image.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    ).copyProperties(image, ['system:time_start']);
}

// L8/L9: 重命名为统一波段
function renameL89(image) {
    return image.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    ).copyProperties(image, ['system:time_start']);
}

// 把 L5 TM 反射率转换到 OLI-equivalent reflectance space
function harmonizeL5ToOLI(image) {
    var bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];

    var slopes = ee.Image.constant([
        0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071
    ]);

    var intercepts = ee.Image.constant([
        0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172
    ]);

    var harmonized = image.select(bands)
        .multiply(slopes)
        .add(intercepts)
        .rename(bands);

    return image.addBands(harmonized, null, true)
        .copyProperties(image, ['system:time_start']);
}

// 统一 NDVI 计算
function addNDVI(image) {
    var ndvi = image.normalizedDifference(['nir', 'red'])
        .rename('NDVI');

    return image.addBands(ndvi)
        .copyProperties(image, ['system:time_start']);
}

// 早期：Landsat 5，harmonized NDVI
var L5_early = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterDate(early_start, early_end)
    .filter(ee.Filter.calendarRange(6, 9, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(applyScaleFactors)
    .map(maskClouds)
    .map(renameL5)
    .map(harmonizeL5ToOLI)
    .map(addNDVI)
    .select('NDVI')
    .median()
    .rename('NDVI_early');

// 近期：Landsat 8
var L8_recent = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(recent_start, recent_end)
    .filter(ee.Filter.calendarRange(6, 9, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(applyScaleFactors)
    .map(maskClouds)
    .map(renameL89)
    .map(addNDVI)
    .select('NDVI');

// 近期：Landsat 9
var L9_recent = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate(recent_start, recent_end)
    .filter(ee.Filter.calendarRange(6, 9, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 50))
    .map(applyScaleFactors)
    .map(maskClouds)
    .map(renameL89)
    .map(addNDVI)
    .select('NDVI');

var L89_recent = L8_recent.merge(L9_recent)
    .median()
    .rename('NDVI_recent');

// delta NDVI
var delta_NDVI = L89_recent
    .subtract(L5_early)
    .rename('delta_NDVI');

var combined = L5_early
    .addBands(L89_recent)
    .addBands(delta_NDVI);

// 按流域提取统计值
var basin_stats = combined.reduceRegions({
    collection: basins,
    reducer: ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), null, true)
        .combine(ee.Reducer.percentile([25, 75]), null, true),
    scale: 30,
    tileScale: 4
});

Export.table.toDrive({
    collection: basin_stats,
    description: 'CAMELS_harmonized_NDVI_stability',
    fileFormat: 'CSV',
    selectors: [
        'hru_id',
        'NDVI_early_mean', 'NDVI_early_stdDev',
        'NDVI_recent_mean', 'NDVI_recent_stdDev',
        'delta_NDVI_mean', 'delta_NDVI_stdDev',
        'delta_NDVI_p25', 'delta_NDVI_p75'
    ]
});

print('导出任务已提交，请在 Tasks 面板中运行');