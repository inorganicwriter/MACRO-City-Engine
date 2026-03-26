## Google Earth Engine bridge

`MACRO-City Engine` 当前的 GEE 工作流使用城市多边形边界，不再推荐旧的点缓冲导出。
兼容性原因下，导出文件名前缀仍保留 `urban_pulse_*`。

### 1. 上传城市边界表

优先上传：

- `data/raw/city_catalog.csv`

要求：

- 使用 `geometry_wkt` 作为几何列
- 几何必须是 `EPSG:4326` 下的 `POLYGON` / `MULTIPOLYGON`
- 在 Earth Engine 中导入为 table asset

不再推荐上传旧的 `gee_city_points.csv`，因为它是历史点缓冲流程的遗留文件。

### 2. 使用现成的 polygon 导出脚本

编辑并运行：

- `data/raw/gee_bundle/gee_export_viirs_monthly.js`
- `data/raw/gee_bundle/gee_export_ghsl_yearly.js`

将脚本中的 `ASSET_ID` 改成你实际上传的 asset id。

官方数据集：

- VIIRS monthly: `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`
- GHSL built surface: `JRC/GHSL/P2023A/GHS_BUILT_S`
- GHSL built volume: `JRC/GHSL/P2023A/GHS_BUILT_V`

### 3. 从 Google Drive 下载导出结果并回灌

```bash
python3 run_gee_bridge.py \
  --import-viirs /path/to/urban_pulse_viirs_monthly.csv \
  --import-ghsl /path/to/urban_pulse_ghsl_yearly.csv
```

这会更新：

- `data/raw/viirs_city_monthly.csv`
- `data/raw/city_ghsl_yearly.csv`

### 4. 重跑主面板

```bash
/mnt/d/Anaconda/python.exe run_pipeline.py --max-cities 295
```

### 5. 兼容性说明

- `run_gee_bridge.py --prepare-bundle` 仍可生成旧模板，但主推荐链路是上传 `city_catalog.csv` 的 polygon 资产
- 导出结果文件名中的 `urban_pulse_*` 前缀保留，仅用于兼容既有导入脚本
