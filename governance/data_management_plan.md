# HydraXAI: Data Management Plan (DMP)

## 1. Spatial Boundaries & Coordinate Reference Systems (CRS)
To enforce absolute spatial consistency across diverse planetary data arrays, the HydraXAI framework targets a rigid bounding box mapping the core river corridors of Uttarakhand and Himachal Pradesh. All geographic variables will be aligned to the uniform spatial projections designated below.

* **Target Geographic Domain:** Indian Himalayan Region (IHR Core Valleys)
* **Bounding Coordinates Grid:**
  * Minimum Longitude: `75.50° E`
  * Maximum Longitude: `81.50° E`
  * Minimum Latitude: `28.50° N`
  * Maximum Latitude: `33.50° N`
* **Standardized Spatial Reference:** `EPSG:4326` (WGS 84 - Geospatial coordinate arrays) reprojected to `EPSG:32643` (WGS 84 / UTM Zone 43N) via GeoPandas and Rasterio for accurate topographically explicit meter-scale calculations.

## 2. Satellite Vector Data Dictionary & Resolution Matrix

| Ingestion Layer | Source Agency | Spatial Resolution | Temporal Frequency | Archive Data Type |
| :--- | :--- | :--- | :--- | :--- |
| **Atmospheric Grids** | India Meteorological Department (IMD) / ECMWF ERA5-Land | 10km to 25km grid cells | Daily Aggregate | Multi-band NetCDF-4 (`.nc`) |
| **High-Relief Terrain** | NASA SRTM v3 | 30-Meter (1 Arc-Second) | Static Surface | GeoTIFF Raster Grid (`.tif`) |
| **Cryosphere Footprint** | NASA MODIS Terra (MOD10A1) | 500-Meter Array | Daily Re-visit | HDF4 / HDF5 Structure |
| **Human Infrastructure** | ESA WorldCover | 10-Meter Grid | Annual Evaluation | LULC Classification Byte |

## 3. Data Processing Pipeline Architecture

┌────────────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
│  Raw Planetary Inputs  │ ───> │ extract_transform.py  │ ───> │ Unified Array Matrices │
│ (NetCDF / GeoTIFF / HDF)│      │  (CRS Mask / Cloud Cut) │      │ (Saved to processed/)│
└────────────────────────┘      └─────────────────────────┘      └────────────────────────┘
 **Ingestion Loop (`data/raw/`):** Downloader routines query Earth Explorer, NSIDC, and Copernicus servers via API endpoints to pull historical raw profiles. 
 **Harmonization Engine (`experiments/src/`):** Rasterio clips planetary extensions to match our structural IHR bounding masks, and GeoPandas aligns polygon arrays.
 **Array Serialization (`data/processed/`):** Aligned parameters are compiled into standardized NumPy numerical arrays, completely avoiding disk overheads during machine learning executions.