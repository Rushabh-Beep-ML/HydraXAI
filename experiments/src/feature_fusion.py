"""
Project HydraXAI: Feature Fusion Array Alignment Engine
Author: Rushabh Ahire Portfolio Initiative
Description: Core harmonization matrix layer. Stitches multi-source geospatial streams 
             (Topographic, Cryospheric, and Atmospheric arrays) into a unified, 
             machine-learning-ready Dataframe for tree ensemble training.
Spatial Target: Standarized (100, 120) Grid Mesh over the IHR Domain
"""

import os
import numpy as np
import pandas as pd

print("[LAUNCHING PIPELINE] Initializing Feature Fusion Matrix Layer...")

# =============================================================================
# 1. CORE ALGORITHMIC FUNCTION: Multi-Stream Matrix Stacking
# =============================================================================
def fuse_geospatial_arrays(slope_grid, twi_grid, fsc_grid, precip_grid):
    """
    Takes separate spatial matrix arrays of identical grid dimensions, flattens them,
    and stacks them feature-wise to construct a structured machine learning table.
    Filters out corrupted NaN data points caused by cloud cover.
    """
    # Verify all arrays match our standardized processing framework dimensions
    assert slope_grid.shape == twi_grid.shape == fsc_grid.shape == precip_grid.shape, \
        "Dimension Mismatch: Geospatial arrays must be aligned before fusion."
    
    rows, cols = slope_grid.shape
    
    # Generate geographic coordinate indexes for spatial tracking
    lat_indices, lon_indices = np.indices((rows, cols))
    
    # Flatten arrays into 1D continuous structural vectors
    flat_lat = lat_indices.flatten()
    flat_lon = lon_indices.flatten()
    flat_slope = slope_grid.flatten()
    flat_twi = twi_grid.flatten()
    flat_fsc = fsc_grid.flatten()
    flat_precip = precip_grid.flatten()
    
    # Construct unified tabular data structural matrix
    fusion_dataframe = pd.DataFrame({
        'Pixel_Index_X': flat_lon,
        'Pixel_Index_Y': flat_lat,
        'Feature_Slope_Rad': flat_slope,
        'Feature_TWI_Score': flat_twi,
        'Feature_Daily_FSC': flat_fsc,
        'Feature_Precip_mm': flat_precip
    })
    
    # Calculate pre-cleaning status
    raw_row_count = len(fusion_dataframe)
    
    # Drop rows containing NaNs (e.g., areas obscured by satellite cloud cover)
    # This aligns directly with our MODIS bitwise masking logic!
    cleaned_fusion_dataframe = fusion_dataframe.dropna().reset_index(drop=True)
    clean_row_count = len(cleaned_fusion_dataframe)
    dropped_rows = raw_row_count - clean_row_count
    
    # Programmatically derive the Target Variable: Flood Risk Proxy
    # High rainfall combined with steep terrain and low snowpack retention increases downstream risk
    # This provides our LightGBM model with an operational training baseline
    cleaned_fusion_dataframe['Target_Flood_Risk'] = (
        (cleaned_fusion_dataframe['Feature_Precip_mm'] * 0.5) +
        (np.sin(cleaned_fusion_dataframe['Feature_Slope_Rad']) * 25.0) +
        (cleaned_fusion_dataframe['Feature_TWI_Score'] * 2.0) -
        (cleaned_fusion_dataframe['Feature_Daily_FSC'] * 15.0)
    )
    
    # Convert absolute scores into a binary hazard classification label (1 = High Risk, 0 = Low Risk)
    risk_threshold = cleaned_fusion_dataframe['Target_Flood_Risk'].median()
    cleaned_fusion_dataframe['Hazard_Label'] = (cleaned_fusion_dataframe['Target_Flood_Risk'] > risk_threshold).astype(int)
    
    return cleaned_fusion_dataframe, dropped_rows

# =============================================================================
# 2. RUNTIME PIPELINE TESTING & INTEGRATION VALIDATION
# =============================================================================
if __name__ == "__main__":
    # Standard dimensions corresponding to our Atmospheric engine target (100x120 mesh)
    GRID_SHAPE = (100, 120)
    np.random.seed(44)
    
    print(f"[DATA INTEGRATION] Generating mock structural streams matching real-world sensors...")
    
    # 1. Mock Topographic inputs (from ingest_srtm.py calculations)
    mock_slope = np.random.uniform(0.1, 0.8, GRID_SHAPE).astype(np.float32)
    mock_twi = np.random.uniform(2.0, 14.0, GRID_SHAPE).astype(np.float32)
    
    # 2. Mock Cryospheric inputs with cloud anomalies (from ingest_modis.py bitwise mask)
    mock_fsc = np.random.uniform(0.0, 1.0, GRID_SHAPE).astype(np.float32)
    # Simulate a cloud-cover vector strip throwing NaN values
    mock_fsc[30:45, 20:60] = np.nan 
    
    # 3. Mock Atmospheric inputs (from ingest_weather.py downscaled grids)
    mock_precip = np.random.uniform(0.0, 150.0, GRID_SHAPE).astype(np.float32)
    
    # Run the fusion engine processing loop
    ml_ready_dataset, rows_filtered = fuse_geospatial_arrays(
        mock_slope, mock_twi, mock_fsc, mock_precip
    )
    
    print("\n[DATA VERIFICATION LOGS - FEATURE FUSION SUCCESS]")
    print(f" -> Input Grid Spatial Domain Dimensions : {GRID_SHAPE[0]}x{GRID_SHAPE[1]} Mesh Grid")
    print(f" -> Total Potential Grid Cells Mapped    : {GRID_SHAPE[0] * GRID_SHAPE[1]} Pixels")
    print(f" -> Cloud-Obscured Rows Dropped (NaNs)  : {rows_filtered} Pixels")
    print(f" -> Valid ML-Ready Vector Rows Remaining : {len(ml_ready_dataset)} Observations")
    print(f" -> Synthesized Feature Matrix Layout   : {list(ml_ready_dataset.columns)}")
    print(f" -> Target Hazard Base Distribution Class: {ml_ready_dataset['Hazard_Label'].value_counts().to_dict()}")
    
    # Verify export environment paths are clear
    fused_storage_path = "../data/processed/fused"
    os.makedirs(fused_storage_path, exist_ok=True)
    ml_ready_dataset.to_csv(f"{fused_storage_path}/hydraxai_training_set.csv", index=False)
    print(f"[STORAGE REGISTER] Tabular training data matrix saved to: {fused_storage_path}/hydraxai_training_set.csv")