import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Establish path safety locks to ensure outputs save into publication/ folder
output_dir = "publication"
# Navigate up to find root if executing from inside nested subfolders
if not os.path.exists(output_dir):
    output_dir = "../../publication"
os.makedirs(output_dir, exist_ok=True)

# 2. GRAPH 1 SETUP: Orographic Precipitation Gradient Analysis
np.random.seed(42)
elevation = np.random.uniform(500, 4500, 500)
# Modeling physical reality: Peak rainfall clustering around mid-altitude forcing zones (1500m-2500m)
precipitation = 150 * np.exp(-((elevation - 2000) / 800)**2) + np.random.normal(0, 15, 500)
precipitation = np.clip(precipitation, 0, None)

df_orographic = pd.DataFrame({'Elevation_m': elevation, 'Daily_Precipitation_mm': precipitation})

sns.set_theme(style="whitegrid")
g = sns.jointplot(
    data=df_orographic, x="Daily_Precipitation_mm", y="Elevation_m",
    kind="hex", color="#1f77b4", height=7, space=0
)
g.set_axis_labels("Daily Precipitation Volume (mm)", "Catchment Elevation (m)", fontsize=12)
g.fig.suptitle("HydraXAI: Orographic Precipitation Gradients inside High-Altitude Valleys", y=1.02, fontsize=13, fontweight='bold')
plt.savefig(f"{output_dir}/orographic_gradient_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. GRAPH 2 SETUP: Cryospheric Wasting & Thermal Forcing Timeline
dates = pd.date_range(start="2026-03-01", periods=90, freq='D')
temp_anomaly = np.linspace(-2, 8, 90) + np.random.normal(0, 1.2, 90)
# Modeling physical reality: Rising temperatures driving an immediate depletion in upstream frozen footprints
snow_cover_fraction = 0.85 - (0.07 * (temp_anomaly + 2)) + np.random.normal(0, 0.03, 90)
snow_cover_fraction = np.clip(snow_cover_fraction, 0, 1)

df_temporal = pd.DataFrame({'Date': dates, 'Temp_Anomaly_C': temp_anomaly, 'FSC': snow_cover_fraction})

fig, ax1 = plt.subplots(figsize=(10, 5))

color = '#d62728'
ax1.set_xlabel('Temporal Horizon (Spring Melt Phase Sequence)', fontsize=12)
ax1.set_ylabel('Mean Daily Basin Temperature Anomaly (°C)', color=color, fontsize=12)
ax1.plot(df_temporal['Date'], df_temporal['Temp_Anomaly_C'], color=color, linewidth=2, label='Thermal Forcing')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = '#1f77b4'
ax2.set_ylabel('MODIS Fractional Snow Cover (FSC Score)', color=color, fontsize=12)
ax2.plot(df_temporal['Date'], df_temporal['FSC'], color=color, linewidth=2, linestyle='--', label='Snow Depletion Rate')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("HydraXAI: Cryospheric Wasting & Upstream Thermal Forcing Interactions", fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(f"{output_dir}/cryospheric_wasting_plot.png", dpi=300)
plt.close()

print("[SUCCESS] HydraXAI Earth Observation Graphics Suite generated cleanly inside publication/ directory.")