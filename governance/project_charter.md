# HydraXAI: Project Charter

## 1. Scientific Motivation & Environmental Context
The Indian Himalayan Region (IHR) is one of the most structurally fragile and climate-vulnerable mountain ecosystems globally. Unlike the predictable, uniform monsoonal features of tropical coastal zones, flash flooding in major Himalayan catchment basins is highly non-linear, unpredictable, and rapid. Disaster vectors in this high-altitude landscape are driven by steep altitudinal gradients, sudden localized convective cloudburst events, cascading landslide-induced river blocks, and shifting cryospheric dynamics (such as rapid seasonal snowmelt patterns and Glacial Lake Outburst Floods).

Traditional physical hydrodynamic models are computationally expensive and heavily dependent on localized river bathymetry data that is rarely available in data-scarce mountain regions. Conversely, standard deep learning configurations operate as uninterpretable "black boxes" that completely ignore physical geomorphological constraints, risking dangerous miscalculations when predicting hazards. 

HydraXAI bridges this gap. It treats high-altitude flood risk not simply as an isolated statistical anomaly, but as a compounding hydro-meteorological, topographic, and cryospheric interaction event. By integrating advanced non-parametric tree ensembles with game-theoretic Post-Hoc Explainable AI (XAI), the framework provides robust risk classification alongside human-interpretable, physically consistent explanations.

## 2. Active Research Questions
* **RQ1:** Can machine learning models accurately classify district-level flood risk across high-altitude watersheds using multi-source environmental and satellite data?
* **RQ2:** Does combining dynamic atmospheric parameters with seasonal cryospheric indices (Fractional Snow Cover and Snow Water Equivalent) noticeably improve prediction performance?
* **RQ3:** Which environmental, topographic, or cryospheric variables contribute most strongly to sudden flash flood generation across rugged mountain channels?
* **RQ4:** Can game-theoretic Post-Hoc Explainable AI (SHAP) provide transparent, physically plausible decision support for regional disaster-management stakeholders?

## 3. Multi-Criteria Core Objectives
* **Objective 1 (Heterogeneous Data Fusion):** To programmatically extract, clear, and align diverse planetary data layers (including IMD precipitation matrices, NASA SRTM elevation terrain models, and NASA MODIS cryospheric indices) into a standardized, projection-aligned grid system.
* **Objective 2 (Predictive Risk Benchmarking):** To rigorously benchmark optimized non-parametric ensembles ($XGBoost$ and $LightGBM$) against classical decision tree baselines for multi-day-ahead flash flood hazard classification.
* **Objective 3 (Axiomatic Explainability):** To deconstruct model interaction matrices using `TreeSHAP`, mapping automated risk scores directly back to established hydrological principles (e.g., verifying that a high risk score is driven by a thermal anomaly accelerating upstream snowmelt right before a precipitation event).

## 4. Open-Science Governance & Reproducibility Standards
To comply with the strict open-science directives mandated across elite European Master's programs and research networks, HydraXAI enforces the following replication protocols:
1. **Immutable Source Separation:** All raw geospatial rasters and vector inventories will be written to `data/raw/` and will never be modified or committed to Git tracking.
2. **Modular Ingestion Lineage:** All data preprocessing steps, coordinate reference system (CRS) transformations, and cloud-masking operations must be executed via automated, repeatable Python scripts inside `experiments/src/`.
3. **Auditable Development Logging:** Major research adjustments, framework choices, and literature reviews will be documented chronologically within the active research journal to create a transparent project timeline.