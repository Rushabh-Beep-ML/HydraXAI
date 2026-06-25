# Research Hypotheses: HydraXAI

**Primary Research Question:** Can multi-source geospatial data fusion, combined with SHAP-based interpretability, improve the robustness and physical grounding of regional flood risk predictions compared to traditional black-box models?

## Hypothesis 1 (Model Performance)
* **Statement:** The integration of spatial-block cross-validation will significantly reduce model overfitting to localized geographic anomalies compared to standard K-fold validation.
* **Test Metric:** Difference in F1-score and AUC-ROC between spatial-block and random split validation sets.

## Hypothesis 2 (Explainability & Physical Grounding)
* **Statement:** TreeSHAP feature attribution will reveal that "Topographic Wetness Index" (TWI) and "Slope" are consistently higher-ranked predictors of flood risk than meteorological variables during peak monsoon periods in the Himalayan region.
* **Test Metric:** Quantitative ranking of feature importance via SHAP global attribution summaries.

## Hypothesis 3 (Disaster Decision Support)
* **Statement:** Explanatory visualizations (SHAP force plots) will provide clearer decision-support for stakeholders than binary probability scores, as measured by the identification of specific risk-contributing variables.
* **Test Metric:** Qualitative analysis of risk-driver identification in high-flood-risk zones.git add research/hypotheses.md
