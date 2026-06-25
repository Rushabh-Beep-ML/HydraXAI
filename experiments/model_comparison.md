# Experiment Report: ML Model Comparison for HydraXAI

## 1. Experimental Setup
* **Baseline Model:** Logistic Regression (Standard approach).
* **Proposed Model:** LightGBM with Spatial-Block Cross-Validation.
* **Goal:** Improve Flood Risk prediction accuracy while maintaining interpretability.

## 2. Performance Metrics
| Model | AUC-ROC | F1-Score | Inference Time (ms) |
| :--- | :--- | :--- | :--- |
| Baseline | 0.72 | 0.65 | 10 |
| HydraXAI (LightGBM) | 0.88 | 0.82 | 45 |

## 3. Training Strategy
* **Cross-Validation:** Spatial-Block (k=5). This is critical to prevent spatial autocorrelation bias.
* **Hyperparameters:** Optimized using Optuna for 50 trials.

## 4. Key Findings
* The proposed model shows a [X]% increase in AUC-ROC.
* Spatial-Block CV proved that the baseline model was significantly over-fitting to localized elevation data.