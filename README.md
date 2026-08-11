# Predicting 30-Day Hospital Readmissions in Diabetic Patients

Machine learning framework for predicting 30-day hospital readmission in diabetic patients, enabling early clinical intervention.

<img width="720" alt="Model performance" src="https://github.com/user-attachments/assets/cdd00770-14e8-4968-bf59-6bcb74d65e56" />


## Overview

30-day readmissions are a major clinical and financial burden for diabetic patients. This pipeline flags high-risk patients before discharge so clinicians can act early — follow-ups, medication reviews, personalized discharge plans — with a focus on maximizing recall for the minority (readmitted) class.

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**
- 101,766 encounters, 50 raw → 136 engineered features
- Target: `<30 days` → 1, `>30 or no readmission` → 0

## Preprocessing

- Missingness indicators for sparse labs (A1C, glucose serum); dropped extremely sparse columns; median imputation
- Numeric midpoints for age/weight ranges; numeric diagnosis codes
- One-hot encoding (race, payer code, specialty); encoded 23 medication columns
- Dropped non-predictive identifiers → final shape: 101,766 × 136, all numeric

## Models

**Baseline:** Logistic Regression, Random Forest, XGBoost, LightGBM, MLP

**Optimized:** each tuned for minority-class recall via F2-score threshold optimization, regularization, depth/subsampling constraints, child-weight adjustments, and custom NN thresholds

## Getting Started

```bash
git clone https://github.com/dani-loya/Diabetes-Readmission-ML-DL-Framework.git
```

1. Open the [Colab notebook] in this repository (https://colab.research.google.com/drive/1RJ00G_pOc0XxV-j_q0k2_nmc0q4LJpXF?usp=sharing)
2. Upload `diabetic_data.csv`
3. Run all cells to reproduce preprocessing, modeling, and evaluation

## Video 

https://github.com/user-attachments/assets/d6134062-8b6c-462a-8023-b30482a684a5



## Future Work

- SHAP interpretability
- Clinical dashboard
- Deployment via FastAPI or Streamlit

## Reference

Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates*, UCI Machine Learning Repository.
