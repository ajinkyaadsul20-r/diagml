# Model Card: diagml breast-cancer classifier

| | |
|---|---|
| **Task** | Binary classification: malignant (0) vs benign (1) |
| **Data** | Wisconsin Diagnostic Breast Cancer, 569 samples, 30 numeric features computed from digitised fine-needle-aspirate images |
| **Model** | Best of logistic regression / random forest / gradient boosting by CV ROC-AUC (currently logistic regression with scaling) |
| **Intended use** | Education, benchmarking, demonstrating ML engineering practice |
| **Out of scope** | Clinical diagnosis, screening, or any decision affecting a person's health |

## Evaluation
Stratified 80/20 split, seed 42. Test ROC-AUC 0.995 (95% bootstrap CI 0.983 to 1.000), accuracy 0.982.
With 114 test samples, one extra error moves accuracy by about 0.9 points: treat small differences as noise.

## Limitations and risks
- Single-institution, small dataset; no external validation, so performance on other populations is unknown.
- Features come from a curated lab pipeline, not raw images.
- Probabilities are reasonably calibrated on this test set but this is not guaranteed under distribution shift.
- False negatives (missed malignancies) are the costly error; adjust `decision_threshold` in the config
  if recall on the malignant class matters more than precision.

## Reproducibility
Seed, data fingerprint, hyper-parameters and library versions are recorded in `artifacts/metadata.json`.
