# DiaSight

**Catch it before it steals sight.** AI screening for diabetic retinopathy from a retina (fundus) photo:
grade 0–4, refer / no-refer (grade 2 or higher = refer), Grad-CAM heatmap, advice in English and Urdu, PDF report.

> Screening tool, not a diagnosis. Course prototype (AI/ML capstone, Zainab & Amna); not for clinical use.

## Run the app

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

With no trained model in `models/`, the app runs in **demo mode** (full flow, untrained network, clearly labelled).
To use a real model, run `notebooks/00_prototype_baseline.ipynb` in Colab (T4 GPU) and copy its output:
`best.pt` + `config.json` → `models/`, sample images → `app/samples/`.

## Layout

```
diasight/            shared package used by BOTH the notebooks and the app
  config.py          paths, grade names, default settings
  preprocess.py      crop black border, resize, Ben Graham, circular mask, normalise
  quality.py         blur (variance of Laplacian) and brightness checks + calibration
  dataset.py         PyTorch Dataset and augmentations
  model.py           build model from timm, load checkpoint (or demo mode)
  predict.py         image -> grade, refer decision, probabilities
  metrics.py         evaluation metrics with confidence intervals
  gradcam.py         heatmap overlay
  advice.py          advice per grade (EN, UR)
  report.py          PDF report
notebooks/           Colab notebooks (training, evaluation)
app/app.py           Streamlit app; app/assets (logo, Urdu font); app/samples (demo images)
models/              best.pt + config.json (weights not committed)
```

## Data

APTOS 2019 Blindness Detection (Kaggle) for training/validation; IDRiD (Porwal et al., 2018) as an independent test set
only. Both are public and anonymised.

## Results

Reported for APTOS validation and the IDRiD external test (`diasight/metrics.py`), with bootstrap 95% CIs:
referable sensitivity, specificity and AUC; quadratic weighted kappa; per-grade sensitivity, specificity and F1;
macro-F1; 5x5 confusion matrix. Accuracy is shown but never used to judge the model.

_Baseline (EfficientNet-B0 @ 224): to be filled in from the notebook._

Urdu font: Noto Naskh Arabic, SIL Open Font License (see `app/assets/fonts/OFL.txt`).
