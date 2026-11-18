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

The app has three pages: **Screen** (photo in, decision, heatmap, advice, PDF), **Performance** (model card built from
`reports/figures/`) and **About**. A sample can be opened directly with `?sample=<file name>`, e.g.
`http://localhost:8501/?sample=grade3_7b29e3783919.png`.

With no trained model in `models/`, the app runs in **demo mode** (full flow, untrained network, clearly labelled).

## Train and evaluate

```bash
python scripts/train.py --data data/aptos --model efficientnet_b3 --size 384 --weights none --epochs 15 --lr 3e-4 --name E_b3_384
python scripts/evaluate.py --data data/idrid          # external test on IDRiD, nothing tuned
python scripts/smoke_check.py                          # quick check that the app's model loads and grades
```

`train.py` writes `models/best.pt` + `models/config.json` (the app loads these), keeps a copy in `models/runs/<name>/`,
picks demo images into `app/samples/`, and writes figures to `reports/figures/<name>/`. It also calibrates the photo
quality gate: minimum sharpness and brightness are the 2nd percentile of 300 dataset photos (rounded down).
The Colab route (`notebooks/00_prototype_baseline.ipynb`, T4 GPU) produces the same files.

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
app/app.py           Streamlit entry point: page setup, top bar, navigation
app/ui.py            shared look and feel: CSS, HTML components (verdict, scale, meter, charts)
app/views/           screen.py, performance.py, about.py
app/assets, samples  logo, Urdu font; demo images (one per grade)
scripts/             train.py, evaluate.py, smoke_check.py
reports/figures/     metrics, confusion matrices and experiment log per run
models/              best.pt + config.json (weights not committed)
```

## Data

APTOS 2019 Blindness Detection (Kaggle) for training/validation; IDRiD (Porwal et al., 2018) as an independent test set
only. Both are public and anonymised.

## Results

Reported for APTOS validation and the IDRiD external test (`diasight/metrics.py`), with bootstrap 95% CIs:
referable sensitivity, specificity and AUC; quadratic weighted kappa; per-grade sensitivity, specificity and F1;
macro-F1; 5x5 confusion matrix. Accuracy is shown but never used to judge the model.

Model in use: **EfficientNet-B3 @ 384px**, classification head, no class weights (run `E_b3_384`), refer cut-off 0.40 (set in `models/config.json`; `train.py` writes 0.5 by default).

| Metric | APTOS validation (n=733) | IDRiD external (n=516) |
|---|---|---|
| Refer sensitivity | 0.936 (0.906 to 0.960) | 0.839 (0.796 to 0.876) |
| Refer specificity | 0.936 (0.912 to 0.959) | 0.907 (0.866 to 0.945) |
| Refer AUC | 0.981 (0.972 to 0.987) | 0.941 (0.920 to 0.960) |
| Quadratic weighted kappa | 0.896 (0.872 to 0.918) | 0.751 (0.696 to 0.801) |
| Macro-F1 (5 grades) | 0.675 | 0.480 |

Known weak spots: grade 3 (severe) sensitivity is low (0.28 validation, 0.13 IDRiD; mostly called grade 2 or 4, which
are still referred) and grade 1 (mild) is often called grade 2. All runs are compared in `reports/figures/experiments.csv`
and on the app's Performance page.

Urdu font: Noto Naskh Arabic, SIL Open Font License (see `app/assets/fonts/OFL.txt`).
