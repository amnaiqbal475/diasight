"""All evaluation metrics in one place, used for APTOS validation and IDRiD alike.
Covers the instructor's list: per-class sensitivity/specificity, macro-F1, confusion matrix,
plus referable sensitivity/specificity/AUC and quadratic weighted kappa.
Bootstrap 95% CIs because IDRiD is small (516 images)."""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score

GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    """One-vs-rest sensitivity, specificity and F1 for each grade."""
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    rows, total = [], cm.sum()
    for g in range(5):
        tp = cm[g, g]; fn = cm[g].sum() - tp; fp = cm[:, g].sum() - tp; tn = total - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * prec * sens / (prec + sens) if prec and sens and prec + sens else 0.0
        rows.append({"grade": GRADES[g], "n": int(cm[g].sum()), "sensitivity": sens,
                     "specificity": spec, "F1": f1})
    return pd.DataFrame(rows).round(3)


def _refer(y_true, refer_score, threshold):
    t = np.asarray(y_true) >= 2; p = np.asarray(refer_score) >= threshold
    sens = (p & t).sum() / t.sum() if t.sum() else np.nan
    spec = (~p & ~t).sum() / (~t).sum() if (~t).sum() else np.nan
    return sens, spec


def summary(y_true, y_pred, refer_score, threshold=0.5, n_boot=1000, seed=42) -> pd.DataFrame:
    """Headline metrics with bootstrap 95% confidence intervals."""
    y_true, y_pred, refer_score = map(np.asarray, (y_true, y_pred, refer_score))

    def compute(idx):
        yt, yp, rs = y_true[idx], y_pred[idx], refer_score[idx]
        s, sp = _refer(yt, rs, threshold)
        auc = roc_auc_score(yt >= 2, rs) if 0 < (yt >= 2).sum() < len(yt) else np.nan
        return {"Refer sensitivity": s, "Refer specificity": sp, "Refer AUC": auc,
                "Quadratic weighted kappa": cohen_kappa_score(yt, yp, weights="quadratic"),
                "Macro-F1 (5 grades)": f1_score(yt, yp, average="macro", labels=range(5), zero_division=0),
                "Accuracy (not used to judge)": (yt == yp).mean()}

    point = compute(np.arange(len(y_true)))
    rng = np.random.default_rng(seed)
    boots = pd.DataFrame([compute(rng.integers(0, len(y_true), len(y_true))) for _ in range(n_boot)])
    return pd.DataFrame({"value": point,
                         "95% CI low": boots.quantile(0.025),
                         "95% CI high": boots.quantile(0.975)}).round(3)


def report(y_true, y_pred, refer_score, threshold=0.5, title=""):
    """Print everything for one dataset; returns (summary_df, per_class_df, confusion_df)."""
    s = summary(y_true, y_pred, refer_score, threshold)
    pc = per_class_table(y_true, y_pred)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(5)),
                      index=[f"true {g}" for g in range(5)], columns=[f"pred {g}" for g in range(5)])
    print(f"=== {title} (n={len(y_true)}, refer cut-off {threshold}) ===")
    print(s.to_string(), "\n"); print(pc.to_string(index=False), "\n"); print(cm.to_string(), "\n")
    return s, pc, cm
