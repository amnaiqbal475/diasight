"""Test a trained DiaSight model on an external dataset it has never seen (IDRiD).

Run from the repo root:
    python scripts/evaluate.py --data data/idrid
    python scripts/evaluate.py --data data/idrid --model-dir models/runs/C_cls_none

Nothing here is tuned: the model, preprocessing and refer cut-off come from config.json
exactly as they are, so the result is an honest estimate for a new hospital and camera.
Writes reports/figures/<run>/external_<tag>_* and adds a row to reports/figures/external_tests.csv.
"""
import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diasight import metrics  # noqa: E402
from diasight.preprocess import load_rgb, preprocess, to_tensor  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def split_of(path: str) -> str:
    """IDRiD's training and testing parts reuse the same file names, so we tell them apart by folder/CSV name."""
    p = path.lower()
    return "test" if "test" in p else "train" if "train" in p else ""


def find_labels(data_dir: Path) -> pd.DataFrame:
    """Read every grading CSV under data_dir and match each row to its image in the same part (train/test).
    Accepts IDRiD columns ('Image name', 'Retinopathy grade') or APTOS-style ('id_code', 'diagnosis')."""
    images = {}
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            images.setdefault((split_of(str(p.relative_to(data_dir))), p.stem), p)
    frames = []
    for csv in sorted(data_dir.rglob("*.csv")):
        df = pd.read_csv(csv)
        df.columns = [str(c).strip() for c in df.columns]
        name_col = next((c for c in df.columns if c.lower() in {"image name", "image", "id_code"}), None)
        grade_col = next((c for c in df.columns if c.lower() in {"retinopathy grade", "diagnosis", "grade", "level"}), None)
        if not (name_col and grade_col):
            continue
        part = split_of(csv.name)
        rows = df[[name_col, grade_col]].dropna()
        rows.columns = ["name", "diagnosis"]
        rows["name"] = rows["name"].astype(str).str.strip()
        rows["part"] = part
        rows["src"] = [str(images.get((part, Path(n).stem)) or images.get(("", Path(n).stem)) or "") or None
                       for n in rows["name"]]
        frames.append(rows)
        print(f"  labels: {csv.relative_to(data_dir)} ({len(rows)} rows, part '{part or '-'}')")
    if not frames:
        sys.exit(f"No grading CSV found under {data_dir} (need columns like 'Image name' + 'Retinopathy grade').")
    df = pd.concat(frames, ignore_index=True)
    df["id_code"] = df["part"].str.cat(df["name"], sep="/").str.strip("/")
    df["diagnosis"] = df["diagnosis"].astype(int)
    missing = int(df["src"].isna().sum())
    if missing:
        print(f"  warning: {missing} labelled images not found; skipping them")
    df = df.dropna(subset=["src"]).drop_duplicates("src").reset_index(drop=True)
    return df[["id_code", "diagnosis", "src"]]


def predict_all(model, cfg, df, device, bs=16):
    import torch
    outs, batch, t0 = [], [], time.time()
    for k, src in enumerate(df["src"]):
        _, model_img = preprocess(load_rgb(src), cfg["img_size"], cfg["ben_graham_sigma"])
        batch.append(to_tensor(model_img, cfg["mean"], cfg["std"]))
        if len(batch) == bs or k == len(df) - 1:
            x = torch.cat(batch).to(device)
            with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
                out = model(x).float()
            outs.append((torch.softmax(out, 1) if cfg["head"] == "classification" else out).cpu().numpy())
            batch = []
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(df)} images ({time.time() - t0:.0f}s)")
    return np.concatenate(outs)


def to_grades(P, cfg):
    return P.argmax(1) if cfg["head"] == "classification" else np.digitize(P[:, 0], cfg["reg_thresholds"])


def refer_scores(P, cfg):
    return P[:, 2:].sum(1) if cfg["head"] == "classification" else P[:, 0]


def save(df, P, cfg, run, tag, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    Y, pred, rs = df["diagnosis"].values, to_grades(P, cfg), refer_scores(P, cfg)
    s, pc, cm = metrics.report(Y, pred, rs, cfg["refer_threshold"], f"{tag} external test, run {run}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    s.to_csv(fig_dir / f"external_{tag}_summary.csv")
    pc.to_csv(fig_dir / f"external_{tag}_per_class.csv", index=False)
    cm.to_csv(fig_dir / f"external_{tag}_confusion.csv")
    df.assign(pred=pred, refer_score=rs).drop(columns=["src"]).to_csv(fig_dir / f"external_{tag}_predictions.csv", index=False)

    cuts = [0.2, 0.3, 0.4, 0.5] if cfg["head"] == "classification" else [1.0, 1.25, 1.5]
    print("Refer cut-off   sensitivity   specificity   (information only: the cut-off is NOT changed on test data)")
    for th in cuts:
        se, sp = metrics._refer(Y, rs, th)
        print(f"     {th:.2f}          {se:.3f}         {sp:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm.values, display_labels=range(5)).plot(ax=ax[0], colorbar=False, cmap="Blues")
    ax[0].set_title(f"{tag}: 5 grades"); ax[0].set_xlabel("Predicted grade"); ax[0].set_ylabel("True grade")
    ConfusionMatrixDisplay.from_predictions(Y >= 2, rs >= cfg["refer_threshold"], display_labels=["no refer", "refer"],
                                            ax=ax[1], colorbar=False, cmap="Blues")
    ax[1].set_title(f"{tag}: refer / no refer (cut-off {cfg['refer_threshold']})")
    plt.tight_layout(); plt.savefig(fig_dir / f"external_{tag}_confusion.png", dpi=150); plt.close()
    return s


def compare_and_log(s, cfg, run, tag, n, fig_dir: Path):
    val = fig_dir / "val_summary.csv"
    if val.exists():
        side = pd.DataFrame({"APTOS validation": pd.read_csv(val, index_col=0)["value"],
                             f"{tag} (external)": s["value"]}).round(3)
        side["change"] = (side.iloc[:, 1] - side.iloc[:, 0]).round(3)
        print("\nSame model, same cut-off, two hospitals:")
        print(side.to_string())
        side.to_csv(fig_dir / f"external_{tag}_vs_validation.csv")
    log = ROOT / "reports" / "figures" / "external_tests.csv"
    row = {"run": run, "dataset": tag, "date": str(date.today()), "images": n,
           "refer_cutoff": cfg["refer_threshold"], **s["value"].round(4).to_dict()}
    table = pd.concat([pd.read_csv(log), pd.DataFrame([row])]) if log.exists() else pd.DataFrame([row])
    table.to_csv(log, index=False)
    print(f"\nSaved to {fig_dir} and added to reports/figures/external_tests.csv")


def main():
    ap = argparse.ArgumentParser(description="External test of a trained DiaSight model")
    ap.add_argument("--data", required=True, help="folder with the external dataset (CSV labels + images)")
    ap.add_argument("--model-dir", default="models", help="folder with best.pt + config.json")
    ap.add_argument("--tag", default="idrid", help="short name of the dataset, used in file names")
    args = ap.parse_args()

    import torch
    from diasight.model import load

    model_dir = ROOT / args.model_dir if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    data_dir = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    model, cfg, is_demo = load(model_dir)
    if is_demo:
        sys.exit(f"No trained model in {model_dir} (needs best.pt and config.json).")
    run = cfg.get("run", model_dir.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model: run '{run}', {cfg['model_name']} at {cfg['img_size']}px, {cfg['head']} head, "
          f"refer cut-off {cfg['refer_threshold']} (from config.json, not tuned here). Device: {device}")

    print(f"1. Reading {args.tag} from {data_dir}")
    df = find_labels(data_dir)
    print(f"   {len(df)} images; per grade: {np.bincount(df['diagnosis'], minlength=5).tolist()}")
    print("2. Predicting (same preprocessing as the app)")
    P = predict_all(model, cfg, df, device)
    print("3. Results")
    fig_dir = ROOT / "reports" / "figures" / run
    s = save(df, P, cfg, run, args.tag, fig_dir)
    compare_and_log(s, cfg, run, args.tag, len(df), fig_dir)


if __name__ == "__main__":
    main()