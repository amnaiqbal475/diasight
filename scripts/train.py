"""Train the DiaSight baseline on your own GPU.

Run from the repo root (the folder with app/, diasight/, models/):
    python scripts/train.py --data data/aptos224

Works with the 224x224 APTOS copy (train.csv + colored_images/<grade>/) and with the
original Kaggle download (train.csv + train_images/). Writes:
    models/best.pt, models/config.json      -> the app loads these
    app/samples/grade0_...png ... grade4_   -> demo images (validation set, never trained on)
    reports/figures/*                       -> metrics tables, confusion matrix, training history
    data/splits/train.csv, val.csv          -> the exact split (seed 42)
"""
import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diasight.config import DEFAULT_CONFIG, SAMPLES_DIR, SEED  # noqa: E402
from diasight.preprocess import load_rgb, preprocess  # noqa: E402
from diasight import quality  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg"}


def find_dataset(data_dir: Path) -> pd.DataFrame:
    """Return a table with id_code, diagnosis, src (path to the original image)."""
    csvs = sorted(data_dir.rglob("train.csv"))
    if not csvs:
        sys.exit(f"No train.csv found under {data_dir}. Check the --data path.")
    df = pd.read_csv(csvs[0])
    if not {"id_code", "diagnosis"} <= set(df.columns):
        sys.exit(f"{csvs[0]} needs the columns id_code and diagnosis, found {list(df.columns)}")
    index = {p.stem: p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    df["src"] = df["id_code"].map(lambda i: str(index[i]) if i in index else None)
    missing = df["src"].isna().sum()
    if missing:
        print(f"  warning: {missing} ids in train.csv have no image file; skipping them")
        df = df.dropna(subset=["src"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df.reset_index(drop=True)


def build_cache(df, cache_dir: Path, size: int, sigma: float) -> pd.DataFrame:
    """Preprocess every image once (crop, resize, Ben Graham, mask) and save it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    t0 = time.time()
    for k, (idc, src) in enumerate(zip(df["id_code"], df["src"])):
        out = cache_dir / f"{idc}.png"
        if not out.exists():
            _, model_img = preprocess(load_rgb(src), size, sigma)
            cv2.imwrite(str(out), cv2.cvtColor(model_img, cv2.COLOR_RGB2BGR))
        paths.append(str(out))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(df)} images ({time.time() - t0:.0f}s)")
    return df.assign(path=paths)


def split(df, out_dir: Path):
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(df, test_size=0.2, stratify=df["diagnosis"], random_state=SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out_dir / "train.csv", index=False)
    va.to_csv(out_dir / "val.csv", index=False)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def train(args, tr, va, cfg):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import cohen_kappa_score
    from diasight.dataset import RetinaDataset, train_augs
    from diasight.model import build_model

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                  "  (no GPU found: this will be slow)"))
    amp = device == "cuda"

    head = cfg["head"]
    train_dl = DataLoader(RetinaDataset(tr, cfg, train_augs(args.size), head), args.bs, shuffle=True,
                          num_workers=args.workers, drop_last=True, pin_memory=amp,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(RetinaDataset(va, cfg, None, head), 64, num_workers=args.workers, pin_memory=amp,
                        persistent_workers=args.workers > 0)

    model = build_model(cfg["model_name"], head, pretrained=True).to(device)
    w = class_weights(tr["diagnosis"], args.weights)
    print(f"Class weights ({args.weights}):", np.round(w, 2).tolist())
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    if head == "classification":
        crit = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        smooth = torch.nn.SmoothL1Loss(reduction="none")
        crit = lambda out, y: (smooth(out[:, 0], y) * weights[y.long()]).mean()  # noqa: E731
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total, warm = args.epochs * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def evaluate():
        model.eval(); P, Y = [], []
        with torch.no_grad(), torch.autocast(device, enabled=amp):
            for x, y in val_dl:
                out = model(x.to(device, non_blocking=True)).float()
                P.append((torch.softmax(out, 1) if head == "classification" else out).cpu()); Y.append(y)
        return torch.cat(P).numpy(), torch.cat(Y).numpy().astype(int)

    best_k, best_P, stale, history, best_epoch = -1.0, None, 0, [], 0
    ckpt = ROOT / "models" / "best.pt"
    for ep in range(args.epochs):
        model.train(); t0, run = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        P, Y = evaluate()
        k = cohen_kappa_score(Y, to_grades(P, cfg), weights="quadratic")
        ref_t = Y >= 2
        sens = ((refer_scores(P, cfg) >= cfg["refer_threshold"]) & ref_t).sum() / ref_t.sum()
        history.append({"epoch": ep + 1, "train_loss": run / len(train_dl), "val_kappa": k, "val_refer_sensitivity": sens})
        mark = ""
        if k > best_k:
            best_k, best_P, stale, mark = k, P, 0, "  <- best, saved"
            best_epoch = ep + 1
            torch.save({"model": model.state_dict()}, ckpt)
        else:
            stale += 1
        print(f"epoch {ep + 1:>2}/{args.epochs}  loss {run / len(train_dl):.3f}  val kappa {k:.3f}  "
              f"refer sensitivity {sens:.3f}  ({time.time() - t0:.0f}s){mark}")
        if stale >= args.patience:
            print(f"No kappa improvement for {args.patience} epochs: stopping early.")
            break
    return best_k, best_P, history, best_epoch


def class_weights(labels, mode: str):
    """inverse = n / (5 * count); sqrt = its square root (gentler); none = all 1."""
    counts = np.bincount(labels, minlength=5).astype(float)
    inv = len(labels) / (5 * counts)
    return {"inverse": inv, "sqrt": np.sqrt(inv), "none": np.ones(5)}[mode]


def to_grades(P, cfg):
    if cfg["head"] == "classification":
        return P.argmax(1)
    return np.digitize(P[:, 0], cfg["reg_thresholds"])


def refer_scores(P, cfg):
    """classification: P(grade >= 2); regression: the raw predicted grade."""
    return P[:, 2:].sum(1) if cfg["head"] == "classification" else P[:, 0]


def save_results(va, P, history, cfg, best_k, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from diasight import metrics

    fig_dir.mkdir(parents=True, exist_ok=True)
    Y, pred, refer_score = va["diagnosis"].values, to_grades(P, cfg), refer_scores(P, cfg)
    s, pc, cm = metrics.report(Y, pred, refer_score, cfg["refer_threshold"], "APTOS validation (best epoch)")
    s.to_csv(fig_dir / "val_summary.csv"); pc.to_csv(fig_dir / "val_per_class.csv", index=False)
    cm.to_csv(fig_dir / "val_confusion.csv"); pd.DataFrame(history).to_csv(fig_dir / "train_history.csv", index=False)

    print("Refer cut-off   sensitivity   specificity   (lower cut-off = more referrals)")
    cuts = [0.2, 0.3, 0.4, 0.5] if cfg["head"] == "classification" else [1.0, 1.25, 1.5]
    for th in cuts:
        se, sp = metrics._refer(Y, refer_score, th)
        print(f"     {th:.2f}          {se:.3f}         {sp:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm.values, display_labels=range(5)).plot(ax=ax[0], colorbar=False, cmap="Blues")
    ax[0].set_title("APTOS validation: 5 grades"); ax[0].set_xlabel("Predicted grade"); ax[0].set_ylabel("True grade")
    ConfusionMatrixDisplay.from_predictions(Y >= 2, refer_score >= cfg["refer_threshold"],
                                            display_labels=["no refer", "refer"], ax=ax[1], colorbar=False, cmap="Blues")
    ax[1].set_title(f"Refer / no refer (cut-off {cfg['refer_threshold']})")
    plt.tight_layout(); plt.savefig(fig_dir / "val_confusion.png", dpi=150); plt.close()
    print(f"\nTables and figures saved to {fig_dir}")
    return s


def log_experiment(path: Path, name, args, best_ep, headline):
    """One row per run, so the report's experiment table builds itself."""
    row = {"run": name, "date": str(date.today()), "model": args.model, "size": args.size, "head": args.head,
           "class_weights": args.weights, "epochs": args.epochs, "lr": args.lr, "best_epoch": best_ep,
           "val_kappa": headline["Quadratic weighted kappa"],
           "refer_sensitivity": headline["Refer sensitivity"], "refer_specificity": headline["Refer specificity"],
           "refer_auc": headline["Refer AUC"], "macro_f1": headline["Macro-F1 (5 grades)"]}
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    table = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    print("\nAll experiments so far (reports/figures/experiments.csv):")
    print(table[["run", "best_epoch", "val_kappa", "refer_sensitivity", "refer_specificity", "refer_auc"]]
          .round(3).to_string(index=False))


def pick_samples(va, P, cfg):
    """One validation image per grade, correctly predicted and passing the quality check."""
    for old in SAMPLES_DIR.glob("grade*"):
        old.unlink()
    va = va.assign(pred=to_grades(P, cfg))
    for g in range(5):
        rows = va[va["diagnosis"] == g]
        rows = pd.concat([rows[rows["pred"] == g], rows[rows["pred"] != g]])  # correct ones first
        for _, r in rows.iterrows():
            if quality.check(load_rgb(r["src"]), cfg["quality"])[0]:
                shutil.copy(r["src"], SAMPLES_DIR / f"grade{g}_{r['id_code']}{Path(r['src']).suffix}")
                break
    print("Sample images:", sorted(p.name for p in SAMPLES_DIR.glob("grade*")))


def main():
    ap = argparse.ArgumentParser(description="Train the DiaSight baseline model")
    ap.add_argument("--data", required=True, help="folder containing train.csv and the images")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--head", choices=["classification", "regression"], default="classification")
    ap.add_argument("--weights", choices=["inverse", "sqrt", "none"], default="inverse",
                    help="class weights in the loss: inverse frequency, its square root, or none")
    ap.add_argument("--name", default=None, help="run name; outputs are also kept in models/runs/<n>/")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prepare-only", action="store_true", help="find data, cache and split, then stop")
    args = ap.parse_args()

    data_dir = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    print(f"1. Reading dataset from {data_dir}")
    df = find_dataset(data_dir)
    print(f"   {len(df)} labelled images; per grade: {np.bincount(df['diagnosis'], minlength=5).tolist()}")

    print(f"2. Preprocessing to {args.size}px (cached, so this is slow only the first time)")
    df = build_cache(df, ROOT / "data" / f"cache{args.size}", args.size, DEFAULT_CONFIG["ben_graham_sigma"])

    print("3. Quality check statistics on dataset images")
    stats = pd.DataFrame([quality.measure(load_rgb(p)) for p in df.sample(min(300, len(df)), random_state=SEED)["src"]])
    q = stats[["sharpness", "brightness"]].quantile([0.02, 0.5, 0.98]).round(1)
    print(q.to_string())

    print("4. Stratified 80/20 split, seed 42")
    tr, va = split(df, ROOT / "data" / "splits")
    print(f"   {len(tr)} train / {len(va)} validation")
    if args.prepare_only:
        print("\n--prepare-only: stopping before training."); return

    name = args.name or f"{args.model}_{args.size}_{args.head}_{args.weights}_{args.epochs}ep"
    cfg = {**DEFAULT_CONFIG, "model_name": args.model, "img_size": args.size, "head": args.head,
           "refer_threshold": 0.5 if args.head == "classification" else 1.5,
           "trained_on": f"{data_dir.name}, 80% stratified split, seed {SEED}", "trained": str(date.today()),
           "run": name, "epochs": args.epochs, "lr": args.lr, "class_weights": args.weights,
           "quality_stats": q.to_dict(),
           # Reject photos worse than the 2nd percentile of real dataset photos (rounded down).
           # max_brightness stays at the fixed "washed out" limit: the dataset's 98th percentile (~130)
           # would wrongly reject brighter cameras, e.g. IDRiD.
           "quality": {"min_sharpness": float(np.floor(q.loc[0.02, "sharpness"])),
                       "min_brightness": float(np.floor(q.loc[0.02, "brightness"])),
                       "max_brightness": DEFAULT_CONFIG["quality"]["max_brightness"]}}
    print(f"5. Training run '{name}': {args.model} at {args.size}px, {args.head} head, "
          f"{args.weights} class weights, up to {args.epochs} epochs, lr {args.lr}")
    best_k, P, history, best_epoch = train(args, tr, va, cfg)
    cfg["val_kappa"] = round(float(best_k), 4)
    cfg["best_epoch"] = best_epoch
    (ROOT / "models" / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"\nBest val kappa {best_k:.3f} at epoch {best_epoch}. Saved models/best.pt and models/config.json")

    print("6. Evaluating the best epoch")
    fig_dir = ROOT / "reports" / "figures" / name
    summary = save_results(va, P, history, cfg, best_k, fig_dir)
    print("7. Picking demo samples")
    pick_samples(va, P, cfg)

    run_dir = ROOT / "models" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "models" / "best.pt", run_dir / "best.pt")
    shutil.copy(ROOT / "models" / "config.json", run_dir / "config.json")
    log_experiment(ROOT / "reports" / "figures" / "experiments.csv", name, args, best_epoch,
                   summary["value"].to_dict())
    print(f"\nRun kept in models/runs/{name}/; figures in reports/figures/{name}/")
    print("Done. Restart the app (streamlit run app/app.py) to use this model.")


if __name__ == "__main__":
    main()