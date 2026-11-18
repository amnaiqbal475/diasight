"""Model performance page: validation vs external test, per grade, confusion matrices, experiment history.
Everything is read from reports/figures/, which scripts/train.py and scripts/evaluate.py write."""
from pathlib import Path

import pandas as pd
import streamlit as st

import ui
from diasight.config import GRADE_NAMES, ROOT

FIG = ROOT / "reports" / "figures"
HEADLINE = ["Refer sensitivity", "Refer specificity", "Refer AUC", "Quadratic weighted kappa"]
EXPLAIN = {"Refer sensitivity": "of eyes that need a doctor are caught",
           "Refer specificity": "of healthy eyes are correctly not referred",
           "Refer AUC": "how well the score separates refer from no-refer",
           "Quadratic weighted kappa": "agreement with the expert grade (0 to 1)"}


def _csv(path: Path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def load(run: str) -> dict:
    d = FIG / run
    return {"val": _csv(d / "val_summary.csv", index_col=0),
            "ext": _csv(d / "external_idrid_summary.csv", index_col=0),
            "val_pc": _csv(d / "val_per_class.csv"),
            "ext_pc": _csv(d / "external_idrid_per_class.csv"),
            "val_cm": _csv(d / "val_confusion.csv", index_col=0),
            "ext_cm": _csv(d / "external_idrid_confusion.csv", index_col=0),
            "runs": _csv(FIG / "experiments.csv"),
            "ext_runs": _csv(FIG / "external_tests.csv")}


def headline_stats(cfg) -> list[tuple[str, str]]:
    """Numbers for the hero on the screening page."""
    r = load(cfg.get("run", ""))
    out = []
    if r["val"] is not None:
        out.append(("Refer AUC, validation", f'{r["val"].loc["Refer AUC", "value"]:.2f}'))
        out.append(("Refer sensitivity", f'{r["val"].loc["Refer sensitivity", "value"] * 100:.0f}%'))
    if r["ext"] is not None:
        out.append(("AUC on an unseen hospital", f'{r["ext"].loc["Refer AUC", "value"]:.2f}'))
    out.append(("Advice languages", "EN + اردو"))
    return out


def _fmt(metric, v):
    return f"{v:.2f}" if metric in ("Refer AUC", "Quadratic weighted kappa") else f"{v * 100:.1f}%"


def _kpis(val, ext):
    tiles = []
    for k in HEADLINE:
        v, lo, hi = val.loc[k, "value"], val.loc[k, "95% CI low"], val.loc[k, "95% CI high"]
        ext_html = ""
        if ext is not None and k in ext.index:
            ext_html = f'<div class="ext"><span>Unseen hospital (IDRiD):</span> <b>{_fmt(k, ext.loc[k, "value"])}</b></div>'
        tiles.append(f'<div class="ds-kpi"><div class="k">{k}</div><div class="v">{_fmt(k, v)}</div>'
                     f'<div class="ci">95% CI {_fmt(k, lo)} to {_fmt(k, hi)} · {EXPLAIN[k]}</div>{ext_html}</div>')
    ui.html(f'<div class="ds-kpis">{"".join(tiles)}</div>')


def _compare(val, ext):
    metrics = [m for m in val.index if m in ext.index]
    rows = []
    for m in metrics:
        bars = ""
        for key, df, name in (("val", val, "APTOS validation"), ("ext", ext, "IDRiD external")):
            v, lo, hi = df.loc[m, "value"], df.loc[m, "95% CI low"], df.loc[m, "95% CI high"]
            bars += (f'<div class="ds-cbar" title="{name} · {m}: {v:.3f} (95% CI {lo:.3f} to {hi:.3f})">'
                     f'<i style="width:calc({v * 100:.1f}% - 3rem);background:{ui.SERIES[key]}"></i>{v:.2f}</div>')
        rows.append(f'<div class="ds-cmp-row"><span>{m}</span><div class="ds-cmp-bars">{bars}</div></div>')
    ui.html(f"""
<div class="ds-legend2"><span><i style="background:{ui.SERIES['val']}"></i>APTOS validation (same source as training)</span>
<span><i style="background:{ui.SERIES['ext']}"></i>IDRiD external test (new hospital and camera)</span></div>
<div class="ds-cmp">{"".join(rows)}</div>
<div class="ds-axis" style="margin-top:.5rem"><span></span><div><span>0</span><span>0.5</span><span>1.0</span></div></div>""")


def _per_class(val_pc, ext_pc):
    rows = ""
    for i, r in val_pc.iterrows():
        e_s = ext_pc.loc[i, "sensitivity"] if ext_pc is not None else None
        flag = '<span class="ds-pill warn">weak</span>' if r["sensitivity"] < 0.5 or (e_s is not None and e_s < 0.4) else ""
        ext_cell = (f'<div class="ds-cellbar"><i style="width:{e_s * 80:.0f}px;background:{ui.SERIES["ext"]}"></i>{e_s:.2f}</div>'
                    if e_s is not None else "-")
        rows += (f'<tr><td>{ui.e(r["grade"])} {flag}</td><td>{int(r["n"])}</td>'
                 f'<td><div class="ds-cellbar"><i style="width:{r["sensitivity"] * 80:.0f}px;background:{ui.SERIES["val"]}"></i>'
                 f'{r["sensitivity"]:.2f}</div></td><td>{ext_cell}</td><td>{r["specificity"]:.2f}</td><td>{r["F1"]:.2f}</td></tr>')
    ui.html(f"""
<table class="ds-table"><thead><tr><th>Grade</th><th>n (val)</th><th>Sensitivity · validation</th>
<th>Sensitivity · IDRiD</th><th>Specificity</th><th>F1</th></tr></thead><tbody>{rows}</tbody></table>""")


def _confusion(cm: pd.DataFrame, title: str):
    vals = cm.values
    row_tot = vals.sum(axis=1, keepdims=True).clip(min=1)
    share = vals / row_tot
    cells = '<div class="h"></div>' + "".join(f'<div class="h">pred {g}</div>' for g in range(5))
    for t in range(5):
        cells += f'<div class="h" style="justify-content:flex-end">true {t}</div>'
        for p in range(5):
            s = share[t, p]
            bg = f"rgba(34,211,238,{0.05 + s * 0.85:.2f})"
            fg = "#03141A" if s > 0.55 else "#CBD5E1"
            border = "box-shadow:inset 0 0 0 1.5px #E2E8F0;" if t == p else ""
            cells += (f'<div style="background:{bg};color:{fg};{border}" '
                      f'title="True grade {t}, predicted {p}: {vals[t, p]} images ({s * 100:.0f}% of row)">{vals[t, p]}</div>')
    ui.html(f'<div class="ds-label" style="margin-top:0">{title}</div><div class="ds-cm">{cells}</div>'
            f'<div class="ds-cmcap">Rows are the expert grade, shading is the share of that row. Outlined cells are correct.</div>')


def _history(runs: pd.DataFrame, active: str, ext_runs):
    ext = {} if ext_runs is None else ext_runs.drop_duplicates("run", keep="last").set_index("run")["Refer AUC"].to_dict()
    best = runs["val_kappa"].max()
    rows = ""
    for _, r in runs.iterrows():
        cls = "active" if r["run"] == active else ""
        tag = ' <span class="ds-pill">in use</span>' if r["run"] == active else ""
        e_auc = f'{ext[r["run"]]:.3f}' if r["run"] in ext else "-"
        rows += (f'<tr class="{cls}"><td>{ui.e(r["run"])}{tag}</td><td>{ui.e(r["model"])} @ {r["size"]}</td>'
                 f'<td>{ui.e(r["head"])}, weights {ui.e(r["class_weights"])}</td>'
                 f'<td><div class="ds-cellbar"><i style="width:{r["val_kappa"] / best * 90:.0f}px"></i>{r["val_kappa"]:.3f}</div></td>'
                 f'<td>{r["refer_sensitivity"]:.3f}</td><td>{r["refer_specificity"]:.3f}</td><td>{r["refer_auc"]:.3f}</td>'
                 f'<td>{e_auc}</td></tr>')
    ui.html(f"""
<table class="ds-table"><thead><tr><th>Run</th><th>Backbone</th><th>Head</th><th>Kappa (val)</th><th>Refer sens.</th>
<th>Refer spec.</th><th>Refer AUC</th><th>AUC on IDRiD</th></tr></thead><tbody>{rows}</tbody></table>""")


def render(model, cfg, is_demo, stats):
    run = cfg.get("run", "")
    ui.html(f"""
<div class="ds-pagehead"><span class="ds-eyebrow">MODEL CARD · VALIDATION REPORT</span>
<h2>How well does DiaSight work?</h2>
<p>Results for the model in use (<b>{ui.e(run or "none")}</b>, {ui.e(cfg.get("model_name"))} at {cfg.get("img_size")}px).
Validation uses 20% of APTOS 2019 held out from training. IDRiD is a separate Indian hospital dataset the model
never saw, which is the honest estimate for a new clinic. All intervals are bootstrap 95% CIs.</p></div>""")
    r = load(run)
    if r["val"] is None:
        st.info("No evaluation results found for this model yet. Run scripts/train.py, then scripts/evaluate.py.")
        return

    _kpis(r["val"], r["ext"])

    if r["ext"] is not None:
        with st.container(key="card_cmp"):
            auc = r["ext"].loc["Refer AUC", "value"]
            ui.html('<div class="ds-h">Same source vs. a new hospital</div>'
                    '<div class="ds-sub">Scores drop on IDRiD, as expected when the camera and population change. '
                    f'Referral AUC on the unseen hospital is {auc:.2f}.</div>')
            _compare(r["val"], r["ext"])

    c1, c2 = st.columns(2, gap="medium")
    if r["val_cm"] is not None:
        with c1, st.container(key="card_cm1"):
            _confusion(r["val_cm"], f"Confusion matrix · APTOS validation (n={int(r['val_cm'].values.sum())})")
    if r["ext_cm"] is not None:
        with c2, st.container(key="card_cm2"):
            _confusion(r["ext_cm"], f"Confusion matrix · IDRiD external (n={int(r['ext_cm'].values.sum())})")

    if r["val_pc"] is not None:
        with st.container(key="card_pc"):
            ui.html('<div class="ds-h">Per grade</div><div class="ds-sub">One grade against the rest. '
                    '"Weak" marks grades the model often misses.</div>')
            _per_class(r["val_pc"], r["ext_pc"])

    ui.html("""
<div class="ds-note" style="margin:1rem 0"><b>Known limitations.</b> Grade 3 (severe) is usually called grade 2 or 4,
so its own sensitivity is low; for referral this matters less, because 2, 3 and 4 are all referred. Mild disease
(grade 1) is often called grade 2. Only retinal-camera photos were tested (APTOS, IDRiD); phone-adapter photos were not.</div>""")

    if r["runs"] is not None:
        with st.container(key="card_hist"):
            ui.html('<div class="ds-h">Experiment history</div><div class="ds-sub">Every training run, '
                    'scored on the same validation split. The run in use is highlighted.</div>')
            _history(r["runs"], run, r["ext_runs"])
    ui.footer(f"Figures from reports/figures/{run}/")
