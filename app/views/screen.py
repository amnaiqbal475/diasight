"""Screening page: photo in, decision + heatmap + advice + PDF out."""
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st

import ui
from diasight import advice, gradcam, predict, quality, report
from diasight.config import GRADE_COLOURS, GRADE_NAMES, SAMPLES_DIR
from diasight.preprocess import load_rgb


def _samples() -> list[Path]:
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _sample_grade(p: Path):
    return int(p.name[5]) if p.name.startswith("grade") and p.name[5:6].isdigit() else None


def _pick_sample(name):
    st.session_state.sample = name
    st.session_state.upl_key = st.session_state.get("upl_key", 0) + 1  # empties the uploader


def _clear_sample():
    st.session_state.sample = None


def _reset():
    st.session_state.sample = None
    st.session_state.upl_key = st.session_state.get("upl_key", 0) + 1


def _analyse(model, cfg, key: str, img: np.ndarray) -> dict:
    """Quality check, grade and Grad-CAM with real per-stage timings.
    Cached per image so moving a slider never re-runs the model."""
    cache = st.session_state.setdefault("_analysis", {})
    if key not in cache:
        t = [time.perf_counter()]
        ok, msg, m = quality.check(img, cfg["quality"])
        t.append(time.perf_counter())
        out = {"ok": ok, "msg": msg, "m": m}
        if ok:
            res = predict.predict(model, cfg, img)
            t.append(time.perf_counter())
            heat = gradcam.heatmap(model, cfg, res)
            t.append(time.perf_counter())
            out.update(res=res, heat=heat)
        out["times"] = [b - a for a, b in zip(t, t[1:])]
        out["secs"] = t[-1] - t[0]
        cache.clear()
        cache[key] = out
    return cache[key]


def _colour(res) -> str:
    """Colour follows the decision, so the card never shows green next to 'Refer'."""
    c = GRADE_COLOURS[res.grade]
    if res.refer and c == "green":
        return "amber"
    if not res.refer and c != "green":
        return "amber"
    return c


def render(model, cfg, is_demo, stats):
    st.session_state.setdefault("sample", None)
    st.session_state.setdefault("upl_key", 0)
    samples = _samples()

    # The photo can come from an upload, a sample button, or ?sample=<file name> in the URL (handy for demos).
    qp = st.query_params.get("sample")
    if qp and st.session_state.get("_qp_used") != qp and any(p.name == qp for p in samples):
        st.session_state.sample, st.session_state._qp_used = qp, qp

    uploaded = st.session_state.get(f"upl_{st.session_state.upl_key}")
    sample = SAMPLES_DIR / st.session_state.sample if st.session_state.sample else None
    has_photo = uploaded is not None or sample is not None

    if not has_photo:
        hero_img = next((p for p in samples if _sample_grade(p) == 0), samples[0] if samples else None)
        ui.hero(hero_img, stats, cfg)
        ui.pipeline_strip(cfg)
    if is_demo:
        ui.html('<div class="ds-demo">&#9888;<div><b>Demo mode.</b> No trained model in <code>models/</code>, '
                'so grades come from an untrained network and mean nothing.</div></div>')

    left, right = st.columns([5, 7], gap="large")

    # ---------------- left: input ----------------
    with left:
        with st.container(key="photo_panel"):
            ui.html('<div class="ds-h"><span class="ds-num">01 / INPUT</span>Retina photo</div>'
                    '<div class="ds-sub">Fundus image from a retinal camera or a phone adapter.</div>')
            uploaded = st.file_uploader("Upload a fundus photo", type=["jpg", "jpeg", "png"],
                                        key=f"upl_{st.session_state.upl_key}", on_change=_clear_sample,
                                        label_visibility="collapsed")
            if samples:
                ui.html('<div class="ds-label">Sample library</div>')
                with st.container(key="gallery"):
                    cols = st.columns(len(samples))
                    for col, p in zip(cols, samples):
                        g = _sample_grade(p)
                        with col:
                            st.image(ui.thumb(str(p), 160), width="stretch")
                            active = st.session_state.sample == p.name and uploaded is None
                            st.button(f"G{g}" if g is not None else p.stem[:6], key=f"s_{p.name}",
                                      type="primary" if active else "secondary", on_click=_pick_sample,
                                      args=(p.name,), help=f"Grade {g}: {GRADE_NAMES[g]}" if g is not None else None)
            ui.html('<div class="ds-label">Patient record · optional</div>')
            with st.expander("Add patient details", icon=":material/person:"):
                patient = {"name": st.text_input("Name or ID", placeholder="e.g. Ali Ahmed or #1042"),
                           "age": st.text_input("Age", placeholder="e.g. 54"),
                           "years": st.text_input("Years with diabetes", placeholder="e.g. 10")}
                st.caption("Printed on the PDF only. Nothing is stored or sent anywhere.")

    if uploaded is not None:
        data, src_name = uploaded.getvalue(), uploaded.name
    elif sample is not None and sample.exists():
        data, src_name = sample.read_bytes(), sample.name
    else:
        data, src_name = None, None

    # ---------------- right: analysis ----------------
    with right:
        with st.container(key="result_panel"):
            if data is None:
                _empty_state()
            else:
                _result(model, cfg, is_demo, data, src_name, patient)


def _empty_state():
    ui.html("""
<div class="ds-h"><span class="ds-num">02 / ANALYSIS</span>Result</div>
<div class="ds-empty"><div class="blink">&gt; awaiting input</div>
<b>Load a retina photo to start</b><span>Upload your own or pick one from the sample library. Analysis takes about a second.</span></div>
<div class="ds-steps">
<div class="ds-step"><small>STEP 01</small><b>Quality gate</b><span>Blurry, dark or tiny photos are rejected before the model sees them.</span></div>
<div class="ds-step"><small>STEP 02</small><b>CNN grading</b><span>Grade 0 to 4 plus the probability of referable disease.</span></div>
<div class="ds-step"><small>STEP 03</small><b>Explainability</b><span>Grad-CAM shows which regions drove the decision.</span></div>
<div class="ds-step"><small>STEP 04</small><b>Report</b><span>English and Urdu advice and a one-page PDF for the patient.</span></div>
</div>""")


def _log(lines):
    ui.html('<div class="ds-log">' + "".join(f"<div>{ln}</div>" for ln in lines) + "</div>")


def _retake(out, cfg, img, src_name):
    m, q = out["m"], cfg["quality"]
    checks = [("RESOLUTION", f'{m["min_side"]}px', m["min_side"] >= quality.MIN_SIDE, f'min {quality.MIN_SIDE}px'),
              ("BRIGHTNESS", f'{m["brightness"]:.1f}', q["min_brightness"] <= m["brightness"] <= q["max_brightness"],
               f'range {q["min_brightness"]:.0f}–{q["max_brightness"]:.0f}'),
              ("SHARPNESS", f'{m["sharpness"]:.1f}', m["sharpness"] >= q["min_sharpness"], f'min {q["min_sharpness"]:.0f}')]
    _log([f'<span class="t">[{out["times"][0] * 1000:6.1f} ms]</span> load <span class="k">{ui.e(src_name[:40])}</span>',
          f'<span class="t">[{out["times"][0] * 1000:6.1f} ms]</span> quality gate … <span class="bad">REJECTED</span>',
          '<span class="t">[   --   ]</span> inference skipped: unreliable input'])
    rows = "".join(
        f'<div class="ds-check{"" if ok else " bad"}"><span>{"&#10003;" if ok else "&#10007;"} {name}</span>'
        f'<span><b>{val}</b> · {need}</span></div>' for name, val, ok, need in checks)
    reason = out["msg"].replace("Please retake: ", "").rstrip(".")
    ui.html(f"""
<div class="ds-retake"><span class="ds-tag">&#9888; Input rejected</span>
<div class="ds-decision">Retake the photo</div>
<div class="ds-follow">{ui.e(reason.capitalize())}. Check focus and lighting, then take it again.
No grade is given for a poor photo, because it would not be reliable.</div>
<div class="ds-checks">{rows}</div></div>""")
    ui.html('<div class="ds-label">Photo received</div>')
    st.image(img, width=320)


def _result(model, cfg, is_demo, data, src_name, patient):
    key = hashlib.sha1(data).hexdigest()
    img = load_rgb(io.BytesIO(data))
    with st.spinner("Running analysis…"):
        out = _analyse(model, cfg, key, img)

    head_l, head_r = st.columns([2, 1.2], vertical_alignment="center")
    head_l.markdown('<div class="ds-h" style="margin:0"><span class="ds-num">02 / ANALYSIS</span>Result</div>',
                    unsafe_allow_html=True)
    head_r.button("New scan", icon=":material/restart_alt:", on_click=_reset, width="stretch")

    if not out["ok"]:
        _retake(out, cfg, img, src_name)
        return

    res, heat, m = out["res"], out["heat"], out["m"]
    tq, tp, tg = (x * 1000 for x in out["times"])
    colour = _colour(res)
    q = cfg["quality"]
    _log([
        f'<span class="t">[{tq:7.1f} ms]</span> quality gate · sharpness <span class="k">{m["sharpness"]:.1f}</span> &ge; {q["min_sharpness"]:.0f}'
        f' · light <span class="k">{m["brightness"]:.0f}</span> · <span class="ok">PASS</span>',
        f'<span class="t">[{tp:7.1f} ms]</span> inference · <span class="k">{ui.e(cfg["model_name"])}</span>'
        f' · P(refer) <span class="k">{res.refer_score:.3f}</span> · grade <span class="k">{res.grade}</span>',
        f'<span class="t">[{tg:7.1f} ms]</span> grad-cam · target class <span class="k">{res.grade}</span>'
        f' · <span class="ok">DONE</span> · total {out["secs"]:.2f}s',
    ])

    ui.verdict(res, colour, advice.FOLLOW_UP[res.grade])
    ui.severity_scale(res.grade)
    scale = 1.0 if cfg["head"] == "classification" else 4.0
    ui.meter(res.refer_score, cfg["refer_threshold"], scale,
             "Referral probability · P(grade ≥ 2)" if scale == 1.0 else "Severity score (0 to 4)", colour)

    ui.html('<div class="ds-label" style="margin-top:1.3rem">Visual explanation</div>')
    strength = st.slider("Heatmap strength", 0, 100, 75, step=5, format="%d%%",
                         help="Blend between the plain photo (0%) and the Grad-CAM heatmap (100%).")
    a = strength / 100
    blend = (res.display_img.astype(np.float32) * (1 - a) + heat.astype(np.float32) * a).astype(np.uint8)
    size = f'{res.display_img.shape[1]}×{res.display_img.shape[0]}'
    ui.hud_pair(res.display_img, blend, "INPUT // FUNDUS", "GRAD-CAM // ATTENTION", size)

    if res.probs:
        ui.html('<div class="ds-label" style="margin-top:1.2rem">Class probabilities</div>')
        ui.prob_bars(res.probs, res.grade)

    en, ur = advice.get(res.grade)
    ui.html(f"""
<div class="ds-advice">
<div><span class="ds-lang">ADVICE · EN</span><p>{ui.e(en)}</p></div>
<div><span class="ds-lang">ADVICE · اردو</span><p class="ds-urdu" lang="ur">{ui.e(ur)}</p></div></div>""")

    pkey = (key, patient["name"], patient["age"], patient["years"])
    if st.session_state.get("_pdf_key") != pkey:
        model_info = f'{cfg.get("model_name", "")} @ {cfg.get("img_size", "")}px, run {cfg.get("run", "demo")}'
        st.session_state._pdf = report.build_pdf(res, colour, heat, patient, is_demo, quality=m, model_info=model_info)
        st.session_state._pdf_key = pkey
    st.download_button("Download PDF report", st.session_state._pdf,
                       file_name=f"diasight_report_{report.report_id(res)}.pdf", mime="application/pdf",
                       type="primary", icon=":material/download:", key="dl")

    with st.expander("Technical details", icon=":material/memory:"):
        d1, d2 = st.columns([1, 1.3], gap="medium")
        d1.image(res.model_img, caption="Network input after preprocessing (Ben Graham + circular mask)", width="stretch")
        with d2:
            ui.html(f"""
<div class="ds-kv">
<span>MODEL</span><b>{ui.e(cfg.get("model_name"))} · {cfg.get("img_size")}px · {ui.e(cfg.get("head"))} head</b>
<span>RUN</span><b>{ui.e(cfg.get("run", "demo"))} · best epoch {cfg.get("best_epoch", "-")}</b>
<span>VAL KAPPA</span><b>{cfg.get("val_kappa", 0):.3f}</b>
<span>THRESHOLD</span><b>{cfg["refer_threshold"]:.2f}</b>
<span>REFER SCORE</span><b>{res.refer_score:.4f}</b>
<span>SHARPNESS</span><b>{m["sharpness"]:.1f} (min {q["min_sharpness"]:.0f})</b>
<span>BRIGHTNESS</span><b>{m["brightness"]:.1f} (range {q["min_brightness"]:.0f}–{q["max_brightness"]:.0f})</b>
<span>REPORT ID</span><b>{report.report_id(res)}</b>
</div>""")
