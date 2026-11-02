"""DiaSight Streamlit app.  Run from the repo root:  streamlit run app/app.py"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `diasight` importable

import streamlit as st

from diasight import advice, gradcam, model as model_lib, predict, quality, report
from diasight.config import ASSETS_DIR, GRADE_COLOURS, SAMPLES_DIR
from diasight.preprocess import load_rgb

st.set_page_config(page_title="DiaSight", page_icon="👁️", layout="wide")

# ---------- look and feel ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
:root { --ink:#1F3A5F; --teal:#0F7C78; --line:#D6E1E0; --muted:#5A6B7D;
        --green:#24804F; --amber:#B07A0C; --red:#B3372F; }
.stApp, .stMarkdown, p, h1, h2, h3, h4, label, input, textarea, button p, li {
  font-family:'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; }
header[data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none; }
[data-testid="stMainBlockContainer"] { max-width:1160px; padding-top:1.75rem; padding-bottom:3rem; }

.ds-top { display:flex; align-items:center; gap:14px; padding-bottom:1rem;
          border-bottom:1px solid var(--line); margin-bottom:1.25rem; }
.ds-top img { height:46px; }
.ds-name { font-size:1.55rem; font-weight:600; color:var(--ink); line-height:1.1; }
.ds-tag { color:var(--muted); font-size:.95rem; }
.ds-disclaimer { margin-left:auto; color:var(--muted); font-size:.85rem; text-align:right; max-width:260px; }
.ds-demo { background:#FFF6E0; border:1px solid #F0D79A; color:#7A5508; border-radius:8px;
           padding:.55rem .8rem; font-size:.9rem; margin-bottom:1rem; }

.st-key-photo_panel, .st-key-result_panel { background:#fff; border:1px solid var(--line);
  border-radius:14px; padding:1.3rem 1.4rem 1.4rem; }
.ds-h { font-size:1.05rem; font-weight:600; color:var(--ink); margin:0 0 .6rem; }

.ds-steps { list-style:none; padding:0; margin:.4rem 0 0; counter-reset:s; }
.ds-steps li { counter-increment:s; display:flex; gap:12px; padding:.7rem 0; border-top:1px solid var(--line);
               color:var(--ink); }
.ds-steps li:first-child { border-top:none; }
.ds-steps li::before { content:counter(s); flex:0 0 26px; height:26px; border-radius:50%;
  border:1.5px solid var(--teal); color:var(--teal); font-weight:600; font-size:.85rem;
  display:flex; align-items:center; justify-content:center; }
.ds-steps span { display:block; color:var(--muted); font-size:.88rem; }

.ds-verdict { border-left:6px solid var(--c); background:var(--bg); border-radius:10px;
              padding:1rem 1.2rem 1.1rem; margin-bottom:1rem; }
.ds-decision { font-size:1.6rem; font-weight:600; color:var(--c); line-height:1.2; }
.ds-grade { display:flex; align-items:baseline; gap:10px; margin-top:.35rem; color:var(--ink); }
.ds-grade b { font-size:2.4rem; font-weight:600; line-height:1; }
.ds-grade span { color:var(--muted); font-size:.95rem; }
.ds-bar { position:relative; height:8px; background:#fff; border-radius:4px; margin:.9rem 0 .35rem;
          border:1px solid var(--line); }
.ds-bar i { position:absolute; left:0; top:0; bottom:0; background:var(--c); border-radius:4px; }
.ds-bar em { position:absolute; top:-5px; bottom:-5px; width:2px; background:var(--ink); }
.ds-barlabel { font-size:.82rem; color:var(--muted); }

.ds-advice { border-top:1px solid var(--line); margin-top:1rem; padding-top:.9rem; }
.ds-advice p { margin:.2rem 0; color:var(--ink); font-size:1.02rem; }
.ds-urdu { font-family:'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
           direction:rtl; text-align:right; font-size:1.2rem !important; line-height:2.3; }

.ds-retake { border-left:6px solid var(--amber); background:#FFF6E0; border-radius:10px; padding:1rem 1.2rem; }
.ds-retake b { color:#7A5508; font-size:1.25rem; }
.ds-retake p { color:var(--ink); margin:.3rem 0 0; }
.ds-retake small { color:var(--muted); }
.st-key-dl button { width:100%; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PALETTE = {"green": ("#24804F", "#EAF5EE"), "amber": ("#B07A0C", "#FFF6E0"), "red": ("#B3372F", "#FBECEA")}


@st.cache_resource
def get_model():
    return model_lib.load()


model, cfg, is_demo = get_model()

# ---------- top bar ----------
logo_path = ASSETS_DIR / "logo.png"
logo_html = ""
if logo_path.exists():
    logo_html = f'<img src="data:image/png;base64,{base64.b64encode(logo_path.read_bytes()).decode()}" alt="DiaSight logo">'
st.markdown(f"""
<div class="ds-top">{logo_html}
  <div><div class="ds-name">DiaSight</div><div class="ds-tag">Catch it before it steals sight.</div></div>
  <div class="ds-disclaimer">{advice.DISCLAIMER}</div>
</div>""", unsafe_allow_html=True)
if is_demo:
    st.markdown('<div class="ds-demo"><b>Demo mode.</b> No trained model in <code>models/</code> yet, '
                'so the grade below comes from an untrained network and means nothing.</div>',
                unsafe_allow_html=True)

left, right = st.columns([5, 7], gap="large")

# ---------- left: photo and patient ----------
with left:
    with st.container(key="photo_panel"):
        st.markdown('<div class="ds-h">Retina photo</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a fundus photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
        samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        choice = st.selectbox("Or use a sample image", ["None"] + [p.name for p in samples],
                              disabled=not samples,
                              help=None if samples else "Sample images appear here after training.")
        with st.expander("Patient details for the report (optional)"):
            patient = {"name": st.text_input("Name or ID"),
                       "age": st.text_input("Age"),
                       "years": st.text_input("Years with diabetes")}
            st.caption("Used only in the PDF. Nothing is saved.")

source = uploaded if uploaded is not None else (SAMPLES_DIR / choice if choice != "None" else None)

# ---------- right: result ----------
with right:
    with st.container(key="result_panel"):
        if source is None:
            st.markdown("""
<div class="ds-h">How the screening works</div>
<ol class="ds-steps">
  <li><div>Add a retina photo<span>Taken with a retinal camera or a phone attachment.</span></div></li>
  <li><div>Quality check<span>Blurry or dark photos are sent back for a retake.</span></div></li>
  <li><div>Grade and decision<span>Grade 0 to 4. Grade 2 or higher means refer to an eye doctor.</span></div></li>
  <li><div>Advice and report<span>English and Urdu advice, and a one-page PDF to hand to the patient.</span></div></li>
</ol>""", unsafe_allow_html=True)
        else:
            img = load_rgb(source)
            ok, msg, m = quality.check(img, cfg["quality"])
            if not ok:
                st.markdown(f"""
<div class="ds-retake"><b>Retake the photo</b>
<p>{msg.replace('Please retake: ', '').capitalize()} Check focus and lighting, then take it again.</p>
<small>Sharpness {m['sharpness']:.1f}, brightness {m['brightness']:.1f}</small></div>""",
                            unsafe_allow_html=True)
                st.image(img, width=280)
            else:
                with st.spinner("Reading the retina…"):
                    res = predict.predict(model, cfg, img)
                    heat = gradcam.heatmap(model, cfg, res)
                colour = GRADE_COLOURS[res.grade]
                c, bg = PALETTE[colour]
                decision = "Refer to an eye doctor" if res.refer else "No referral needed"
                scale = 1.0 if cfg["head"] == "classification" else 4.0  # regression score runs 0 to 4
                score = max(0.0, min(scale, res.refer_score))
                cut = max(0.0, min(scale, cfg["refer_threshold"]))
                score_name = "Refer score" if scale == 1.0 else "Severity score (0 to 4)"
                st.markdown(f"""
<div class="ds-verdict" style="--c:{c};--bg:{bg}">
  <div class="ds-decision">{decision}</div>
  <div class="ds-grade"><b>{res.grade}</b><div>{res.grade_name}<span> diabetic retinopathy, grade {res.grade} of 4</span></div></div>
  <div class="ds-bar"><i style="width:{score/scale*100:.0f}%"></i><em style="left:{cut/scale*100:.0f}%"></em></div>
  <div class="ds-barlabel">{score_name} {score:.2f}. Refers at {cut:.2f} or above.</div>
</div>""", unsafe_allow_html=True)

                i1, i2 = st.columns(2)
                i1.image(res.display_img, caption="Retina photo", width="stretch")
                i2.image(heat, caption="Where the model looked", width="stretch")

                en, ur = advice.get(res.grade)
                st.markdown(f"""
<div class="ds-advice"><div class="ds-h">Advice</div>
<p>{en}</p><p class="ds-urdu" lang="ur">{ur}</p></div>""", unsafe_allow_html=True)

                pdf_bytes = report.build_pdf(res, colour, heat, patient, is_demo)
                st.download_button("Download PDF report", pdf_bytes, file_name="diasight_report.pdf",
                                   mime="application/pdf", type="primary", key="dl")

                with st.expander("Details for clinicians"):
                    d1, d2 = st.columns([1, 1])
                    d1.image(res.model_img, caption="What the model sees after preprocessing", width="stretch")
                    if res.probs:
                        d2.bar_chart({"Probability": {f"Grade {g}": p for g, p in enumerate(res.probs)}},
                                     color="#0F7C78")