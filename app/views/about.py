"""About page: how DiaSight works, data, safety and limits."""
import streamlit as st

import ui
from diasight import advice


def render(model, cfg, is_demo, stats):
    ui.html("""
<div class="ds-pagehead"><span class="ds-eyebrow">ABOUT · SYSTEM OVERVIEW</span>
<h2>Screening, not diagnosis</h2>
<p>Diabetic retinopathy causes no symptoms until sight is already at risk, and many people with diabetes never get an
eye exam. DiaSight is a course prototype that shows how a single retina photo could be triaged at a clinic: who can wait
for the routine re-screen, and who should see an eye doctor now.</p></div>""")

    ui.html("""
<div class="ds-flow">
<div><small>STEP 1</small><b>Photo in</b><span>Fundus photo from a retinal camera, JPG or PNG.</span></div>
<div><small>STEP 2</small><b>Quality gate</b><span>Resolution, brightness and blur limits set from real dataset photos.</span></div>
<div><small>STEP 3</small><b>Preprocess</b><span>Crop the black border, resize, Ben Graham contrast, circular mask.</span></div>
<div><small>STEP 4</small><b>Grade</b><span>EfficientNet gives grade 0 to 4 and the chance of grade 2 or higher.</span></div>
<div><small>STEP 5</small><b>Explain</b><span>Grad-CAM heatmap, advice in English and Urdu, one-page PDF.</span></div>
</div>""")

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="card_grades"):
        rows = "".join(f'<tr><td><b>{g}</b></td><td>{n}</td><td>{"Refer" if g >= 2 else "Re-screen"}</td>'
                       f'<td>{ui.e(advice.FOLLOW_UP[g])}</td></tr>'
                       for g, n in enumerate(["No DR", "Mild", "Moderate", "Severe", "Proliferative"]))
        ui.html(f'<div class="ds-h">The five grades</div><table class="ds-table"><thead><tr><th>Grade</th><th>Name</th>'
                f'<th>Decision</th><th>Follow-up</th></tr></thead><tbody>{rows}</tbody></table>')
    with c2, st.container(key="card_data"):
        ui.html(f"""
<div class="ds-h">Data and model</div>
<div class="ds-kv">
<span>Training</span><b>APTOS 2019 Blindness Detection (Kaggle), 80% stratified split</b>
<span>Validation</span><b>The other 20% of APTOS, never trained on</b>
<span>External test</span><b>IDRiD (Porwal et al., 2018), 516 photos from an Indian clinic</b>
<span>Model</span><b>{ui.e(cfg.get("model_name"))} at {cfg.get("img_size")}px, {ui.e(cfg.get("head"))} head</b>
<span>Refer rule</span><b>Refer when P(grade 2 or higher) is {cfg["refer_threshold"]:.2f} or more</b>
<span>Privacy</span><b>Photos and patient details stay in memory for this session only. Nothing is stored.</b>
</div>""")

    ui.html(f"""
<div class="ds-note" style="margin-top:1rem"><b>Safety.</b> {ui.e(advice.DISCLAIMER)} DiaSight was built for an AI/ML capstone
course and has not been clinically validated. The Urdu advice should be reviewed by a native speaker before any use with
patients.</div>""")
    ui.footer("Team: Zainab and Amna · Urdu font: Noto Naskh Arabic / Noto Nastaliq Urdu (SIL Open Font License)")
