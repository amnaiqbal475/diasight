"""DiaSight Streamlit app.  Run from the repo root:  streamlit run app/app.py"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))  # make `diasight` importable
sys.path.insert(0, str(APP_DIR))         # make `ui` and `views` importable

import streamlit as st  # noqa: E402

from diasight import model as model_lib  # noqa: E402
from diasight.config import ASSETS_DIR  # noqa: E402

st.set_page_config(page_title="DiaSight · Diabetic retinopathy screening", page_icon=str(ASSETS_DIR / "logo.png"),
                   layout="wide", initial_sidebar_state="collapsed")

import ui  # noqa: E402
from views import about, performance, screen  # noqa: E402


@st.cache_resource(show_spinner="Loading the model…")
def get_model():
    return model_lib.load()


model, cfg, is_demo = get_model()
stats = performance.headline_stats(cfg) if not is_demo else []



def _page(view):
    def run():
        view.render(model, cfg, is_demo, stats)
    run.__name__ = view.__name__.split(".")[-1]
    return run


pages = {
    "Screen": st.Page(_page(screen), title="Screen", url_path="screen", default=True),
    "Performance": st.Page(_page(performance), title="Performance",
                           url_path="performance"),
    "About": st.Page(_page(about), title="About", url_path="about"),
}
nav = st.navigation(list(pages.values()), position="hidden")
active = next((i for i, p in enumerate(pages.values(), 1) if p.url_path == nav.url_path), 1)
ui.inject_css(active)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
model_label = ("untrained network" if is_demo else
               f'{cfg["model_name"].replace("efficientnet_", "EffNet-").upper()} · {n_params:.1f}M · κ {cfg.get("val_kappa", 0):.3f}')
ui.topbar(pages, ASSETS_DIR / "logo.png", model_label, is_demo)
nav.run()
