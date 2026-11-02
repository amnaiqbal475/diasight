"""One-page PDF screening report (fpdf2). Urdu needs the bundled Noto Naskh font + uharfbuzz."""
from datetime import datetime

from fpdf import FPDF
from PIL import Image

from . import advice
from .config import ASSETS_DIR

TEAL, NAVY = (15, 124, 120), (31, 58, 95)
BAND = {"green": (46, 160, 67), "amber": (230, 150, 20), "red": (200, 40, 40)}
URDU_FONT = ASSETS_DIR / "fonts" / "NotoNaskhArabic-Regular.ttf"


def build_pdf(result, colour: str, heatmap_img, patient: dict, is_demo: bool = False) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(False)
    pdf.add_page()
    logo = ASSETS_DIR / "logo.png"
    if logo.exists():
        pdf.image(str(logo), x=15, y=12, h=16)
    pdf.set_xy(35, 13); pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, "DiaSight Screening Report")
    pdf.set_xy(35, 21); pdf.set_font("Helvetica", "", 10); pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, f"Generated {datetime.now():%d %b %Y, %H:%M}")
    pdf.set_draw_color(*TEAL); pdf.line(15, 32, 195, 32)

    if is_demo:
        pdf.set_xy(15, 34); pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(200, 40, 40)
        pdf.cell(0, 6, "DEMO MODE: untrained model. This result is not meaningful.")

    # patient fields
    pdf.set_xy(15, 42); pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 11)
    fields = [("Patient", patient.get("name") or "-"), ("Age", patient.get("age") or "-"),
              ("Years with diabetes", patient.get("years") or "-")]
    pdf.cell(0, 6, "     ".join(f"{k}: {v}" for k, v in fields))

    # result band
    pdf.set_fill_color(*BAND[colour]); pdf.rect(15, 52, 180, 22, "F")
    pdf.set_xy(20, 55); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, f"Grade {result.grade}: {result.grade_name}")
    pdf.set_xy(20, 63); pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, "REFER to an eye doctor" if result.refer else "No referral needed now")

    # images
    pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 9)
    pdf.image(Image.fromarray(result.display_img), x=15, y=80, w=85)
    pdf.image(Image.fromarray(heatmap_img), x=110, y=80, w=85)
    pdf.set_xy(15, 166); pdf.cell(85, 5, "Retina image", align="C")
    pdf.set_xy(110, 166); pdf.cell(85, 5, "Where the model looked (Grad-CAM)", align="C")

    # advice
    en, ur = advice.get(result.grade)
    pdf.set_xy(15, 178); pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 7, "Advice")
    pdf.set_xy(15, 186); pdf.set_font("Helvetica", "", 12); pdf.multi_cell(180, 7, en)
    if URDU_FONT.exists():
        try:
            pdf.add_font("NotoNaskh", "", str(URDU_FONT))
            pdf.set_font("NotoNaskh", "", 14)
            pdf.set_text_shaping(use_shaping_engine=True, direction="rtl", script="arab", language="urd")
            pdf.set_xy(15, pdf.get_y() + 3); pdf.multi_cell(180, 9, ur, align="R")
            pdf.set_text_shaping(False)
        except Exception:  # uharfbuzz missing: keep the report English-only
            pass

    # footer
    pdf.set_xy(15, 270); pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(180, 5, advice.DISCLAIMER + " Prototype for an AI/ML course project; "
                   "not for clinical use.")
    return bytes(pdf.output())
