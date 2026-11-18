"""One-page PDF screening report (fpdf2). Urdu needs the bundled Noto Naskh font + uharfbuzz.

Any free text (patient name, etc.) goes through `_put_text`, which picks a font that can draw it:
Helvetica for Latin text, Noto Naskh (shaped, right-to-left) for Urdu/Arabic text, and a safe
fallback that never raises, so an unusual character can never break the download."""
import hashlib
import unicodedata
from datetime import datetime

from fpdf import FPDF
from PIL import Image

from . import advice
from .config import ASSETS_DIR, GRADE_NAMES

TEAL, NAVY, MUTED, LINE = (15, 124, 120), (31, 58, 95), (90, 107, 125), (214, 225, 224)
BAND = {"green": (36, 128, 79), "amber": (176, 122, 12), "red": (179, 55, 47)}
SCALE = [(36, 128, 79), (110, 160, 70), (176, 122, 12), (200, 90, 40), (179, 55, 47)]
URDU_FONT = ASSETS_DIR / "fonts" / "NotoNaskhArabic-Regular.ttf"
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "…": "...", " ": " "})


def _is_arabic_script(ch: str) -> bool:
    return "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ" or "ﭐ" <= ch <= "﻿"


class _Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(False)
        self.urdu_ok = False
        if URDU_FONT.exists():
            try:
                import uharfbuzz  # noqa: F401  (text shaping engine)
                self.add_font("NotoNaskh", "", str(URDU_FONT))
                self.urdu_ok = True
            except Exception:
                pass

    def put_text(self, x, y, w, h, text, size=11, style="", align="L", color=NAVY):
        """Write one line of free text in whichever font can draw it. Never raises on odd characters."""
        text = unicodedata.normalize("NFC", str(text)).translate(_PUNCT).strip() or "-"
        self.set_text_color(*color)
        self.set_xy(x, y)
        try:
            text.encode("latin-1")
            self.set_font("Helvetica", style, size)
            self.cell(w, h, text, align=align)
            return
        except UnicodeEncodeError:
            pass
        arabic = sum(_is_arabic_script(c) for c in text)
        if self.urdu_ok and arabic >= len(text.replace(" ", "")) / 2:
            # Noto Naskh has no Latin letters, so keep only what it can draw.
            keep = "".join(c for c in text if _is_arabic_script(c) or c.isspace() or c.isdigit() or c in ".-/()")
            try:
                self.set_font("NotoNaskh", "", size + 2)
                self.set_text_shaping(use_shaping_engine=True, direction="rtl", script="arab", language="urd")
                self.cell(w, h, keep, align=align)
                return
            except Exception:
                pass
            finally:
                self.set_text_shaping(False)
        safe = text.encode("latin-1", "replace").decode("latin-1")
        self.set_font("Helvetica", style, size)
        self.set_xy(x, y)
        self.cell(w, h, safe, align=align)


def report_id(result) -> str:
    """Short, stable ID derived from the image, so the same photo always gets the same report number."""
    return hashlib.sha1(result.display_img.tobytes()).hexdigest()[:8].upper()


def build_pdf(result, colour: str, heatmap_img, patient: dict, is_demo: bool = False,
              quality: dict | None = None, model_info: str = "") -> bytes:
    pdf = _Report()
    pdf.add_page()

    # ---- header
    pdf.set_fill_color(*TEAL); pdf.rect(0, 0, 210, 4, "F")
    logo = ASSETS_DIR / "logo.png"
    if logo.exists():
        pdf.image(str(logo), x=15, y=11, h=15)
    pdf.put_text(33, 11, 120, 8, "DiaSight Screening Report", 19, "B")
    pdf.put_text(33, 19, 120, 6, "Diabetic retinopathy screening from a retina photo", 9.5, color=MUTED)
    pdf.put_text(135, 11, 60, 6, f"Report  {report_id(result)}", 9.5, "B", align="R")
    pdf.put_text(135, 17, 60, 6, f"{datetime.now():%d %b %Y, %H:%M}", 9.5, align="R", color=MUTED)
    pdf.set_draw_color(*LINE); pdf.set_line_width(0.3); pdf.line(15, 30, 195, 30)

    y = 33
    if is_demo:
        pdf.set_fill_color(255, 246, 224); pdf.rect(15, y, 180, 8, "F")
        pdf.put_text(18, y + 1, 175, 6, "DEMO MODE: untrained model. This result is not meaningful.", 10, "B",
                     color=(179, 55, 47))
        y += 10

    # ---- patient
    cols = [("PATIENT", patient.get("name")), ("AGE", patient.get("age")),
            ("YEARS WITH DIABETES", patient.get("years"))]
    for i, (k, v) in enumerate(cols):
        x = 15 + i * 60
        pdf.put_text(x, y + 1, 58, 5, k, 7.5, "B", color=MUTED)
        pdf.put_text(x, y + 6, 58, 7, v or "-", 12)
    y += 17

    # ---- result band
    c = BAND[colour]
    pdf.set_fill_color(*c); pdf.rect(15, y, 180, 26, "F")
    decision = "REFER to an eye doctor" if result.refer else "No referral needed now"
    pdf.put_text(21, y + 4, 120, 9, decision, 17, "B", color=(255, 255, 255))
    pdf.put_text(21, y + 14, 120, 7, f"Grade {result.grade} of 4: {result.grade_name} diabetic retinopathy",
                 11.5, color=(255, 255, 255))
    pdf.put_text(140, y + 4, 50, 6, "REFER SCORE", 7.5, "B", align="R", color=(255, 255, 255))
    pdf.put_text(140, y + 10, 50, 10, f"{result.refer_score:.2f}", 20, "B", align="R", color=(255, 255, 255))
    y += 30

    # ---- severity scale
    seg_w = 180 / 5
    for g in range(5):
        x = 15 + g * seg_w
        pdf.set_fill_color(*SCALE[g]) if g == result.grade else pdf.set_fill_color(236, 241, 241)
        pdf.rect(x + 0.6, y, seg_w - 1.2, 7, "F")
        col = (255, 255, 255) if g == result.grade else MUTED
        pdf.put_text(x, y + 0.7, seg_w, 6, f"{g}  {GRADE_NAMES[g]}", 8.5, "B" if g == result.grade else "",
                     align="C", color=col)
    y += 12

    # ---- images
    pdf.image(Image.fromarray(result.display_img), x=15, y=y, w=85)
    pdf.image(Image.fromarray(heatmap_img), x=110, y=y, w=85)
    pdf.put_text(15, y + 86, 85, 5, "Retina photo", 9, align="C", color=MUTED)
    pdf.put_text(110, y + 86, 85, 5, "Where the model looked (Grad-CAM)", 9, align="C", color=MUTED)
    y += 95

    # ---- advice
    pdf.set_draw_color(*LINE); pdf.line(15, y, 195, y)
    en, ur = advice.get(result.grade)
    pdf.put_text(15, y + 3, 100, 7, "Advice", 12, "B")
    pdf.set_xy(15, y + 11); pdf.set_font("Helvetica", "", 12); pdf.set_text_color(*NAVY)
    pdf.multi_cell(180, 7, en)
    if pdf.urdu_ok:
        try:
            pdf.set_font("NotoNaskh", "", 14)
            pdf.set_text_shaping(use_shaping_engine=True, direction="rtl", script="arab", language="urd")
            pdf.set_xy(15, pdf.get_y() + 2); pdf.multi_cell(180, 9, ur, align="R")
        except Exception:
            pass
        finally:
            pdf.set_text_shaping(False)

    # ---- technical line
    tech = []
    if quality:
        tech.append(f"Image quality: sharpness {quality['sharpness']:.0f}, brightness {quality['brightness']:.0f} (passed)")
    if model_info:
        tech.append(f"Model: {model_info}")
    if tech:
        pdf.put_text(15, 258, 180, 5, "   |   ".join(tech), 8, color=MUTED)

    # ---- footer
    pdf.set_draw_color(*LINE); pdf.line(15, 265, 195, 265)
    pdf.set_xy(15, 268); pdf.set_font("Helvetica", "I", 8.5); pdf.set_text_color(*MUTED)
    pdf.multi_cell(180, 4.5, advice.DISCLAIMER + " Prototype for an AI/ML course project; not for clinical use.")
    return bytes(pdf.output())
