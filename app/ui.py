"""Look and feel shared by every DiaSight page: a dark "medical AI console" theme, HTML components, image helpers.

Streamlit renders st.markdown HTML through a Markdown parser, so every component is emitted as a
single line (see `html`) to stop indentation being read as a code block."""
import base64
import io
from functools import lru_cache
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from diasight.config import GRADE_NAMES

# Status colours are reserved for the decision. Tuned for the dark surface.
STATUS = {"green": ("#34D399", "rgba(52,211,153,.10)", "rgba(52,211,153,.45)"),
          "amber": ("#FBBF24", "rgba(251,191,36,.10)", "rgba(251,191,36,.45)"),
          "red": ("#F87171", "rgba(248,113,113,.10)", "rgba(248,113,113,.45)")}
GRADE_SCALE = ["#34D399", "#A3E635", "#FBBF24", "#FB923C", "#F87171"]
SERIES = {"val": "#0E9FB8", "ext": "#8B5CF6"}  # validated categorical pair on the dark surface

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
:root{--bg:#060A13;--panel:rgba(14,21,36,.78);--panel2:#0E1524;--line:#1C2740;--line2:#2A3A5A;--ink:#E2E8F0;--muted:#8A9BB4;
--dim:#5B6B84;--cyan:#22D3EE;--teal:#2DD4BF;--violet:#8B5CF6;
--mono:'JetBrains Mono',ui-monospace,Consolas,monospace;--disp:'Space Grotesk','Inter',system-ui,sans-serif;}
.stApp{background:
  radial-gradient(900px 500px at 12% -5%,rgba(34,211,238,.10),transparent 60%),
  radial-gradient(800px 500px at 95% 10%,rgba(139,92,246,.10),transparent 60%),
  linear-gradient(rgba(148,163,184,.035) 1px,transparent 1px) 0 0/44px 44px,
  linear-gradient(90deg,rgba(148,163,184,.035) 1px,transparent 1px) 0 0/44px 44px,var(--bg) !important;}
.stApp,.stMarkdown,p,li,label,input,textarea,button,[data-testid="stWidgetLabel"] p{font-family:'Inter','Segoe UI',system-ui,sans-serif;}
h1,h2,h3{font-family:var(--disp) !important;letter-spacing:-.02em;}
header[data-testid="stHeader"],[data-testid="stToolbar"],footer,[data-testid="stDecoration"]{display:none !important;}
[data-testid="stMainBlockContainer"]{max-width:1240px;padding-top:1rem;padding-bottom:4rem;}
[data-testid="stSidebarCollapsedControl"]{display:none;}
@media (min-width:820px){[data-testid="stColumn"]:has(.st-key-photo_panel){position:sticky;top:1rem;align-self:flex-start;}}

/* ---------- top bar ---------- */
.st-key-topbar{background:rgba(10,16,30,.78);backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:14px;
  padding:.6rem .7rem .6rem .8rem;margin-bottom:1.2rem;min-height:64px;box-shadow:0 10px 30px -15px rgba(0,0,0,.8);}
.st-key-topbar [data-testid="stElementContainer"],.st-key-topbar [data-testid="stMarkdownContainer"],.st-key-topbar .stMarkdown{margin:0 !important;}
.st-key-topbar [data-testid="stMarkdownContainer"] > div,.st-key-topbar p{margin:0 !important;}
.ds-brand{display:flex;align-items:center;gap:13px;padding-right:.5rem;}
.ds-logo{flex:0 0 auto;padding:2px;border-radius:12px;background:linear-gradient(135deg,#22D3EE,#8B5CF6);
  box-shadow:0 0 22px -6px rgba(34,211,238,.8),0 0 22px -8px rgba(139,92,246,.8);}
.ds-logo img{display:block;height:36px;width:36px;border-radius:10px;}
.ds-brand b{font-family:var(--disp);font-size:1.22rem;color:#fff;display:block;line-height:1.15;letter-spacing:-.01em;white-space:nowrap;}
.ds-brand b i{font-style:normal;background:linear-gradient(90deg,#22D3EE,#A78BFA);-webkit-background-clip:text;background-clip:text;color:transparent;}
.ds-brand span{display:block;font-family:var(--mono);font-size:.6rem;color:var(--muted);letter-spacing:.16em;line-height:1.5;margin-top:2px;white-space:nowrap;}
.st-key-navgroup{background:rgba(6,10,19,.7);border:1px solid var(--line);border-radius:11px;padding:3px;gap:2px !important;}
.st-key-navgroup [data-testid="stPageLink"] a{display:flex;align-items:center;border-radius:8px;padding:.38rem .95rem;border:1px solid transparent;white-space:nowrap;min-height:0;}
.st-key-navgroup [data-testid="stPageLink"] a:hover{background:rgba(34,211,238,.06);}
.st-key-navgroup .ds-active a{background:linear-gradient(180deg,color-mix(in srgb,var(--ic) 20%,transparent),color-mix(in srgb,var(--ic) 8%,transparent));
  border-color:color-mix(in srgb,var(--ic) 45%,transparent);box-shadow:0 0 16px -6px var(--ic);}
.st-key-navgroup [data-testid="stPageLink"] p{color:var(--muted);font-weight:500;font-size:.88rem;white-space:nowrap;}
.st-key-navgroup [data-testid="stPageLink"] a:hover p{color:var(--ink);}
.st-key-navgroup .ds-active a p{color:#fff;}
.st-key-navgroup .ds-active a [data-testid="stIconMaterial"]{color:var(--cyan);}
.ds-status{display:flex;width:max-content;align-items:center;gap:8px;font-family:var(--mono);font-size:.7rem;
  color:var(--muted);border:1px solid var(--line);background:rgba(6,10,19,.6);border-radius:8px;padding:.32rem .65rem;white-space:nowrap;}
.ds-status b{color:#34D399;font-weight:600;letter-spacing:.08em;}
.ds-status b.demo{color:#FBBF24;}
.ds-pulse{width:7px;height:7px;border-radius:50%;background:#34D399;animation:ds-pulse 2s infinite;}
.ds-pulse.demo{background:#FBBF24;}
@keyframes ds-pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.55);}70%{box-shadow:0 0 0 8px rgba(52,211,153,0);}100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}}

/* ---------- hero ---------- */
.ds-hero{position:relative;overflow:hidden;border-radius:20px;padding:2.4rem 2.5rem;margin-bottom:1rem;
  background:linear-gradient(135deg,rgba(14,21,36,.92),rgba(10,16,30,.85));border:1px solid var(--line);
  display:flex;gap:2rem;align-items:center;}
.ds-hero:before{content:"";position:absolute;inset:0;border-radius:20px;padding:1px;pointer-events:none;
  background:linear-gradient(120deg,rgba(34,211,238,.55),transparent 35%,transparent 65%,rgba(139,92,246,.5));
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;}
.ds-hero-text{flex:1;position:relative;z-index:1;}
.ds-kicker{font-family:var(--mono);font-size:.74rem;color:var(--cyan);letter-spacing:.14em;}
.ds-kicker:before{content:"";display:inline-block;width:22px;height:1px;background:var(--cyan);vertical-align:middle;margin-right:.6rem;}
.ds-hero h1{color:#fff;font-size:2.7rem;line-height:1.06;margin:.85rem 0 .8rem;padding:0;font-weight:700;}
.ds-hero h1 em{font-style:normal;background:linear-gradient(90deg,#22D3EE,#8B5CF6);-webkit-background-clip:text;background-clip:text;color:transparent;}
.ds-hero p{color:var(--muted);font-size:1.02rem;max-width:560px;margin:0;line-height:1.6;}
.ds-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;margin-top:1.6rem;max-width:640px;}
.ds-stat{border:1px solid var(--line);background:rgba(6,10,19,.55);border-radius:12px;padding:.7rem .8rem;}
.ds-stat b{display:block;font-family:var(--mono);font-size:1.35rem;color:#fff;line-height:1.1;font-weight:600;}
.ds-stat span{font-size:.7rem;color:var(--muted);letter-spacing:.02em;}

/* HUD scanner */
.ds-scan{position:relative;z-index:1;width:300px;height:300px;flex:0 0 300px;}
.ds-scan img{position:absolute;inset:45px;width:210px;height:210px;border-radius:50%;object-fit:cover;
  box-shadow:0 0 0 1px rgba(34,211,238,.4),0 0 60px -6px rgba(34,211,238,.45);}
.ds-scan .r1,.ds-scan .r2,.ds-scan .r3{position:absolute;border-radius:50%;}
.ds-scan .r1{inset:28px;border:1px dashed rgba(34,211,238,.45);animation:ds-spin 24s linear infinite;}
.ds-scan .r2{inset:14px;border:2px solid transparent;border-top-color:var(--cyan);border-right-color:rgba(34,211,238,.25);animation:ds-spin 3.2s linear infinite;}
.ds-scan .r3{inset:2px;border:1px solid rgba(139,92,246,.35);border-bottom-color:transparent;animation:ds-spin 9s linear infinite reverse;}
.ds-scan .sweep{position:absolute;inset:45px;border-radius:50%;overflow:hidden;}
.ds-scan .sweep:after{content:"";position:absolute;left:0;right:0;height:40%;top:-40%;
  background:linear-gradient(180deg,transparent,rgba(34,211,238,.28) 85%,rgba(34,211,238,.9) 100%);animation:ds-sweep 3s ease-in-out infinite;}
.ds-scan .cross{position:absolute;left:50%;top:50%;width:18px;height:18px;transform:translate(-50%,-50%);}
.ds-scan .cross:before,.ds-scan .cross:after{content:"";position:absolute;background:rgba(255,255,255,.7);}
.ds-scan .cross:before{left:8px;top:0;width:2px;height:18px;}.ds-scan .cross:after{top:8px;left:0;height:2px;width:18px;}
.ds-scan .tag{position:absolute;font-family:var(--mono);font-size:.62rem;color:var(--muted);letter-spacing:.08em;white-space:nowrap;}
.ds-scan .tag b{color:var(--cyan);font-weight:500;}
@keyframes ds-spin{to{transform:rotate(360deg);}}
@keyframes ds-sweep{0%{top:-40%;}60%{top:100%;}100%{top:100%;}}
@media (prefers-reduced-motion:reduce){.ds-scan *,.ds-scan .sweep:after,.ds-pulse,.ds-hud .scanline,.ds-empty .blink:after{animation:none !important;}}
@media (max-width:900px){.ds-hero{flex-direction:column-reverse;padding:1.6rem;}.ds-hero h1{font-size:2rem;}
  .ds-stats{grid-template-columns:repeat(2,minmax(0,1fr));}}

/* pipeline strip */
.ds-pipeline{display:grid;grid-template-columns:repeat(5,1fr);margin-bottom:1.2rem;border:1px solid var(--line);
  border-radius:14px;overflow:hidden;background:rgba(10,16,30,.6);}
.ds-pipeline div{padding:.8rem 1rem;border-right:1px solid var(--line);}
.ds-pipeline div:last-child{border-right:none;}
.ds-pipeline small{font-family:var(--mono);color:var(--cyan);font-size:.66rem;letter-spacing:.1em;}
.ds-pipeline b{display:block;color:var(--ink);font-size:.88rem;font-weight:600;margin:.15rem 0 .1rem;}
.ds-pipeline span{font-family:var(--mono);font-size:.68rem;color:var(--dim);}
@media (max-width:820px){.ds-pipeline{grid-template-columns:1fr 1fr;}.ds-pipeline div{border-bottom:1px solid var(--line);}}

/* ---------- panels ---------- */
.st-key-photo_panel,.st-key-result_panel,[class*="st-key-card_"]{position:relative;background:var(--panel);backdrop-filter:blur(6px);
  border:1px solid var(--line);border-radius:16px;padding:1.25rem 1.35rem 1.35rem;box-shadow:0 20px 40px -25px rgba(0,0,0,.9);}
.st-key-photo_panel:before,.st-key-result_panel:before,[class*="st-key-card_"]:before{content:"";position:absolute;left:18px;right:18px;top:-1px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.6),transparent);}
[class*="st-key-card_"]{margin-bottom:.4rem;}
.ds-h{display:flex;align-items:baseline;gap:.7rem;font-family:var(--disp);font-size:1.08rem;font-weight:600;color:#fff;margin:0 0 .7rem;}
.ds-h .ds-num{font-family:var(--mono);font-size:.68rem;color:var(--cyan);letter-spacing:.12em;font-weight:500;}
.ds-sub{color:var(--muted);font-size:.86rem;margin:-.35rem 0 .85rem;}
.ds-label{font-family:var(--mono);font-size:.66rem;font-weight:500;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin:1rem 0 .5rem;}
.ds-label:before{content:"// ";color:var(--cyan);}

[data-testid="stFileUploaderDropzone"]{border:1px dashed rgba(34,211,238,.35);background:rgba(34,211,238,.03);border-radius:12px;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cyan);background:rgba(34,211,238,.07);}

.st-key-gallery [data-testid="stHorizontalBlock"]{gap:.5rem;}
.st-key-gallery [data-testid="stImage"] img{border-radius:50%;aspect-ratio:1;object-fit:cover;box-shadow:0 0 0 1px var(--line2);transition:box-shadow .2s;}
.st-key-gallery [data-testid="stColumn"]:hover [data-testid="stImage"] img{box-shadow:0 0 0 2px var(--cyan),0 0 18px -2px rgba(34,211,238,.6);}
.st-key-gallery .stButton button{width:100%;padding:.12rem .2rem;min-height:0;font-family:var(--mono);font-size:.7rem;border-radius:6px;
  border:1px solid var(--line2);color:var(--muted);background:transparent;}
.st-key-gallery .stButton{display:flex;justify-content:center;}
.st-key-gallery .stButton button:hover{border-color:var(--cyan);color:var(--cyan);}
.st-key-gallery .stButton button[kind="primary"]{background:rgba(34,211,238,.14);border-color:var(--cyan);color:var(--cyan);}

.stDownloadButton button{width:100%;border-radius:10px;padding:.7rem 1rem;font-weight:600;}
.stDownloadButton button[kind="primary"]{background:linear-gradient(90deg,#0EA5B7,#7C5CF0);border:none;color:#fff;box-shadow:0 10px 30px -10px rgba(34,211,238,.6);}
.stDownloadButton button[kind="primary"]:hover{filter:brightness(1.1);color:#fff;}
[data-testid="stExpander"] details{border-radius:12px;border-color:var(--line);background:rgba(6,10,19,.4);}
[data-testid="stExpander"] summary p{font-weight:600;}

/* ---------- empty state ---------- */
.ds-steps{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;}
.ds-step{border:1px solid var(--line);border-radius:12px;padding:1rem;background:rgba(6,10,19,.45);}
.ds-step small{font-family:var(--mono);color:var(--cyan);font-size:.66rem;letter-spacing:.1em;}
.ds-step b{display:block;color:#fff;font-size:.95rem;margin:.3rem 0 .15rem;}
.ds-step span{color:var(--muted);font-size:.84rem;}
.ds-empty{border:1px dashed var(--line2);border-radius:12px;padding:1rem 1.1rem;margin-bottom:.8rem;}
.ds-empty .blink{font-family:var(--mono);color:var(--cyan);font-size:.8rem;}
.ds-empty .blink:after{content:"_";animation:ds-blink 1s steps(1) infinite;}
@keyframes ds-blink{50%{opacity:0;}}
.ds-empty b{display:block;color:#fff;font-family:var(--disp);font-size:1.15rem;margin-top:.3rem;}
.ds-empty span{color:var(--muted);font-size:.88rem;}
@media (max-width:640px){.ds-steps{grid-template-columns:1fr;}}

/* ---------- result ---------- */
.ds-log{font-family:var(--mono);font-size:.72rem;line-height:1.8;background:#050810;border:1px solid var(--line);border-radius:12px;
  padding:.7rem .9rem;margin-bottom:1rem;color:var(--muted);overflow-x:auto;}
.ds-log div{white-space:nowrap;}
.ds-log .t{color:var(--dim);}
.ds-log .ok{color:#34D399;}
.ds-log .bad{color:#F87171;}
.ds-log .k{color:var(--cyan);}
.ds-verdict{position:relative;border:1px solid var(--bd);background:linear-gradient(120deg,var(--bg2),rgba(6,10,19,.4) 80%);
  border-radius:16px;padding:1.2rem 1.3rem;display:flex;gap:1.3rem;align-items:center;box-shadow:0 0 40px -18px var(--c);}
.ds-ring{flex:0 0 96px;width:96px;height:96px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:conic-gradient(var(--c) calc(var(--p)*1%),#1A2338 0);position:relative;filter:drop-shadow(0 0 10px var(--bd));}
.ds-ring:before{content:"";position:absolute;inset:7px;border-radius:50%;background:#0A0F1C;}
.ds-ring b,.ds-ring span{position:relative;}
.ds-ring b{font-family:var(--mono);font-size:2.1rem;line-height:1;color:#fff;font-weight:600;}
.ds-ring span{font-family:var(--mono);font-size:.58rem;color:var(--muted);letter-spacing:.12em;margin-top:.2rem;}
.ds-vtext{flex:1;min-width:0;}
.ds-tag{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:.66rem;font-weight:600;letter-spacing:.12em;
  text-transform:uppercase;color:var(--c);border:1px solid var(--bd);border-radius:6px;padding:.16rem .5rem;background:rgba(6,10,19,.5);}
.ds-decision{font-family:var(--disp);font-size:1.75rem;font-weight:700;color:#fff;line-height:1.12;margin:.45rem 0 .25rem;}
.ds-follow{color:var(--muted);font-size:.93rem;}
.ds-follow b{color:var(--ink);font-weight:600;}

.ds-scale{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin:1.1rem 0 .35rem;}
.ds-seg{height:8px;border-radius:2px;background:#1A2338;}
.ds-seg.on{box-shadow:0 0 14px 1px var(--sc);}
.ds-scalelab{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;font-family:var(--mono);font-size:.62rem;color:var(--dim);text-align:center;}
.ds-scalelab .on{color:#fff;}
.ds-scalenote{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.62rem;color:var(--dim);margin-top:.35rem;}

.ds-meter{margin-top:1rem;}
.ds-meter-top{display:flex;justify-content:space-between;align-items:baseline;font-size:.82rem;color:var(--muted);}
.ds-meter-top b{color:#fff;font-family:var(--mono);font-size:1.05rem;}
.ds-track{position:relative;height:6px;border-radius:3px;background:linear-gradient(90deg,rgba(52,211,153,.35),rgba(251,191,36,.35) 55%,rgba(248,113,113,.45));margin-top:.5rem;}
.ds-track i{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;background:var(--c);border:2px solid #0A0F1C;
  box-shadow:0 0 12px 2px var(--c);transform:translate(-50%,-50%);}
.ds-track em{position:absolute;top:-6px;bottom:-6px;width:1px;background:#fff;opacity:.7;}
.ds-track-lab{position:relative;height:1.1rem;font-family:var(--mono);font-size:.62rem;color:var(--dim);}
.ds-track-lab span{position:absolute;transform:translateX(-50%);white-space:nowrap;top:.3rem;}

/* HUD image frames */
.ds-hudrow{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.ds-hud{position:relative;background:#03060C;border:1px solid var(--line);border-radius:12px;padding:12px;overflow:hidden;}
.ds-hud img{display:block;width:100%;border-radius:6px;}
.ds-hud .c{position:absolute;width:14px;height:14px;border-color:var(--cyan);border-style:solid;}
.ds-hud .c1{left:6px;top:6px;border-width:2px 0 0 2px;}.ds-hud .c2{right:6px;top:6px;border-width:2px 2px 0 0;}
.ds-hud .c3{left:6px;bottom:6px;border-width:0 0 2px 2px;}.ds-hud .c4{right:6px;bottom:6px;border-width:0 2px 2px 0;}
.ds-hud .lab{position:absolute;left:18px;top:16px;font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;color:var(--cyan);
  background:rgba(3,6,12,.75);padding:.1rem .35rem;border-radius:3px;}
.ds-hud .lab2{left:auto;right:18px;top:auto;bottom:16px;color:var(--muted);}
.ds-hud .scanline{position:absolute;left:12px;right:12px;height:2px;top:12px;background:linear-gradient(90deg,transparent,rgba(34,211,238,.9),transparent);
  box-shadow:0 0 12px rgba(34,211,238,.8);animation:ds-line 4s ease-in-out infinite;}
@keyframes ds-line{0%{top:12px;opacity:0;}10%{opacity:1;}50%{top:calc(100% - 14px);opacity:1;}60%{opacity:0;}100%{opacity:0;}}
.ds-legend{display:flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.62rem;color:var(--dim);margin-top:.5rem;}
.ds-legend i{flex:1;max-width:160px;height:6px;border-radius:3px;background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000);}

.ds-advice{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin:1rem 0;}
.ds-advice>div{border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem;background:rgba(6,10,19,.45);}
.ds-lang{font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;color:var(--cyan);}
.ds-advice p{margin:.3rem 0 0;color:var(--ink);font-size:1rem;}
.ds-urdu{font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq',serif !important;direction:rtl;text-align:right;
  font-size:1.1rem !important;line-height:2.2;}
@media (max-width:640px){.ds-advice{grid-template-columns:1fr;}}

.ds-probs{display:flex;flex-direction:column;gap:.4rem;}
.ds-prow{display:grid;grid-template-columns:130px 1fr 52px;gap:.6rem;align-items:center;font-family:var(--mono);font-size:.72rem;color:var(--muted);}
.ds-prow .bar{height:8px;border-radius:0 3px 3px 0;background:#141C2E;position:relative;}
.ds-prow .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:0 3px 3px 0;background:#3B4B69;min-width:2px;}
.ds-prow.top{color:#fff;}
.ds-prow.top .bar i{background:linear-gradient(90deg,#0EA5B7,#22D3EE);box-shadow:0 0 12px rgba(34,211,238,.5);}
.ds-prow span:last-child{text-align:right;}
.ds-kv{display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;font-size:.84rem;color:var(--ink);}
.ds-kv span{color:var(--dim);font-family:var(--mono);font-size:.72rem;padding-top:.1rem;}
.ds-kv b{font-weight:500;}

.ds-retake{border:1px solid rgba(251,191,36,.45);background:linear-gradient(120deg,rgba(251,191,36,.08),rgba(6,10,19,.4) 80%);
  border-radius:16px;padding:1.2rem 1.3rem;box-shadow:0 0 40px -20px #FBBF24;--c:#FBBF24;--bd:rgba(251,191,36,.45);}
.ds-checks{display:flex;flex-direction:column;gap:.4rem;margin-top:.9rem;}
.ds-check{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.74rem;border:1px solid var(--line);
  background:rgba(6,10,19,.5);border-radius:8px;padding:.45rem .7rem;color:var(--muted);}
.ds-check span:first-child{color:#34D399;}
.ds-check.bad{border-color:rgba(248,113,113,.5);}
.ds-check.bad span:first-child{color:#F87171;}
.ds-check b{color:#fff;font-weight:500;}

.ds-demo{display:flex;gap:.6rem;align-items:center;background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.4);color:#FCD34D;
  border-radius:10px;padding:.6rem .9rem;font-size:.88rem;margin-bottom:1rem;}

/* ---------- performance / about ---------- */
.ds-pagehead h2{font-size:2.1rem;color:#fff;margin:.5rem 0 .3rem;padding:0;}
.ds-pagehead p{color:var(--muted);margin:0 0 1.3rem;max-width:780px;line-height:1.6;}
.ds-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:1rem;}
.ds-kpi{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem;overflow:hidden;}
.ds-kpi:after{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:linear-gradient(180deg,var(--cyan),var(--violet));}
.ds-kpi .k{font-family:var(--mono);font-size:.64rem;color:var(--dim);letter-spacing:.1em;text-transform:uppercase;}
.ds-kpi .v{font-family:var(--mono);font-size:2rem;font-weight:600;color:#fff;line-height:1.15;margin:.35rem 0 .25rem;}
.ds-kpi .ci{font-size:.74rem;color:var(--muted);}
.ds-kpi .ext{margin-top:.6rem;padding-top:.55rem;border-top:1px dashed var(--line2);font-size:.78rem;color:var(--ink);}
.ds-kpi .ext span{color:var(--muted);}
.ds-kpi .ext b{font-family:var(--mono);color:#C4B5FD;}
@media (max-width:820px){.ds-kpis{grid-template-columns:1fr 1fr;}}

.ds-legend2{display:flex;gap:1.2rem;flex-wrap:wrap;font-size:.8rem;color:var(--ink);margin:.1rem 0 .9rem;}
.ds-legend2 span{display:inline-flex;align-items:center;gap:6px;}
.ds-legend2 i{width:10px;height:10px;border-radius:2px;display:inline-block;}
.ds-cmp{display:flex;flex-direction:column;gap:.85rem;}
.ds-cmp-row{display:grid;grid-template-columns:180px 1fr;gap:.8rem;align-items:center;}
.ds-cmp-row>span{font-size:.84rem;color:var(--ink);}
.ds-cmp-bars{display:flex;flex-direction:column;gap:2px;}
.ds-cbar{display:flex;align-items:center;gap:.45rem;font-family:var(--mono);font-size:.7rem;color:var(--muted);}
.ds-cbar i{height:10px;border-radius:0 3px 3px 0;display:block;}
.ds-cbar:hover i{filter:brightness(1.25);}
.ds-axis{display:grid;grid-template-columns:180px 1fr;gap:.8rem;font-family:var(--mono);font-size:.64rem;color:var(--dim);}
.ds-axis div{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding-top:.25rem;}

table.ds-table{width:100%;border-collapse:collapse;font-size:.84rem;}
.ds-table th{text-align:left;font-family:var(--mono);font-weight:500;color:var(--dim);font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;
  border-bottom:1px solid var(--line2);padding:.5rem .5rem;}
.ds-table td{border-bottom:1px solid var(--line);padding:.55rem .5rem;color:var(--ink);font-variant-numeric:tabular-nums;}
.ds-table tr.active td{background:rgba(34,211,238,.07);color:#fff;}
.ds-table tr:hover td{background:rgba(148,163,184,.04);}
.ds-cellbar{display:flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.76rem;}
.ds-cellbar i{height:6px;border-radius:0 3px 3px 0;background:var(--cyan);display:block;}
.ds-pill{display:inline-block;font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;text-transform:uppercase;border-radius:4px;
  padding:.1rem .4rem;background:rgba(34,211,238,.12);color:var(--cyan);border:1px solid rgba(34,211,238,.35);}
.ds-pill.warn{background:rgba(251,191,36,.1);color:#FBBF24;border-color:rgba(251,191,36,.4);}

.ds-cm{display:grid;grid-template-columns:auto repeat(5,1fr);gap:3px;font-family:var(--mono);font-size:.76rem;}
.ds-cm div{display:flex;align-items:center;justify-content:center;border-radius:4px;min-height:38px;}
.ds-cm .h{color:var(--dim);font-size:.62rem;min-height:0;padding:.2rem .3rem;}
.ds-cmcap{font-size:.74rem;color:var(--dim);margin-top:.5rem;}

.ds-eyebrow{display:inline-block;font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;color:var(--cyan);
  border:1px solid rgba(34,211,238,.35);background:rgba(34,211,238,.08);border-radius:6px;padding:.2rem .55rem;}
.ds-note{border:1px solid rgba(251,191,36,.35);border-left:3px solid #FBBF24;background:rgba(251,191,36,.06);border-radius:10px;
  padding:.85rem 1rem;color:var(--ink);font-size:.9rem;line-height:1.6;}
.ds-note b{color:#FBBF24;}
.ds-flow{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;margin-bottom:1rem;}
.ds-flow div{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.95rem;}
.ds-flow b{display:block;font-size:.92rem;color:#fff;margin:.35rem 0 .2rem;}
.ds-flow span{font-size:.8rem;color:var(--muted);}
.ds-flow small{font-family:var(--mono);color:var(--cyan);font-size:.66rem;letter-spacing:.1em;}
@media (max-width:820px){.ds-flow{grid-template-columns:1fr 1fr;}}
.ds-foot{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--dim);font-family:var(--mono);font-size:.66rem;
  display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;letter-spacing:.04em;}

@media (max-width:1180px){.ds-status .det{display:none;}}
@media (max-width:640px){
.st-key-gallery [data-testid="stHorizontalBlock"]{flex-direction:row;flex-wrap:wrap;gap:.4rem;}
.st-key-gallery [data-testid="stColumn"]{flex:1 1 0 !important;min-width:0 !important;width:auto !important;}
.st-key-topbar{row-gap:.6rem !important;}
.st-key-topbar .ds-status{display:none;}
.st-key-navgroup{width:100% !important;justify-content:space-between;}
.st-key-navgroup [data-testid="stPageLink"] a{padding:.35rem .6rem;}.st-key-navgroup [data-testid="stPageLink"] p{font-size:.8rem;}
.ds-scan{transform:scale(.85);margin:-20px auto;}
.ds-verdict{flex-direction:column;align-items:flex-start;}
.ds-scalelab{font-size:.52rem;}
.ds-cmp-row,.ds-axis{grid-template-columns:110px 1fr;}
[data-testid="stMainBlockContainer"]{padding-left:1rem;padding-right:1rem;}
}
</style>
"""


# Nav icons are inline SVG masks, coloured per page. Streamlit's icon font is not used here: if it loads
# slowly the icon shows as its name ("visibility") and crowds the label.
_ICON_PATHS = {
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>'
            '<circle cx="12" cy="12" r="3.2"/><path d="M5.5 12s2.4-4.5 6.5-4.5 6.5 4.5 6.5 4.5-2.4 4.5-6.5 4.5S5.5 12 5.5 12z"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/><circle cx="19" cy="8" r="1.2"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.6" r=".9"/>',
}
NAV_ICONS = [("scan", "#22D3EE"), ("chart", "#A78BFA"), ("info", "#34D399")]


def _svg_url(name: str) -> str:
    from urllib.parse import quote
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')
    return f'url("data:image/svg+xml,{quote(svg)}")'


def _nav_css(active_nav: int) -> str:
    base = '.st-key-navgroup > [data-testid="stElementContainer"]'
    rules = [f'{base} [data-testid="stPageLink"] a::before{{content:"";flex:0 0 16px;width:16px;height:16px;margin-right:8px;'
             f'background:var(--ic);-webkit-mask:var(--svg) center/contain no-repeat;mask:var(--svg) center/contain no-repeat;'
             f'opacity:.75;transition:opacity .2s,filter .2s;}}',
             f'{base} [data-testid="stPageLink"] a:hover::before{{opacity:1;}}',
             f'{base}:nth-child({active_nav}) [data-testid="stPageLink"] a::before{{opacity:1;filter:drop-shadow(0 0 6px var(--ic));}}']
    for i, (name, colour) in enumerate(NAV_ICONS, 1):
        rules.append(f'{base}:nth-child({i}){{--ic:{colour};--svg:{_svg_url(name)};}}')
    return "<style>" + "".join(rules) + "</style>"


def inject_css(active_nav: int = 1):
    """active_nav: 1-based position of the current page in the top nav."""
    sel = f'.st-key-navgroup > [data-testid="stElementContainer"]:nth-child({active_nav})'
    active = CSS.replace(".st-key-navgroup .ds-active", sel).replace("</style>", _nav_css(active_nav)[7:])
    st.markdown(active, unsafe_allow_html=True)


def html(s: str):
    """Render an HTML snippet. Collapses lines so Markdown never treats indentation as code."""
    st.markdown(" ".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


def e(text) -> str:
    return escape(str(text))


# ---------- images ----------
def png_b64(img: np.ndarray | Image.Image, max_side: int | None = None, fmt: str = "PNG") -> str:
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    if max_side:
        im = im.copy(); im.thumbnail((max_side, max_side))
    if fmt == "JPEG":
        im = im.convert("RGB")
    buf = io.BytesIO(); im.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=32)
def thumb(path: str, side: int = 240) -> Image.Image:
    """Small square crop of a fundus photo (black border removed) for galleries and the hero."""
    from diasight.preprocess import crop_black_border
    im = Image.open(path).convert("RGB")
    im.thumbnail((900, 900))
    arr = crop_black_border(np.array(im))
    h, w = arr.shape[:2]; s = min(h, w)
    arr = arr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    out = Image.fromarray(arr); out.thumbnail((side, side))
    return out


# ---------- components ----------
def topbar(pages: dict, logo: Path, status: str, is_demo: bool):
    """Brand on the left, segmented page nav in the middle, live model status on the right."""
    logo_tag = f'<img src="data:image/png;base64,{png_b64(Image.open(logo), 96)}" alt="">' if logo.exists() else ""
    d = " demo" if is_demo else ""
    with st.container(key="topbar", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="medium"):
        st.markdown(f'<div class="ds-brand"><div class="ds-logo">{logo_tag}</div><div><b>Dia<i>Sight</i></b>'
                    f'<span>RETINAL&nbsp;AI&nbsp;·&nbsp;v0.1</span></div></div>',
                    unsafe_allow_html=True, width="content")
        with st.container(key="navgroup", horizontal=True, width="content", vertical_alignment="center", gap=None):
            for label, page in pages.items():
                st.page_link(page, label=label, width="content")
        st.markdown(f'<div class="ds-status"><span class="ds-pulse{d}"></span><b class="{d.strip()}">'
                    f'{"DEMO" if is_demo else "ONLINE"}</b><span class="det">{e(status)}</span></div>', unsafe_allow_html=True, width="content")


def hero(sample_path: Path | None, stats: list[tuple[str, str]], cfg: dict):
    scan = ""
    if sample_path is not None and sample_path.exists():
        img = png_b64(thumb(str(sample_path), 420), fmt="JPEG")
        size = cfg.get("img_size", 384)
        name = str(cfg.get("model_name", "")).replace("efficientnet_", "EffNet-").upper()
        scan = f"""
<div class="ds-scan"><div class="r3"></div><div class="r2"></div><div class="r1"></div>
<img src="data:image/jpeg;base64,{img}" alt="Retina photo"><div class="sweep"></div><div class="cross"></div>
<div class="tag" style="left:0;top:4px">INPUT <b>{size}×{size}</b></div>
<div class="tag" style="right:0;top:4px">MODEL <b>{e(name)}</b></div>
<div class="tag" style="left:0;bottom:4px">EXPLAIN <b>GRAD-CAM</b></div>
<div class="tag" style="right:0;bottom:4px">STATUS <b>READY</b></div></div>"""
    tiles = "".join(f'<div class="ds-stat"><b>{e(v)}</b><span>{e(k)}</span></div>' for k, v in stats)
    html(f"""
<div class="ds-hero"><div class="ds-hero-text">
<div class="ds-kicker">AI-POWERED RETINAL SCREENING</div>
<h1>Detect diabetic eye disease <em>from a single photo.</em></h1>
<p>A deep convolutional network grades diabetic retinopathy from 0 to 4, decides who needs an eye doctor,
and shows exactly where it looked. Quality-gated, explainable, bilingual.</p>
<div class="ds-stats">{tiles}</div></div>{scan}</div>""")


def pipeline_strip(cfg: dict):
    steps = [("01", "Quality gate", "blur · light · size"),
             ("02", "Preprocess", f"Ben Graham σ{cfg.get('ben_graham_sigma', 10)}"),
             ("03", "CNN inference", f"{cfg.get('model_name', '')} @{cfg.get('img_size', '')}"),
             ("04", "Explain", "Grad-CAM heatmap"),
             ("05", "Report", "EN + اردو · PDF")]
    html('<div class="ds-pipeline">' + "".join(
        f'<div><small>{n}</small><b>{t}</b><span>{e(s)}</span></div>' for n, t, s in steps) + '</div>')


def verdict(res, colour: str, follow: str):
    c, bg, bd = STATUS[colour]
    decision = "Refer to an eye doctor" if res.refer else "No referral needed"
    tag = ("&#9650; Referable" if res.grade >= 2 else "&#9650; Refer · borderline score") if res.refer else "&#9679; Not referable"
    p = (res.grade + 1) / 5 * 100
    html(f"""
<div class="ds-verdict" style="--c:{c};--bg2:{bg};--bd:{bd};--p:{p:.0f}">
<div class="ds-ring"><b>{res.grade}</b><span>GRADE</span></div>
<div class="ds-vtext"><span class="ds-tag">{tag}</span>
<div class="ds-decision">{decision}</div>
<div class="ds-follow"><b>{e(res.grade_name)}</b> diabetic retinopathy · {e(follow)}</div></div></div>""")


def severity_scale(grade: int):
    segs = "".join(
        f'<div class="ds-seg{" on" if g == grade else ""}" style="background:{GRADE_SCALE[g] if g <= grade else "#1A2338"};'
        f'--sc:{GRADE_SCALE[g]}" title="Grade {g}: {GRADE_NAMES[g]}"></div>' for g in range(5))
    labs = "".join(f'<span class="{"on" if g == grade else ""}">{g} {GRADE_NAMES[g].upper()}</span>' for g in range(5))
    html(f'<div class="ds-scale">{segs}</div><div class="ds-scalelab">{labs}</div>'
         f'<div class="ds-scalenote"><span>0–1 → RE-SCREEN</span><span>2–4 → REFER</span></div>')


def meter(score: float, cut: float, scale: float, label: str, colour: str):
    c = STATUS[colour][0]
    s = max(0.0, min(scale, score)); k = max(0.0, min(scale, cut))
    html(f"""
<div class="ds-meter" style="--c:{c}"><div class="ds-meter-top"><span>{e(label)}</span><b>{s:.3f}</b></div>
<div class="ds-track" title="{e(label)} {s:.3f}; refers at {k:.2f} or above"><em style="left:{k / scale * 100:.1f}%"></em>
<i style="left:{s / scale * 100:.1f}%"></i></div>
<div class="ds-track-lab"><span style="left:0;transform:none">0.00</span><span style="left:{k / scale * 100:.1f}%">THRESHOLD {k:.2f}</span>
<span style="right:0;left:auto;transform:none">{scale:.2f}</span></div></div>""")


def hud_pair(left: np.ndarray, right: np.ndarray, left_label: str, right_label: str, size_note: str):
    def frame(img, lab, scan):
        return (f'<div class="ds-hud"><img src="data:image/jpeg;base64,{png_b64(img, fmt="JPEG")}" alt="{e(lab)}">'
                f'<span class="c c1"></span><span class="c c2"></span><span class="c c3"></span><span class="c c4"></span>'
                f'<span class="lab">{e(lab)}</span><span class="lab lab2">{e(size_note)}</span>'
                f'{"<span class=scanline></span>" if scan else ""}</div>')
    html(f'<div class="ds-hudrow">{frame(left, left_label, False)}{frame(right, right_label, True)}</div>'
         '<div class="ds-legend">LOW ATTENTION<i></i>HIGH ATTENTION</div>')


def prob_bars(probs: list[float], grade: int):
    rows = "".join(
        f'<div class="ds-prow{" top" if g == grade else ""}" title="Grade {g} {GRADE_NAMES[g]}: {p * 100:.1f}%">'
        f'<span>{g} {GRADE_NAMES[g].upper()}</span><div class="bar"><i style="width:{p * 100:.1f}%"></i></div>'
        f'<span>{p * 100:.1f}%</span></div>' for g, p in enumerate(probs))
    html(f'<div class="ds-probs">{rows}</div>')


def footer(text: str):
    html(f'<div class="ds-foot"><span>{e(text)}</span><span>DIASIGHT · AI/ML CAPSTONE PROTOTYPE · NOT FOR CLINICAL USE</span></div>')
