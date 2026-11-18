"""Quick check that DiaSight loads and can grade an image. Exits 0 on success."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diasight import model as model_lib, predict, quality
from diasight.config import SAMPLES_DIR
from diasight.preprocess import load_rgb

model, cfg, is_demo = model_lib.load()
print("Model loaded" + (" (demo mode)" if is_demo else ""))

samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
if samples:
    img = load_rgb(samples[0])
    ok, msg, _ = quality.check(img, cfg["quality"])
    if ok:
        res = predict.predict(model, cfg, img)
        print(f"{samples[0].name}: grade {res.grade}, refer={res.refer}")
    else:
        print(f"{samples[0].name}: failed quality check ({msg})")
print("All good")