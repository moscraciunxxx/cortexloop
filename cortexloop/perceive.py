from __future__ import annotations

from . import CLASS_COLORS, CLASS_NAMES
from .native import perceive as native_perceive


def color_prior(rgb: bytes, width: int, height: int) -> dict:
    x0, x1 = width * 2 // 5, width * 3 // 5
    y0, y1 = height * 2 // 5, height * 3 // 5
    sr = sg = sb = n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = (y * width + x) * 3
            sr += rgb[i]
            sg += rgb[i + 1]
            sb += rgb[i + 2]
            n += 1
    if n == 0:
        return {"class_id": 0, "class_name": "empty", "confidence": 0.0}
    r, g, b = sr / n, sg / n, sb / n
    best, best_d = 0, 1e9
    for cid, (cr, cg, cb) in CLASS_COLORS.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best, best_d = cid, d
    # floor/ceiling average is dark gray → empty
    if r + g + b < 90:
        best = 0
    conf = max(0.15, min(0.98, 1.0 - best_d / 80000.0))
    return {"class_id": best, "class_name": CLASS_NAMES[best], "confidence": conf}


def perceive_fused(rgb: bytes, width: int, height: int) -> dict:
    prior = color_prior(rgb, width, height)
    try:
        cnn = native_perceive(rgb, width, height)
    except Exception as exc:
        prior["source"] = "color_prior"
        prior["error"] = str(exc)
        prior["preprocess_us"] = 0
        prior["infer_us"] = 0
        prior["total_us"] = 0
        return prior
    use_cnn = cnn["confidence"] >= 0.42
    out = dict(cnn if use_cnn else prior)
    out["cnn"] = cnn
    out["prior"] = prior
    out["source"] = "cnn" if use_cnn else "color_prior"
    out["preprocess_us"] = cnn["preprocess_us"]
    out["infer_us"] = cnn["infer_us"]
    out["total_us"] = cnn["total_us"]
    return out
