from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .native import available_backends, backend_name, bench_json, cpu_features, set_backend
from .perceive import perceive_fused
from .planner import plan
from .render import render_camera
from .world import make_warehouse, remaining_targets, try_move

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

_lock = threading.Lock()
_world = make_warehouse()
_cam = b""
_perc = {}
_running = True
_hz = 0.0


def _loop():
    global _cam, _perc, _hz
    last = time.perf_counter()
    acc = 0
    tacc = 0.0
    while _running:
        t0 = time.perf_counter()
        with _lock:
            rgb = render_camera(_world, 96, 64)
            perc = perceive_fused(rgb, 96, 64)
            plan(_world, perc)
            try_move(_world, 0.05)
            _cam = rgb
            _perc = perc
        dt = time.perf_counter() - t0
        acc += 1
        tacc += time.perf_counter() - last
        last = time.perf_counter()
        if tacc >= 0.5:
            _hz = acc / tacc
            acc = 0
            tacc = 0.0
        time.sleep(max(0.0, 0.05 - dt))


def _state() -> dict:
    with _lock:
        w = _world
        r = w.robot
        return {
            "version": __version__,
            "cpu": cpu_features(),
            "backend": backend_name(),
            "backends": available_backends(),
            "hz": round(_hz, 2),
            "t": round(w.t, 2),
            "mission": w.mission,
            "status": w.status,
            "remaining": len(remaining_targets(w)),
            "robot": {
                "x": round(r.x, 3),
                "y": round(r.y, 3),
                "theta": round(r.theta, 3),
                "carrying": r.carrying,
                "delivered": r.delivered,
                "collisions": r.collisions,
                "energy_j": round(r.energy_j, 2),
                "speed": round(r.speed, 3),
            },
            "grid": {"w": w.w, "h": w.h, "tiles": w.tiles},
            "perception": {
                "class_name": _perc.get("class_name"),
                "class_id": _perc.get("class_id"),
                "confidence": _perc.get("confidence"),
                "source": _perc.get("source"),
                "preprocess_us": _perc.get("preprocess_us"),
                "infer_us": _perc.get("infer_us"),
                "total_us": _perc.get("total_us"),
                "cnn": _perc.get("cnn"),
                "prior": _perc.get("prior"),
            },
            "log": w.log[-12:],
            "camera": {"w": 96, "h": 64, "rgb": list(_cam) if _cam else []},
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send(200, json.dumps(_state()))
            return
        if path == "/api/bench":
            self._send(200, bench_json())
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            payload = {}
        global _world
        if path == "/api/backend":
            name = set_backend(payload.get("backend", "neon_i8mm"))
            self._send(200, json.dumps({"backend": name}))
            return
        if path == "/api/reset":
            with _lock:
                _world = make_warehouse(seed=int(payload.get("seed", 7)), mission=payload.get("mission", "collect_red"))
            self._send(200, json.dumps({"ok": True}))
            return
        if path == "/api/mission":
            with _lock:
                _world.mission = payload.get("mission", "collect_red")
                _world.note(f"mission set {_world.mission}")
            self._send(200, json.dumps({"mission": _world.mission}))
            return
        self._send(404, json.dumps({"error": "not found"}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    threading.Thread(target=_loop, daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CortexLoop dashboard  http://{args.host}:{args.port}")
    print(f"CPU {cpu_features()}")
    print(f"backend {backend_name()}  available={available_backends()}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
