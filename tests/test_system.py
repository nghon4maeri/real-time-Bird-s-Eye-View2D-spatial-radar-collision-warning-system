import json
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.config import load_config, save_config
from core.estimator import CameraHealthMonitor
import web as W


def test_config_save_and_load(tmp_path):
    cfg_path = str(tmp_path / "config.json")
    save_config({"pitch": 19.5, "height": 1.65, "hfov": 74.0, "muted": True, "radar_range": 8.0}, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded["pitch"] == 19.5
    assert loaded["height"] == 1.65
    assert loaded["hfov"] == 74.0
    assert loaded["muted"] is True
    assert loaded["radar_range"] == 8.0


def test_camera_health_monitor():
    mon = CameraHealthMonitor(duration_thresh=1.0)
    bright = np.random.default_rng(42).integers(60, 200, (120, 160, 3), dtype=np.uint8)
    h1 = mon.update(bright, now=10.0)
    assert h1["ok"] and not h1["alert"]

    dark = np.zeros((120, 160, 3), dtype=np.uint8)
    h2 = mon.update(dark, now=10.2)
    assert not h2["alert"]  # chua du 1.0s
    h3 = mon.update(dark, now=11.3)
    assert h3["alert"] and not h3["ok"] and h3["reason"] == "dark"

    h4 = mon.update(bright, now=11.5)
    assert h4["ok"] and not h4["alert"]


class _DummyApp:
    def __init__(self):
        self.state = {"t": time.time(), "level": 0, "sectors": [], "tracks": []}
        self.event_seq = 1
        self.event_cond = threading.Condition()
        self.stream_cond = threading.Condition()
        self._client_lock = threading.Lock()
        self.cam_clients = 0
        self.radar_clients = 0
        self.jpg = {}
        self.commands = []

    def command(self, c, q):
        self.commands.append((c, q))


def test_web_routes_and_stale_failsafe():
    app = _DummyApp()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), W.make_handler(app))
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            assert r.status == 200
            assert b"Webcam Radar" in r.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/static/radar.js", timeout=2) as r:
            assert r.status == 200
            assert b"RadarCanvas" in r.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2) as r:
            st = json.loads(r.read().decode())
            assert st["stale"] is False and st["level"] == 0

        # Ep stale > 1.5s -> fail-safe bao DO
        app.state["t"] = time.time() - 2.5
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2) as r:
            st_stale = json.loads(r.read().decode())
            assert st_stale["stale"] is True and st_stale["level"] == 2

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/cmd?c=mute", timeout=2) as r:
            assert r.status == 204
        assert app.commands[-1][0] == "mute"

        # SSE /events tra ve JSON hop le
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=2) as r:
            line = r.readline().decode().strip()
            assert line.startswith("data: ")
            ev = json.loads(line[len("data: "):])
            assert ev["stale"] is True and ev["level"] == 2
    finally:
        srv.shutdown()
        srv.server_close()
