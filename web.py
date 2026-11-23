#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Server web Radar 2D: Async Camera/Depth + SSE (/events) + Canvas HUD + Spatial Audio."""
import argparse
import json
import math
import mimetypes
import os
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from core import accel
from core.config import (
    DEFAULT_CONFIG_PATH,
    HFOV_DEG,
    CAM_HEIGHT,
    MODEL,
    PITCH_DEG,
    POSE_MODEL,
    load_config,
    save_config,
)
from core.pipeline import Beeper, FrameReader, RadarEngine, draw_camera, draw_radar

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
PAGE_PATH = os.path.join(ROOT_DIR, "web.html")


class App:
    def __init__(self, args):
        import torch

        torch.set_num_threads(args.threads)
        self.cfg_path = args.config
        cfg = load_config(self.cfg_path)

        hfov = args.hfov if args.hfov is not None else cfg["hfov"]
        height = args.height if args.height is not None else cfg["height"]
        pitch = args.pitch if args.pitch is not None else cfg["pitch"]

        self.reader = FrameReader(args.video if args.video else args.cam, realtime=True)
        self.reader.start()
        self.w, self.h = self.reader.w, self.reader.h

        self.engine = RadarEngine(
            self.w,
            self.h,
            hfov,
            height,
            pitch,
            model_path=args.model,
            pose_model_path=args.pose_model,
            imgsz=args.imgsz,
            det_every=args.det_every,
            use_pose=not args.no_pose,
            use_depth=not args.no_depth,
            use_kalman=True,
            use_sensor=not args.no_sensor,
            use_ego=not args.no_ego,
            backend=args.backend,
            depth_backend=args.depth_backend,
            radar_range=cfg["radar_range"],
            calib_dist=cfg["calib_dist"],
            offset=cfg["offset"],
        )

        self.beeper = Beeper()
        self.beeper.muted = bool(cfg["muted"])
        self.beeper.start()

        self.flip = bool(cfg["flip"])
        self.voice_enabled = bool(cfg["voice_enabled"])
        self.spatial_audio = bool(cfg["spatial_audio"])
        self.calib = False
        self.click = None
        self.calib_ack = 0.0

        self.jpg = {}
        self.state = {}
        self.event_seq = 0
        self.cam_clients = 0
        self.radar_clients = 0
        self._client_lock = threading.Lock()
        self.stream_cond = threading.Condition()
        self.event_cond = threading.Condition()

        print(
            f"[Startup] compute_backend={accel.get_backend()} | ai_backend={self.engine.ai_backend} | "
            f"sensor={self.engine.sensor.source} | depth={'ON' if self.engine.depth else 'OFF'}",
            flush=True,
        )

    def _persist(self):
        save_config(
            {
                "pitch": round(math.degrees(self.engine.cam.pitch), 2),
                "offset": round(self.engine.offset, 2),
                "height": round(self.engine.cam.height, 3),
                "hfov": round(self.engine.cam.hfov_deg, 1),
                "calib_dist": round(self.engine.calib_dist, 2),
                "flip": self.flip,
                "muted": self.beeper.muted,
                "voice_enabled": self.voice_enabled,
                "spatial_audio": self.spatial_audio,
                "radar_range": self.engine.radar_range,
            },
            self.cfg_path,
        )

    def loop(self):
        while True:
            try:
                self._loop()
            except Exception:
                traceback.print_exc()
                self.beeper.level = 2
                time.sleep(0.2)

    def _loop(self):
        last_seq = 0
        while True:
            ok, frame, _, seq = self.reader.read_latest(last_seq)
            if not ok or frame is None:
                self.beeper.level = 2
                time.sleep(0.15)
                continue
            if seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            if self.flip:
                frame = cv2.flip(frame, 1)

            with self._client_lock:
                need_cam = self.cam_clients > 0
                need_radar = self.radar_clients > 0

            vis, tracks, scan, dropoffs, overall, depth_snap, health = self.engine.step(
                frame, draw_keypoints=need_cam
            )
            self.beeper.level = overall

            if self.click:
                self.engine.calibrate_click(*self.click)
                self.click, self.calib = None, False
                self.calib_ack = time.time()
                self._persist()

            snap = self.snapshot(overall, tracks, scan, dropoffs, depth_snap, health)
            with self.event_cond:
                self.state = snap
                self.event_seq += 1
                self.event_cond.notify_all()

            if need_cam or need_radar:
                enc = [cv2.IMWRITE_JPEG_QUALITY, 80]
                new_jpg = {}
                if need_cam:
                    cam_img = draw_camera(
                        vis,
                        self.engine.cam,
                        tracks,
                        self.calib,
                        calib_dist=self.engine.calib_dist,
                        health=health,
                    )
                    new_jpg["cam"] = cv2.imencode(".jpg", cam_img, enc)[1].tobytes()
                if need_radar:
                    radar_img = draw_radar(
                        self.h,
                        self.engine.cam,
                        tracks,
                        self.engine.tracker,
                        overall,
                        scan=scan,
                        dropoffs=dropoffs,
                        overheads=self.engine.last_overheads,
                        guidance=self.engine.last_guidance,
                        radar_range=self.engine.radar_range,
                        labels=False,
                    )
                    new_jpg["radar"] = cv2.imencode(".jpg", radar_img, enc)[1].tobytes()
                with self.stream_cond:
                    self.jpg.update(new_jpg)
                    self.stream_cond.notify_all()

    def snapshot(self, overall, tracks, scan, dropoffs, depth_snap, health):
        eng = self.engine
        tr = eng.tracker
        cam = eng.cam
        ser_tracks = []
        for o in tracks:
            ser_tracks.append(
                {
                    "id": o.get("id"),
                    "cls": o.get("cls"),
                    "name": o.get("name"),
                    "X": round(float(o["X"]), 3),
                    "Z": round(float(o["Z"]), 3),
                    "r": round(float(o["r"]), 3),
                    "vx": round(float(o.get("vx", 0.0)), 3),
                    "vz": round(float(o.get("vz", 0.0)), 3),
                    "mode": o.get("mode", ""),
                    "dep": bool(o.get("dep", False)),
                    "closing_speed": round(float(o.get("closing_speed", 0.0)), 3),
                    "x_cpa": round(float(o.get("x_cpa", o["X"])), 3),
                    "cpa": round(float(o.get("cpa", o["r"])), 3),
                    "ttc": None if o.get("ttc") is None else round(float(o["ttc"]), 2),
                    "is_moving": bool(o.get("is_moving", False)),
                }
            )
        return {
            "t": time.time(),
            "level": int(overall),
            "fps": round(eng.fps, 1),
            "n": len(tracks),
            "worst_sector": tr.worst_sector(),
            "sectors": [
                {
                    "d": None if v == math.inf else round(v, 2),
                    "level": int(lv),
                    "dyn_level": int(dlv),
                    "ttc": None if ttc is None else round(ttc, 2),
                    "dropoff": bool(drp),
                    "overhead": bool(ovh),
                }
                for v, lv, dlv, ttc, drp, ovh in zip(
                    tr.d, tr.level, tr.dyn_level, tr.min_ttc, tr.dropoff, tr.overhead
                )
            ],
            "tracks": ser_tracks,
            "scan": [[round(float(x), 2), round(float(z), 2)] for x, z in scan],
            "dropoffs": [[round(float(x), 2), round(float(z), 2)] for x, z in dropoffs],
            "overheads": [[round(float(x), 2), round(float(z), 2)] for x, z in eng.last_overheads],
            "grid_cells": [
                [round(float(x), 2), round(float(z), 2), round(float(occ), 2), int(k)]
                for x, z, occ, k in eng.last_grid_cells
            ],
            "guidance": eng.last_guidance,
            "ego": eng.last_ego,
            "timings": eng.timings,
            "pitch_hint": eng.last_pitch_hint,
            "pitch": round(math.degrees(cam.pitch), 1),
            "height": round(cam.height, 2),
            "hfov": round(cam.hfov_deg, 1),
            "blind": round(cam.min_visible(), 2),
            "tilt": None if eng.sensor.tilt is None else round(eng.sensor.tilt, 1),
            "roll": round(eng.sensor.roll, 1),
            "offset": round(eng.offset, 1),
            "sensor_hz": round(eng.sensor.hz, 1),
            "sensor_source": eng.sensor.source,
            "compute_backend": accel.get_backend(),
            "ai_backend": eng.ai_backend,
            "depth_backend": eng.depth.runtime_label if eng.depth else "OFF",
            "depth": None if eng.depth is None else depth_snap,
            "health": health,
            "muted": self.beeper.muted,
            "voice_enabled": self.voice_enabled,
            "spatial_audio": self.spatial_audio,
            "flip": self.flip,
            "calib": self.calib,
            "calib_dist": eng.calib_dist,
            "calib_ack": self.calib_ack,
            "radar_range": eng.radar_range,
        }

    def command(self, c, q):
        eng = self.engine
        cam = eng.cam
        changed = True
        if c == "calib":
            self.calib = not self.calib
            changed = False
        elif c == "click" and self.calib:
            x, y = int(q["x"][0]), int(q["y"][0])
            if 0 <= x < self.w and 0 <= y < self.h:
                self.click = (x, y)
            changed = False
        elif c == "set_calib_dist":
            v = float(q.get("v", [2.0])[0])
            if 0.5 <= v <= 6.0:
                eng.calib_dist = v
        elif c in ("pitch_up", "pitch_down"):
            d = 1.0 if c == "pitch_up" else -1.0
            if eng.sensor.tilt is not None:
                eng.offset += d
            else:
                cam.pitch += math.radians(d)
        elif c == "h_up":
            cam.height += 0.05
        elif c == "h_down":
            cam.height = max(0.2, cam.height - 0.05)
        elif c in ("hfov_up", "hfov_down"):
            d = 2.0 if c == "hfov_up" else -2.0
            cam.set_hfov(cam.hfov_deg + d)
            eng.tracker.update_hfov(cam.hfov_deg)
        elif c == "set_hfov":
            v = float(q.get("v", [cam.hfov_deg])[0])
            cam.set_hfov(v)
            eng.tracker.update_hfov(cam.hfov_deg)
        elif c == "radar_range":
            v = float(q.get("v", [5.0])[0])
            if 2.0 <= v <= 15.0:
                eng.set_radar_range(v)
        elif c == "flip":
            self.flip = not self.flip
        elif c == "mute":
            self.beeper.muted = not self.beeper.muted
        elif c == "voice":
            self.voice_enabled = not self.voice_enabled
        elif c == "spatial":
            self.spatial_audio = not self.spatial_audio
        else:
            changed = False

        if changed:
            self._persist()


def _safe_state(app):
    st = dict(app.state)
    if not st or time.time() - st.get("t", 0) > 1.5:
        st["level"] = 2
        st["stale"] = True
    else:
        st["stale"] = False
    return st


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send_bytes(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                with open(PAGE_PATH, "rb") as f:
                    self._send_bytes(200, "text/html; charset=utf-8", f.read())
            elif url.path.startswith("/static/"):
                rel = os.path.normpath(url.path[len("/static/"):]).lstrip("\\/")
                full = os.path.abspath(os.path.join(STATIC_DIR, rel))
                if not full.startswith(os.path.abspath(STATIC_DIR)) or not os.path.isfile(full):
                    self.send_error(404)
                    return
                ctype, _ = mimetypes.guess_type(full)
                if full.endswith(".js"):
                    ctype = "application/javascript; charset=utf-8"
                elif not ctype:
                    ctype = "application/octet-stream"
                with open(full, "rb") as f:
                    self._send_bytes(200, ctype, f.read())
            elif url.path == "/state":
                body = json.dumps(_safe_state(app)).encode("utf-8")
                self._send_bytes(200, "application/json", body)
            elif url.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        with app.event_cond:
                            if app.event_seq == last_seq:
                                app.event_cond.wait(timeout=0.5)
                            last_seq = app.event_seq
                            payload = json.dumps(_safe_state(app)).encode("utf-8")
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
            elif url.path == "/cmd":
                q = parse_qs(url.query)
                app.command(q.get("c", [""])[0], q)
                self.send_response(204)
                self.end_headers()
            elif url.path == "/stream":
                view = parse_qs(url.query).get("v", ["cam"])[0]
                key = "radar" if view == "radar" else "cam"
                with app._client_lock:
                    if key == "cam":
                        app.cam_clients += 1
                    else:
                        app.radar_clients += 1
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with app.stream_cond:
                            app.stream_cond.wait(timeout=1.0)
                            jpg = app.jpg.get(key)
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    with app._client_lock:
                        if key == "cam":
                            app.cam_clients = max(0, app.cam_clients - 1)
                        else:
                            app.radar_clients = max(0, app.radar_clients - 1)
            else:
                self.send_error(404)

    return H


def main():
    ap = argparse.ArgumentParser(description="Radar 2D tu webcam - giao dien web")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--hfov", type=float, default=None, help=f"mac dinh {HFOV_DEG} hoac config.json")
    ap.add_argument("--height", type=float, default=None, help=f"mac dinh {CAM_HEIGHT} hoac config.json")
    ap.add_argument("--pitch", type=float, default=None, help=f"mac dinh {PITCH_DEG} hoac config.json")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--pose-model", type=str, default=POSE_MODEL)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--det-every", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-sensor", action="store_true")
    ap.add_argument("--no-ego", action="store_true")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx", "openvino"])
    ap.add_argument("--depth-backend", type=str, default=None, choices=["auto", "pytorch", "onnx", "openvino"])
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args()

    app = App(args)
    threading.Thread(target=app.loop, daemon=True).start()
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "<IP-laptop>"
    print(f"Mo: http://localhost:{args.port}   (dien thoai cung wifi: http://{ip}:{args.port})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
