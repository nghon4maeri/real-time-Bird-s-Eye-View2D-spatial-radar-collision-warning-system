# -*- coding: utf-8 -*-
"""Pipeline xu ly bat dong bo dung chung cho ca web.py va radar.py."""
import math
import os
import subprocess
import threading
import time

import cv2
import numpy as np

from core import accel
from core.camera import Camera
from core.config import (
    CLASSES,
    CONF,
    D_RED,
    LEVEL_COLOR,
    LEVEL_NAME,
    RADAR_RANGE,
    SECTOR_NAME,
)
from core.depth import DEPTH_HZ, DEPTH_HZ_FAST, DEPTH_HZ_IDLE, Depth
from core.ego import EgoMotionEstimator
from core.estimator import CameraHealthMonitor, estimate, person_distance
from core.tracker import SectorTracker, level_of, meas_cov

try:
    import winsound
except ImportError:
    winsound = None

try:
    import radar_rs as _rust
except ImportError:
    _rust = None


# ======================= AM THANH DESKTOP =======================
class Beeper(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.level = 0
        self.muted = False

    def run(self):
        while True:
            lv = 0 if self.muted else self.level
            if lv == 0:
                time.sleep(0.1)
                continue
            freq, dur, gap = (1800, 120, 0.08) if lv == 2 else (1000, 100, 0.45)
            if winsound:
                winsound.Beep(freq, dur)
            else:
                print("\a", end="", flush=True)
                time.sleep(dur / 1000.0)
            time.sleep(gap)


# ======================= TAI MODEL AI =======================
def load_yolo_model(model_path, backend="auto", imgsz=416, task="detect"):
    """Tai YOLO theo backend (auto|pytorch|onnx|openvino) voi fallback muot ve PyTorch."""
    import torch
    from ultralytics import YOLO

    has_cuda = torch.cuda.is_available()
    req = (backend or "auto").lower()
    if req == "auto":
        if has_cuda:
            req = "pytorch"
        else:
            try:
                import openvino  # noqa: F401
                req = "openvino"
            except ImportError:
                try:
                    import onnxruntime  # noqa: F401
                    req = "onnx"
                except ImportError:
                    req = "pytorch"

    if req == "onnx":
        try:
            import onnxruntime  # noqa: F401
            onnx_path = os.path.splitext(model_path)[0] + ".onnx"
            if not os.path.exists(onnx_path):
                YOLO(model_path).export(format="onnx", imgsz=imgsz)
            return YOLO(onnx_path, task=task), "ONNX", False
        except Exception as ex:
            print(f"[AI] Fallback PyTorch do loi ONNX ({model_path}): {ex}")

    if req == "openvino":
        try:
            import openvino  # noqa: F401
            ov_dir = os.path.splitext(model_path)[0] + "_openvino_model"
            if not os.path.exists(ov_dir):
                YOLO(model_path).export(format="openvino", imgsz=imgsz)
            return YOLO(ov_dir, task=task), "OpenVINO", False
        except Exception as ex:
            print(f"[AI] Fallback PyTorch do loi OpenVINO ({model_path}): {ex}")

    model = YOLO(model_path)
    if has_cuda:
        return model, "PyTorch CUDA", False
    return model, "PyTorch CPU", False


# ======================= DOC CAMERA DA LUONG =======================
class FrameReader(threading.Thread):
    """Doc khung hinh o luong rieng, luon giu frame moi nhat de triet tieu buffer lag."""

    def __init__(self, source, width=640, height=480, realtime=True):
        super().__init__(daemon=True)
        self.is_video = isinstance(source, str)
        self.cap = cv2.VideoCapture(source)
        if not self.is_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Khong mo duoc camera/video.")
        self.h, self.w = frame.shape[:2]
        cap_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_fps = min(max(float(cap_fps), 5.0), 60.0)
        self.realtime = realtime
        self._lock = threading.Lock()
        self._latest = (True, frame, time.time(), 1)
        self._seq = 1
        self._stop = False
        self._last_read_t = time.time()

    def run(self):
        period = 1.0 / self.video_fps if self.is_video and self.realtime else 0.0
        while not self._stop:
            if period > 0.0:
                dt = time.time() - self._last_read_t
                if dt < period:
                    time.sleep(period - dt)
            self._last_read_t = time.time()
            ok, frame = self.cap.read()
            if not ok:
                if self.is_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with self._lock:
                    self._latest = (False, None, time.time(), self._seq)
                time.sleep(0.1)
                continue
            with self._lock:
                self._seq += 1
                self._latest = (True, frame, time.time(), self._seq)

    def read_latest(self, last_seq=0):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop = True
        try:
            self.cap.release()
        except Exception:
            pass


# ======================= ASYNC DEPTH WORKER =======================
class AsyncDepthWorker(threading.Thread):
    """Chay Depth Anything V2 tren luong nen voi tan suat thich ung (Adaptive Cadence)."""

    def __init__(self, depth_model, radar_range=RADAR_RANGE):
        super().__init__(daemon=True)
        self.depth = depth_model
        self.radar_range = float(radar_range)
        self.target_hz = DEPTH_HZ
        self._cond = threading.Condition()
        self._pending = None
        self._stop = False
        self._snap_lock = threading.Lock()
        self._snap = {
            "obstacles": [],
            "dropoffs": [],
            "overheads": [],
            "ready": False,
            "ms": 0.0,
            "anchors": 0,
            "hz": DEPTH_HZ,
            "runtime": getattr(depth_model, "runtime_label", "PyTorch"),
            "pitch_hint": None,
        }

    def set_range(self, r):
        self.radar_range = float(r)

    def set_target_hz(self, hz):
        self.target_hz = float(hz)

    def submit(self, frame, objs, copy_frame=False):
        now = time.time()
        if now - self.depth.t_last < 1.0 / max(self.target_hz, 0.5):
            return
        anchors = [
            {"box": tuple(float(v) for v in o["box"]), "Z": float(o["Z"]), "mode": o["mode"]}
            for o in objs
        ]
        slot_frame = frame.copy() if copy_frame else frame
        with self._cond:
            self._pending = (slot_frame, anchors, self.target_hz)
            self._cond.notify()

    def measure_sync(self, box):
        with self._snap_lock:
            return self.depth.measure(box)

    def snapshot(self):
        with self._snap_lock:
            return dict(self._snap)

    def run(self):
        while not self._stop:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.5)
                if self._stop:
                    break
                frame, objs, hz = self._pending
                self._pending = None
            try:
                self.depth.step(frame, objs, max_r=self.radar_range, target_hz=hz)
                with self._snap_lock:
                    self._snap = {
                        "obstacles": list(self.depth.obstacles),
                        "dropoffs": list(self.depth.dropoffs),
                        "overheads": list(self.depth.overheads),
                        "ready": self.depth.ready(),
                        "ms": round(self.depth.ms, 1),
                        "anchors": self.depth.n_anchor,
                        "hz": round(hz, 1),
                        "runtime": self.depth.runtime_label,
                        "pitch_hint": self.depth.pitch_hint,
                    }
            except Exception:
                pass

    def stop(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()


# ======================= CAM BIEN NGHIENG =======================
class SensorReader:
    """Doc gia toc ke man hinh: uu tien Rust WinRT neu co, fallback PowerShell."""

    def __init__(self, enabled=True):
        self.tilt = None
        self.roll = 0.0
        self.hz = 0.0
        self.source = "none"
        self._stop = False
        self._rust_sensor = None
        if enabled:
            self._start()

    def _start(self):
        if _rust is not None and hasattr(_rust, "WinAccelerometer"):
            try:
                self._rust_sensor = _rust.WinAccelerometer()
                self.source = "rust-winrt"
                threading.Thread(target=self._poll_rust, daemon=True).start()
                return
            except Exception:
                self._rust_sensor = None
        self.source = "powershell"
        threading.Thread(target=self._run_powershell, daemon=True).start()

    def _poll_rust(self):
        while not self._stop and self._rust_sensor is not None:
            pitch, roll, hz = self._rust_sensor.read()
            if pitch is not None:
                self.tilt = pitch
                self.roll = roll
                self.hz = hz
            time.sleep(0.02)

    def _run_powershell(self):
        ps = (
            "[Windows.Devices.Sensors.Accelerometer,Windows.Devices.Sensors,ContentType=WindowsRuntime]|Out-Null;"
            "$s=[Windows.Devices.Sensors.Accelerometer]::GetDefault(); if(!$s){exit};"
            "$c=[cultureinfo]::InvariantCulture; while($true){$r=$s.GetCurrentReading();"
            "try{[Console]::WriteLine([string]::Format($c,'{0} {1} {2}',$r.AccelerationX,$r.AccelerationY,$r.AccelerationZ));"
            "[Console]::Out.Flush()}catch{exit}; Start-Sleep -Milliseconds 100}"
        )
        try:
            p = subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, AttributeError):
            self.source = "none"
            return
        t0, count = time.time(), 0
        for line in p.stdout:
            if self._stop:
                break
            try:
                x, y, z = map(float, line.split())
            except ValueError:
                continue
            pitch = math.degrees(math.atan2(z, -y))
            roll = math.degrees(math.atan2(x, -y))
            self.tilt = pitch if self.tilt is None else self.tilt + 0.2 * (pitch - self.tilt)
            self.roll += 0.2 * (roll - self.roll)
            count += 1
            now = time.time()
            if now - t0 >= 1.0:
                self.hz = count / (now - t0)
                t0, count = now, 0
        try:
            p.terminate()
        except Exception:
            pass

    def stop(self):
        self._stop = True
        if self._rust_sensor is not None:
            try:
                self._rust_sensor.stop()
            except Exception:
                pass


# ======================= VE HINH OPENCV (CHO STREAM & DESKTOP) =======================
def put(img, text, org, color=(230, 230, 230), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_camera(frame, cam, objs, calib, calib_dist=2.0, health=None):
    img = frame.copy()
    p0, p1 = cam.horizon_line()
    if (0 <= p0[1] < cam.h) or (0 <= p1[1] < cam.h):
        cv2.line(img, p0, p1, (150, 150, 150), 1, cv2.LINE_AA)
        put(img, "chan troi", (6, max(14, min(cam.h - 6, p0[1] - 5))), (180, 180, 180), 0.4)
    for o in objs:
        if "box" not in o or o["box"] is None:
            continue
        x1, y1, x2, y2 = map(int, o["box"])
        col = LEVEL_COLOR[level_of(o["r"])]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.circle(img, ((x1 + x2) // 2, y2), 4, col, -1)
        mark = {"sat": "<", "hc": "~", "che": "~", "pose": "*", "depth": "d"}.get(o.get("mode", ""), "")
        mark += "d" if o.get("dep") else ""
        ttc_txt = f" {o['ttc']:.1f}s" if o.get("ttc") is not None and o["ttc"] < 4.0 else ""
        tag = f'{o["name"]} {mark}{o["r"]:.1f}m{ttc_txt}'
        put(img, tag, (x1 + 2, max(y1 - 6, 14)), col, 0.5)
    if calib:
        cv2.rectangle(img, (0, 0), (cam.w, 50), (0, 0, 0), -1)
        put(img, f"HIEU CHINH: dat vat cach dung {calib_dist:.1f} m,", (8, 20), (0, 255, 255))
        put(img, "click vao diem vat cham san (c de huy)", (8, 42), (0, 255, 255))
    if health and health.get("alert"):
        cv2.rectangle(img, (0, 0), (cam.w, cam.h), (0, 0, 180), 6)
        put(img, "CAMERA BI CHE / THIEU SANG!", (18, cam.h // 2), (0, 0, 255), 0.8, 2)
    return img


def draw_radar(
    size,
    cam,
    objs,
    tracker,
    overall,
    scan=(),
    dropoffs=(),
    overheads=(),
    guidance=None,
    radar_range=RADAR_RANGE,
    labels=True,
):
    img = np.full((size, size, 3), 22, np.uint8)
    ox, oy = size // 2, size - 25
    R = size - 60
    scale = R / max(radar_range, 1.0)
    half = cam.hfov_deg / 2.0
    a0, a1 = -90 - half, -90 + half
    edges = [a0, -90 - half / 3, -90 + half / 3, a1]

    overlay = img.copy()
    for i in range(3):
        lv = tracker.level[i]
        if lv > 0:
            cv2.ellipse(overlay, (ox, oy), (R, R), 0, edges[i], edges[i + 1], LEVEL_COLOR[lv], -1)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)

    rb = int(cam.min_visible() * scale)
    if rb > 0:
        cv2.ellipse(img, (ox, oy), (rb, rb), 0, a0, a1, (60, 60, 60), -1)
        put(img, "vung mu", (ox - 28, oy - rb // 2), (140, 140, 140), 0.4)

    for d, col in ((D_RED, (0, 0, 220)), (2.5, (0, 180, 230))):
        if d <= radar_range:
            cv2.ellipse(img, (ox, oy), (int(d * scale),) * 2, 0, a0, a1, col, 2, cv2.LINE_AA)
    for r in range(1, int(radar_range) + 1):
        cv2.ellipse(img, (ox, oy), (int(r * scale),) * 2, 0, a0, a1, (70, 70, 70), 1, cv2.LINE_AA)
        ang = math.radians(a1)
        put(img, f"{r}m", (int(ox + r * scale * math.cos(ang)) + 4, int(oy + r * scale * math.sin(ang))), (150, 150, 150), 0.4)
    for e in edges:
        ang = math.radians(e)
        cv2.line(img, (ox, oy), (int(ox + R * math.cos(ang)), int(oy + R * math.sin(ang))),
                 (90, 90, 90) if e in (a0, a1) else (50, 50, 50), 1, cv2.LINE_AA)

    for x, z in scan:
        cv2.circle(img, (int(ox + x * scale), int(oy - z * scale)), 2, (170, 170, 170), -1)
    for x, z in dropoffs:
        cv2.circle(img, (int(ox + x * scale), int(oy - z * scale)), 3, (0, 140, 255), -1)
    for x, z in overheads:
        cv2.circle(img, (int(ox + x * scale), int(oy - z * scale)), 3, (220, 80, 200), -1)

    if guidance and guidance.get("action") != "stop":
        rad = math.radians(guidance["target_deg"])
        len_m = min(guidance["clearance_m"], radar_range * 0.85)
        gx = int(ox + len_m * math.sin(rad) * scale)
        gy = int(oy - len_m * math.cos(rad) * scale)
        cv2.arrowedLine(img, (ox, oy), (gx, gy), (255, 200, 50), 2, cv2.LINE_AA, tipLength=0.18)

    for o in objs:
        if o["Z"] > radar_range + 0.3:
            continue
        px, py = int(ox + o["X"] * scale), int(oy - o["Z"] * scale)
        col = LEVEL_COLOR[level_of(o["r"])]
        if "vx" in o and math.hypot(o["vx"], o["vz"]) > 0.2:
            cv2.arrowedLine(img, (px, py), (int(px + o["vx"] * scale), int(py - o["vz"] * scale)),
                            col, 1, cv2.LINE_AA, tipLength=0.25)
        if o["cls"] == 0:
            cv2.circle(img, (px, py), 8, col, -1)
        else:
            cv2.rectangle(img, (px - 7, py - 7), (px + 7, py + 7), col, -1)
        put(img, o["name"], (px + 10, py + 4), (220, 220, 220), 0.4)

    cv2.fillPoly(img, [np.array([[ox, oy - 12], [ox - 9, oy + 6], [ox + 9, oy + 6]])], (255, 255, 255))

    if not labels:
        return img
    put(img, LEVEL_NAME[overall], (12, 32), LEVEL_COLOR[overall], 0.9, 2)
    for i, name in enumerate(SECTOR_NAME):
        d = tracker.d[i]
        txt = f"{name}: {d:.1f}m" if d < math.inf else f"{name}: --"
        put(img, txt, (size - 130, 22 + i * 20), LEVEL_COLOR[tracker.level[i]], 0.5)
    return img


# ======================= ENGINE CHUNG =======================
class RadarEngine:
    """Bo xu ly trung tam: Detection + Pose + Async Depth + Ego + Occupancy Grid + Guidance."""

    def __init__(
        self,
        w,
        h,
        hfov,
        height,
        pitch,
        model_path="yolo11n.pt",
        pose_model_path="yolo11n-pose.pt",
        imgsz=320,
        det_every=3,
        use_pose=True,
        use_depth=True,
        use_kalman=True,
        use_sensor=True,
        use_ego=True,
        backend="auto",
        depth_backend=None,
        radar_range=RADAR_RANGE,
        calib_dist=2.0,
        offset=0.0,
    ):
        self.w, self.h = int(w), int(h)
        self.imgsz = int(imgsz)
        self.det_every = max(1, int(det_every))
        self.radar_range = float(radar_range)
        self.calib_dist = float(calib_dist)
        self.offset = float(offset)
        self.k = 0
        self.fps = 0.0
        self._t_prev = time.time()
        self._last_health_alert = False

        self.model, self.ai_backend, _ = load_yolo_model(
            model_path, backend=backend, imgsz=self.imgsz, task="detect"
        )
        self.pose = None
        if use_pose:
            self.pose, _, _ = load_yolo_model(
                pose_model_path, backend=backend, imgsz=self.imgsz, task="pose"
            )

        self.cam = Camera(self.w, self.h, hfov, height, pitch)
        self.tracker = SectorTracker(hfov)
        self.kf = accel.KalmanTracker() if use_kalman else None
        self.grid = accel.OccupancyGrid2D()
        self.ego = EgoMotionEstimator() if use_ego else None
        self.health_mon = CameraHealthMonitor(duration_thresh=1.0)
        self.sensor = SensorReader(enabled=use_sensor)

        self.depth = None
        self.depth_worker = None
        self.depth_backend = depth_backend or backend
        if use_depth:
            try:
                self.depth = Depth(self.cam, use_fp16=True, backend=self.depth_backend)
                self.depth_worker = AsyncDepthWorker(self.depth, radar_range=self.radar_range)
                self.depth_worker.start()
            except ImportError as ex:
                print("Khong co torch/transformers -> tat depth:", ex)

        self.last_ego = {"d_yaw": 0.0, "d_x": 0.0, "d_z": 0.0, "inliers": 0, "conf": 0.0}
        self.last_guidance = {"action": "straight", "target_deg": 0.0, "clearance_m": 5.0}
        self.last_grid_cells = []
        self.last_overheads = []
        self.last_pitch_hint = None
        self.timings = {
            "yolo_ms": 0.0,
            "pose_ms": 0.0,
            "depth_ms": 0.0,
            "ego_ms": 0.0,
            "core_ms": 0.0,
            "total_ms": 0.0,
        }

    def set_radar_range(self, r):
        self.radar_range = float(r)
        if self.depth_worker is not None:
            self.depth_worker.set_range(self.radar_range)

    def _ema_time(self, key, val_ms):
        prev = self.timings[key]
        self.timings[key] = round(val_ms if prev == 0.0 else prev + 0.2 * (val_ms - prev), 2)

    def step(self, frame, draw_keypoints=False, now=None):
        """Xu ly 1 frame. Tra ve (vis_frame, tracks_or_objs, scan_pts, dropoffs, overall, depth_snap, health)."""
        t_start = time.perf_counter()
        t_wall = time.time() if now is None else float(now)
        dt = max(t_wall - self._t_prev, 1e-4)
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps > 0 else 1.0 / dt
        self._t_prev = t_wall

        health = self.health_mon.update(frame, t_wall)

        if self.sensor.tilt is not None:
            self.cam.pitch = math.radians(self.sensor.tilt + self.offset)
            self.cam.roll = math.radians(self.sensor.roll)

        self.k += 1
        dets = []
        t0 = time.perf_counter()
        if not self.pose or self.k % self.det_every == 0:
            classes = [c for c in CLASSES if c != 0] if self.pose else list(CLASSES)
            res = self.model.predict(
                frame, imgsz=self.imgsz, conf=CONF, classes=classes, verbose=False
            )[0]
            if res.boxes is not None and len(res.boxes):
                dets = [
                    (b, c, None)
                    for b, c in zip(
                        res.boxes.xyxy.cpu().numpy(),
                        res.boxes.cls.cpu().numpy().astype(int),
                    )
                ]
        yolo_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        if self.pose:
            pr = self.pose.predict(
                frame, imgsz=self.imgsz, conf=CONF, verbose=False
            )[0]
            if pr.boxes is not None and len(pr.boxes):
                kps = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else []
                dets += [(b, 0, k) for b, k in zip(pr.boxes.xyxy.cpu().numpy(), kps)]
        pose_ms = (time.perf_counter() - t1) * 1000.0

        t_core0 = time.perf_counter()
        vis = frame.copy() if (draw_keypoints and self.pose is not None) else frame
        objs = []
        for box, c, kp in dets:
            e = estimate(self.cam, box, c)
            if kp is not None and (e is None or e[2] in ("sat", "hc", "cao")):
                p = person_distance(self.cam, box, kp)
                if p:
                    e = (*p, "pose")
                if draw_keypoints and vis is not frame:
                    for x, y, s in kp:
                        if s > 0.5:
                            cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 0), -1)
            if e is None and self.depth_worker is not None:
                p = self.depth_worker.measure_sync(box)
                if p:
                    e = (*p, "depth")
            if e:
                X, Z, mode = e
                objs.append(
                    dict(
                        box=box,
                        cls=int(c),
                        name=CLASSES.get(int(c), str(c)),
                        X=float(X),
                        Z=float(Z),
                        r=math.hypot(X, Z),
                        mode=mode,
                    )
                )

        accel.resolve_occlusion(objs)

        for o in objs:
            o["meas"] = [(np.array([o["X"], o["Z"]]), meas_cov(o["Z"], o["mode"]))]
            if (
                self.depth_worker is not None
                and o["mode"] not in ("ok", "pose", "san", "depth")
            ):
                p = self.depth_worker.measure_sync(o["box"])
                if p:
                    o["meas"].append((np.array(p), meas_cov(p[1], "depth")))
                    o["dep"] = True

        # Ego-motion compensation
        t_ego0 = time.perf_counter()
        if self.ego is not None:
            self.last_ego = self.ego.estimate(frame, self.cam, objs)
        else:
            self.last_ego = {"d_yaw": 0.0, "d_x": 0.0, "d_z": 0.0, "inliers": 0, "conf": 0.0}
        ego_ms = (time.perf_counter() - t_ego0) * 1000.0
        ego_vec = (self.last_ego["d_yaw"], self.last_ego["d_x"], self.last_ego["d_z"])

        if self.depth_worker is not None:
            depth_snap = self.depth_worker.snapshot()
        else:
            depth_snap = {
                "obstacles": [],
                "dropoffs": [],
                "overheads": [],
                "ready": False,
                "ms": 0.0,
                "anchors": 0,
                "hz": 0.0,
                "runtime": "OFF",
                "pitch_hint": None,
            }

        scan = depth_snap["obstacles"]
        dropoffs = depth_snap["dropoffs"]
        overheads = depth_snap.get("overheads", [])
        self.last_overheads = overheads
        self.last_pitch_hint = depth_snap.get("pitch_hint") if self.sensor.tilt is None else None

        if self.kf is not None:
            tracks = self.kf.step(
                objs, t_wall, det_every=self.det_every, fps=max(self.fps, 1.0), ego=ego_vec
            )
        else:
            tracks = objs

        guidance, grid_cells = self.grid.step(
            ego_vec[0],
            ego_vec[1],
            ego_vec[2],
            dt,
            self.cam.hfov_deg,
            self.cam.min_visible(),
            obstacles=scan,
            dropoffs=dropoffs,
            overheads=overheads,
            tracks=tracks,
        )
        self.last_guidance = guidance
        self.last_grid_cells = grid_cells

        scan_objs = [dict(X=x, Z=z, r=math.hypot(x, z)) for x, z in scan]
        overall = self.tracker.update(tracks + scan_objs, dropoffs=dropoffs, overheads=overheads)
        if health["alert"]:
            overall = 2

        # Adaptive Depth Cadence
        if self.depth_worker is not None:
            health_changed = bool(health["alert"]) != self._last_health_alert
            self._last_health_alert = bool(health["alert"])
            target_hz = DEPTH_HZ_FAST if (overall >= 1 or health_changed) else DEPTH_HZ_IDLE
            self.depth_worker.set_target_hz(target_hz)
            self.depth_worker.submit(frame, objs, copy_frame=False)

        core_ms = (time.perf_counter() - t_core0) * 1000.0 - ego_ms
        total_ms = (time.perf_counter() - t_start) * 1000.0

        self._ema_time("yolo_ms", yolo_ms)
        self._ema_time("pose_ms", pose_ms)
        self._ema_time("depth_ms", float(depth_snap.get("ms", 0.0)))
        self._ema_time("ego_ms", ego_ms)
        self._ema_time("core_ms", max(0.0, core_ms))
        self._ema_time("total_ms", total_ms)

        return vis, tracks, scan, dropoffs, overall, depth_snap, health

    def calibrate_click(self, x, y):
        self.cam.calibrate_pitch(x, y, self.calib_dist)
        if self.sensor.tilt is not None:
            self.offset = math.degrees(self.cam.pitch) - self.sensor.tilt

    def stop(self):
        self.sensor.stop()
        if self.depth_worker is not None:
            self.depth_worker.stop()
