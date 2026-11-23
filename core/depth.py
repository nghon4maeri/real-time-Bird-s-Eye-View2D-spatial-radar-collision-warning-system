# -*- coding: utf-8 -*-
"""Depth Anything V2 Small voi ho tro ONNX/OpenVINO/PyTorch, loc cum, drop-off, overhead va auto-pitch."""
import math
import os
import time

import cv2
import numpy as np

from core import accel

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
ONNX_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "depth_v2_small.onnx"
)
IN_W, IN_H = 252, 196
GW, GH = 84, 64
GW_RUST, GH_RUST = 168, 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ANCHOR_MODES = ("ok", "pose", "san")
OBST_H = (0.15, 1.9)
DROP_H = -0.25
OVER_H = (1.35, 2.05)
MIN_CLUSTER_COLS = 3
MIN_CONFIRM = 2
FIT_EMA = 0.3
DEPTH_HZ = 4.0
DEPTH_HZ_FAST = 6.0
DEPTH_HZ_IDLE = 2.5


class Depth:
    """Uoc luong depth don mat (PyTorch/ONNX/OpenVINO), quy doi met va quet vat can/bac thang/vat treo."""

    def __init__(self, cam, grid=None, use_fp16=True, backend="auto", onnx_path=ONNX_DEFAULT):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = bool(use_fp16 and self.device == "cuda")
        self.cam = cam
        self.onnx_path = onnx_path

        # Pre-allocated preprocessing buffers (tranh cap phat thua moi frame)
        self._resized = np.empty((IN_H, IN_W, 3), dtype=np.uint8)
        self._in_buf = np.empty((1, 3, IN_H, IN_W), dtype=np.float32)

        self.model = None
        self.ort_session = None
        self.ort_input_name = "pixel_values"
        self.ov_compiled = None
        self.runtime_label = "PyTorch CPU"
        self._init_backend(backend)

        if grid is None:
            self.gw, self.gh = (GW_RUST, GH_RUST) if accel.get_backend() == "rust" else (GW, GH)
        else:
            self.gw, self.gh = int(grid[0]), int(grid[1])

        self.t_last = 0.0
        self.disp = None
        self.fit = None
        self.ms = 0.0
        self.n_anchor = 0
        self.obstacles = []
        self.dropoffs = []
        self.overheads = []
        self._floor_err_ema = None
        self._floor_err_count = 0
        self.pitch_hint = None
        self._init_grid_state(self.gw, self.gh)

    def _load_pytorch_model(self):
        from transformers import AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
        return model

    def _ensure_onnx(self):
        if os.path.exists(self.onnx_path):
            return self.onnx_path
        model = self._load_pytorch_model().cpu()
        dummy = self.torch.zeros(1, 3, IN_H, IN_W, dtype=self.torch.float32)
        with self.torch.inference_mode():
            self.torch.onnx.export(
                model,
                dummy,
                self.onnx_path,
                input_names=["pixel_values"],
                output_names=["predicted_depth"],
                opset_version=17,
            )
        return self.onnx_path

    def _init_backend(self, backend):
        req = (backend or "auto").lower()
        if req == "auto":
            if self.device == "cuda":
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

        if req == "openvino":
            try:
                import openvino as ov

                onnx_p = self._ensure_onnx()
                core = ov.Core()
                ov_model = core.read_model(onnx_p)
                self.ov_compiled = core.compile_model(ov_model, "CPU")
                self.runtime_label = "OpenVINO CPU"
                return
            except Exception as ex:
                print(f"[Depth] Fallback PyTorch do loi OpenVINO: {ex}")

        if req == "onnx":
            try:
                import onnxruntime as ort

                onnx_p = self._ensure_onnx()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self.ort_session = ort.InferenceSession(onnx_p, providers=providers)
                self.ort_input_name = self.ort_session.get_inputs()[0].name
                used = self.ort_session.get_providers()[0]
                self.runtime_label = "ONNX CUDA" if "CUDA" in used else "ONNX CPU"
                return
            except Exception as ex:
                print(f"[Depth] Fallback PyTorch do loi ONNX: {ex}")

        self.model = self._load_pytorch_model().to(self.device)
        if self.fp16:
            self.model = self.model.half()
            self.runtime_label = "PyTorch CUDA FP16"
        else:
            self.runtime_label = "PyTorch CUDA" if self.device == "cuda" else "PyTorch CPU"

    def _init_grid_state(self, gw, gh):
        self.gw, self.gh = gw, gh
        self.obst_streak = np.zeros(gw, dtype=np.int32)
        self.drop_streak = np.zeros(gw, dtype=np.int32)
        self.over_streak = np.zeros(gw, dtype=np.int32)
        self.prev_obst_z = np.zeros(gw, dtype=np.float64)
        self.prev_drop_z = np.zeros(gw, dtype=np.float64)
        self.prev_over_z = np.zeros(gw, dtype=np.float64)

    def _ensure_state(self, gw, gh):
        if not hasattr(self, "obst_streak") or len(self.obst_streak) != gw or not hasattr(self, "over_streak"):
            self._init_grid_state(gw, gh)

    def _preprocess(self, frame):
        cv2.resize(frame, (IN_W, IN_H), dst=self._resized, interpolation=cv2.INTER_AREA)
        # BGR uint8 -> RGB normalized CHW float32 in-place
        rgb = self._resized[:, :, ::-1]
        for ch in range(3):
            self._in_buf[0, ch] = (rgb[:, :, ch] * (1.0 / 255.0) - MEAN[ch]) / STD[ch]
        return self._in_buf

    def _infer_depth(self, frame):
        inp = self._preprocess(frame)
        if self.ov_compiled is not None:
            res = self.ov_compiled([inp])
            return np.asarray(next(iter(res.values()))[0], dtype=np.float32)
        if self.ort_session is not None:
            out = self.ort_session.run(None, {self.ort_input_name: inp})[0]
            return np.asarray(out[0], dtype=np.float32)
        with self.torch.inference_mode():
            dtype = self.torch.float16 if getattr(self, "fp16", False) else self.torch.float32
            pixel_values = self.torch.from_numpy(inp).to(self.device, dtype=dtype)
            d = self.model(pixel_values=pixel_values).predicted_depth[0].float().cpu().numpy()
        return d

    def step(self, frame, objs, max_r=5.0, target_hz=DEPTH_HZ):
        """Chay model neu da den luot (target_hz), fit thang do va quet vat can + dropoff + overhead."""
        t = time.time()
        if t - self.t_last < 1.0 / max(target_hz, 0.5):
            return
        self.t_last = t
        d = self._infer_depth(frame)
        disp = cv2.medianBlur(cv2.resize(d, (self.gw, self.gh)).astype(np.float32), 5)
        self._fit(disp, [(o["box"], o["Z"], o["mode"]) for o in objs])
        self.disp = disp
        if self.fit is not None:
            self.scan_all(max_r)
        self.ms = (time.time() - t) * 1000.0

    def _cell(self, box):
        """Vung giua box (tranh nen phia sau) tren luoi."""
        gw = self.disp.shape[1] if self.disp is not None else getattr(self, "gw", GW)
        gh = self.disp.shape[0] if self.disp is not None else getattr(self, "gh", GH)
        x1, y1, x2, y2 = box
        sx, sy = gw / self.cam.w, gh / self.cam.h
        w, h = x2 - x1, y2 - y1
        r0 = max(0, min(gh - 1, int((y1 + 0.3 * h) * sy)))
        r1 = min(gh, max(r0 + 1, int((y1 + 0.7 * h) * sy)))
        c0 = max(0, min(gw - 1, int((x1 + 0.3 * w) * sx)))
        c1 = min(gw, max(c0 + 1, int((x1 + 0.7 * w) * sx)))
        return slice(r0, r1), slice(c0, c1)

    def _fit(self, disp, items, backend=None):
        """Fit 1/zc = a * disp + b voi mask vat can khoi san + RANSAC/Huber."""
        gh, gw = disp.shape
        self._ensure_state(gw, gh)
        anchor_boxes, anchor_zs, mask_boxes = [], [], []
        for box, z, mode in items:
            mask_boxes.append(box)
            if mode in ANCHOR_MODES and z is not None and math.isfinite(z) and z > 0.05:
                anchor_boxes.append(box)
                anchor_zs.append(z)

        cam = self.cam
        res = accel.depth_fit_ransac(
            disp,
            anchor_boxes,
            anchor_zs,
            mask_boxes,
            cam.w,
            cam.h,
            cam.cx,
            cam.cy,
            cam.f,
            cam.height,
            cam.pitch,
            getattr(cam, "roll", 0.0),
            cam.min_visible(),
            self.fit,
            backend=backend,
        )
        if res is None:
            self.n_anchor = 0
            return
        a, b, n_anc = res
        self.fit = (a, b)
        self.n_anchor = n_anc

    def _zc(self, disp):
        a, b = self.fit
        inv = a * disp + b
        return np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), np.inf)

    def ready(self):
        return self.disp is not None and self.fit is not None

    def measure(self, box):
        """Khoang cach (X, Z) tren san cua vat trong box, theo depth."""
        if not self.ready():
            return None
        r, c = self._cell(box)
        zc = float(np.median(self._zc(self.disp[r, c])))
        if not math.isfinite(zc):
            return None
        gh, gw = self.disp.shape
        u_mean = 0.5 * (c.start + c.stop) * self.cam.w / gw
        v_mean = 0.5 * (r.start + r.stop) * self.cam.h / gh
        xr, yr = self.cam.unroll((u_mean - self.cam.cx) / self.cam.f, (v_mean - self.cam.cy) / self.cam.f)
        cp, sp = math.cos(self.cam.pitch), math.sin(self.cam.pitch)
        return zc * xr, zc * (cp - yr * sp)

    def _update_pitch_hint(self, floor_err):
        if math.isnan(floor_err) or getattr(self, "n_anchor", 0) < 3:
            self._floor_err_count = 0
            self.pitch_hint = None
            return
        prev_ema = getattr(self, "_floor_err_ema", None)
        self._floor_err_ema = floor_err if prev_ema is None else prev_ema + 0.15 * (floor_err - prev_ema)
        self._floor_err_count = getattr(self, "_floor_err_count", 0) + 1
        if self._floor_err_count >= 10:
            # Sai so chieu cao san duong (hgt > 0) nghia la pitch uoc luong hoi nho/cao
            delta_deg = max(-8.0, min(8.0, math.degrees(math.atan2(self._floor_err_ema, 1.8))))
            if abs(delta_deg) >= 1.5:
                cur_deg = math.degrees(self.cam.pitch)
                self.pitch_hint = {
                    "suggested_deg": round(cur_deg + delta_deg, 1),
                    "delta_deg": round(delta_deg, 1),
                }
            else:
                self.pitch_hint = None

    def scan_all(self, max_r, min_confirm=MIN_CONFIRM, min_cluster_cols=MIN_CLUSTER_COLS, backend=None):
        """Tra ve (obstacles, dropoffs) va cap nhat self.overheads + auto-pitch hint."""
        if not self.ready():
            return [], []
        gh, gw = self.disp.shape
        self._ensure_state(gw, gh)
        cam = self.cam
        obst, drop, over, nos, nds, novs, noz, ndz, novz, ferr = accel.depth_scan_and_cluster(
            self.disp,
            self.fit[0],
            self.fit[1],
            cam.w,
            cam.h,
            cam.cx,
            cam.cy,
            cam.f,
            cam.height,
            cam.pitch,
            getattr(cam, "roll", 0.0),
            max_r,
            self.obst_streak,
            self.drop_streak,
            poz := self.prev_obst_z,
            self.prev_drop_z,
            OBST_H[0],
            OBST_H[1],
            DROP_H,
            min_cluster_cols,
            min_confirm,
            prev_over_streak=self.over_streak,
            prev_over_z=self.prev_over_z,
            over_h_min=OVER_H[0],
            over_h_max=OVER_H[1],
            backend=backend,
        )
        del poz
        self.obst_streak = nos
        self.drop_streak = nds
        self.over_streak = novs
        self.prev_obst_z = noz
        self.prev_drop_z = ndz
        self.prev_over_z = novz
        self.obstacles = obst
        self.dropoffs = drop
        self.overheads = over
        self._update_pitch_hint(ferr)
        return obst, drop

    def scan(self, max_r, min_confirm=MIN_CONFIRM, min_cluster_cols=MIN_CLUSTER_COLS, backend=None):
        """Giu tuong thich: tra ve danh sach vat can [(X, Z)]."""
        obst, _ = self.scan_all(
            max_r, min_confirm=min_confirm, min_cluster_cols=min_cluster_cols, backend=backend
        )
        return obst
