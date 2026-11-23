import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import depth as D
from core import accel
from core.camera import Camera
from core.ego import EgoMotionEstimator
from core.tracker import SectorTracker, meas_cov


def test_ego_motion_yaw_and_mask():
    cam = Camera(320, 240, 70.0, 1.5, 20.0)
    rng = np.random.default_rng(123)
    bg = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(180):
        x = int(rng.integers(10, 310))
        y = int(rng.integers(10, 230))
        cv2.circle(bg, (x, y), int(rng.integers(2, 5)), (220, 220, 220), -1)

    ego = EgoMotionEstimator(max_corners=80, proc_w=320)
    r0 = ego.estimate(bg, cam, objs=[])
    assert r0["inliers"] == 0

    # Gia lap camera quay sang phai ~0.04 rad (anh dich sang trai ~dx_px = -cam.f * 0.04)
    shift_px = cam.f * 0.04
    M = np.float32([[1, 0, -shift_px], [0, 1, 0]])
    bg2 = cv2.warpAffine(bg, M, (320, 240))
    # Them vat di chuyen nguoc chieu trong box (phai bi mask khong lam lech ego)
    cv2.rectangle(bg2, (120, 80), (200, 220), (40, 180, 250), -1)
    r1 = ego.estimate(bg2, cam, objs=[{"box": (120, 80, 200, 220)}])
    assert r1["inliers"] >= 10
    assert abs(r1["d_yaw"] - 0.04) < 0.015


def test_kalman_ego_keeps_static_object_still():
    # Nguoi dung di thang 0.1 m/khung ve phia truoc -> ghe tinh tien gan tu 3.0m ve 2.0m
    # Khi co ego=(0, 0, 0.1), van toc thuc cua ghe phai gan 0 (is_moving=False)
    kf = accel.KalmanTracker()
    for k in range(10):
        Z_meas = 3.0 - 0.1 * k
        ego = (0.0, 0.0, 0.1) if k > 0 else (0.0, 0.0, 0.0)
        tr = kf.step(
            [dict(cls=56, name="ghe", mode="ok", meas=[(np.array([0.0, Z_meas]), meas_cov(Z_meas, "ok"))])],
            k * 0.1,
            ego=ego,
        )
    assert len(tr) == 1
    assert abs(tr[0]["vz"]) < 0.25
    assert tr[0]["is_moving"] is False


def test_occupancy_grid_blind_zone_memory_and_steering():
    grid = accel.OccupancyGrid2D()
    # Dat vat can o giua va ben phai (1.8 m) -> huong ne an toan phai la "left"
    obs = [(x, 1.8) for x in np.linspace(-0.2, 1.5, 16)]
    for _ in range(3):
        g, cells = grid.step(0.0, 0.0, 0.0, 0.1, 70.0, 1.15, obstacles=obs)
    assert g["action"] == "left" and g["target_deg"] < -4.0

    # Xoa vat can ben phai, chi chan ben trai & giua -> huong ne la "right"
    grid.reset()
    obs_l = [(x, 1.8) for x in np.linspace(-1.5, 0.2, 16)]
    for _ in range(3):
        g2, _ = grid.step(0.0, 0.0, 0.0, 0.1, 70.0, 1.15, obstacles=obs_l)
    assert g2["action"] == "right" and g2["target_deg"] > 4.0

    # Chan toan bo hanh lang o 0.7 m -> "stop"
    grid.reset()
    obs_all = [(x, 0.7) for x in np.linspace(-1.5, 1.5, 25)]
    for _ in range(3):
        g3, _ = grid.step(0.0, 0.0, 0.0, 0.1, 70.0, 1.15, obstacles=obs_all)
    assert g3["action"] == "stop"

    # Kiem tra nho vung mu: vat di vao vung mu (r < 1.15 m) van duoc nho trong >= 0.8 s
    grid.reset()
    grid.step(0.0, 0.0, 0.0, 0.1, 70.0, 1.15, obstacles=[(0.0, 0.8), (0.1, 0.8)])
    _, cells_mem = grid.step(0.0, 0.0, 0.0, 0.5, 70.0, 1.15, obstacles=[])
    assert any(z < 1.15 for _, z, _, _ in cells_mem)


def test_overhead_hazard_detection():
    cam = Camera(640, 480, 70, 1.5, 20.0)
    gw, gh = 168, 128
    d = object.__new__(D.Depth)
    d.cam, d.fit, d.disp, d.n_anchor = cam, (0.5, -0.25), None, 2
    d._floor_err_ema, d._floor_err_count, d.pitch_hint = None, 0, None
    d._init_grid_state(gw, gh)

    u = (np.arange(gw) + 0.5) * cam.w / gw
    v = (np.arange(gh) + 0.5) * cam.h / gh
    gx = np.tile((u - cam.cx) / cam.f, (gh, 1))
    gy = np.tile(((v - cam.cy) / cam.f)[:, None], (1, gw))
    _, yr = cam.unroll(gx, gy)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    ray = cp - yr * sp
    down = yr * cp + sp
    zc_floor = np.where(down > 1e-3, cam.height / np.maximum(down, 1e-3), np.inf)
    disp = (2.0 / np.maximum(zc_floor, 0.2) + 0.5).astype(np.float32)

    # Thanh chan ngang o do cao hgt ~ 1.70 m, khoang cach Z = 2.0 m, ben duoi trong
    for c in range(60, 108):
        for r in range(10, 26):
            zc_bar = 2.0 / max(float(ray[r, c]), 0.1)
            hgt_bar = cam.height - zc_bar * float(down[r, c])
            if 1.45 <= hgt_bar <= 1.95:
                disp[r, c] = 2.0 / zc_bar + 0.5

    d.disp = disp
    d.scan_all(5.0, min_confirm=2)
    obst, drop = d.scan_all(5.0, min_confirm=2)
    assert len(d.overheads) >= 3
    assert all(1.5 < z < 2.5 for _, z in d.overheads)

    st = SectorTracker(70.0)
    lv = st.update([], dropoffs=drop, overheads=d.overheads)
    assert lv >= 1 and st.overhead[1] is True


def test_depth_onnx_fallback(tmp_path, monkeypatch):
    import torch

    class _DummyDepthNet(torch.nn.Module):
        def forward(self, pixel_values):
            class _Out:
                def __init__(self, d):
                    self.predicted_depth = d
            return _Out(torch.ones((1, pixel_values.shape[2], pixel_values.shape[3]), dtype=pixel_values.dtype))

    monkeypatch.setattr(D.Depth, "_load_pytorch_model", lambda self: _DummyDepthNet())
    monkeypatch.setattr(torch.onnx, "export", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no onnx")))

    cam = Camera(640, 480, 70, 1.5, 25.0)
    d = D.Depth(cam, backend="onnx", onnx_path=str(tmp_path / "missing_export.onnx"))
    assert d.runtime_label in ("PyTorch CUDA FP16", "PyTorch CUDA", "PyTorch CPU")
    out = d._infer_depth(np.zeros((240, 320, 3), dtype=np.uint8))
    assert out.shape == (D.IN_H, D.IN_W)
