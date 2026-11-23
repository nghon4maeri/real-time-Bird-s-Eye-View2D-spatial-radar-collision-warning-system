import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import accel
from core.camera import Camera
from core.tracker import meas_cov

pytestmark = pytest.mark.skipif(not accel.has_rust(), reason="radar_rs khong kha dung")


def test_parity_resolve_occlusion():
    objs = [
        dict(box=(280, 120, 360, 320), cls=0, mode="ok", X=0.2, Z=7.0, r=7.0),
        dict(box=(270, 180, 370, 380), cls=60, mode="ok", X=0.1, Z=4.0, r=4.0),
        dict(box=(260, 340, 380, 440), cls=56, mode="ok", X=0.0, Z=1.5, r=1.5),
    ]
    py_objs = [dict(o) for o in objs]
    rs_objs = [dict(o) for o in objs]
    accel.resolve_occlusion(py_objs, backend="python")
    accel.resolve_occlusion(rs_objs, backend="rust")
    for p, r in zip(py_objs, rs_objs):
        assert p["cls"] == r["cls"] and p["mode"] == r["mode"]
        assert abs(p["X"] - r["X"]) < 1e-4
        assert abs(p["Z"] - r["Z"]) < 1e-4


def test_parity_depth_fit_and_scan():
    cam = Camera(640, 480, 70, 1.5, 22.0, roll_deg=10.0)
    gw, gh = 168, 128
    u = (np.arange(gw) + 0.5) * cam.w / gw
    v = (np.arange(gh) + 0.5) * cam.h / gh
    gx = np.tile((u - cam.cx) / cam.f, (gh, 1))
    gy = np.tile(((v - cam.cy) / cam.f)[:, None], (1, gw))
    xr, yr = cam.unroll(gx, gy)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    ray = cp - yr * sp
    down = yr * cp + sp
    zc = np.minimum(
        np.where(down > 1e-3, cam.height / np.maximum(down, 1e-3), np.inf),
        3.2 / np.maximum(ray, 1e-3),
    )
    disp = (2.0 / np.maximum(zc, 0.2) + 0.5).astype(np.float32)

    anc_boxes = [(300, 150, 340, 320), (440, 260, 500, 400), (120, 180, 160, 310)]
    anc_zs = [3.1, 1.8, 0.4]
    mask_boxes = [(300, 150, 340, 320), (260, 400, 380, 478)]

    res_py = accel.depth_fit_ransac(
        disp, anc_boxes, anc_zs, mask_boxes, cam.w, cam.h, cam.cx, cam.cy, cam.f,
        cam.height, cam.pitch, cam.roll, cam.min_visible(), prev_fit=None, backend="python"
    )
    res_rs = accel.depth_fit_ransac(
        disp, anc_boxes, anc_zs, mask_boxes, cam.w, cam.h, cam.cx, cam.cy, cam.f,
        cam.height, cam.pitch, cam.roll, cam.min_visible(), prev_fit=None, backend="rust"
    )
    assert res_py is not None and res_rs is not None
    assert abs(res_py[0] - res_rs[0]) < 1e-4
    assert abs(res_py[1] - res_rs[1]) < 1e-4
    assert res_py[2] == res_rs[2]

    a, b, _ = res_rs
    os_py = np.zeros(gw, dtype=np.int32)
    ds_py = np.zeros(gw, dtype=np.int32)
    ovs_py = np.zeros(gw, dtype=np.int32)
    oz_py = np.zeros(gw, dtype=np.float64)
    dz_py = np.zeros(gw, dtype=np.float64)
    ovz_py = np.zeros(gw, dtype=np.float64)
    os_rs, ds_rs, ovs_rs = os_py.copy(), ds_py.copy(), ovs_py.copy()
    oz_rs, dz_rs, ovz_rs = oz_py.copy(), dz_py.copy(), ovz_py.copy()

    for _ in range(2):
        op, dp, ovp, os_py, ds_py, ovs_py, oz_py, dz_py, ovz_py, fpy = accel.depth_scan_and_cluster(
            disp, a, b, cam.w, cam.h, cam.cx, cam.cy, cam.f, cam.height, cam.pitch, cam.roll,
            5.0, os_py, ds_py, oz_py, dz_py, prev_over_streak=ovs_py, prev_over_z=ovz_py,
            min_confirm=2, backend="python"
        )
        or_, dr, ovr, os_rs, ds_rs, ovs_rs, oz_rs, dz_rs, ovz_rs, frs = accel.depth_scan_and_cluster(
            disp, a, b, cam.w, cam.h, cam.cx, cam.cy, cam.f, cam.height, cam.pitch, cam.roll,
            5.0, os_rs, ds_rs, oz_rs, dz_rs, prev_over_streak=ovs_rs, prev_over_z=ovz_rs,
            min_confirm=2, backend="rust"
        )
    assert len(op) == len(or_) and len(dp) == len(dr) and len(ovp) == len(ovr)
    for (x1, z1), (x2, z2) in zip(op, or_):
        assert abs(x1 - x2) < 1e-4 and abs(z1 - z2) < 1e-4
    if not math.isnan(fpy):
        assert abs(fpy - frs) < 1e-4


def test_parity_kalman_with_ego_and_occupancy_grid():
    rng = np.random.default_rng(7)
    kf_py = accel.KalmanTracker(backend="python")
    kf_rs = accel.KalmanTracker(backend="rust")
    grid_py = accel.OccupancyGrid2D(backend="python")
    grid_rs = accel.OccupancyGrid2D(backend="rust")

    for k in range(20):
        t = k * 0.1
        ego = (0.02, 0.0, 0.05) if k > 0 else (0.0, 0.0, 0.0)
        objs = [
            dict(
                cls=0,
                name="nguoi",
                mode="pose",
                meas=[(np.array([0.2 + rng.normal(0, 0.03), 4.0 - 0.8 * t]), meas_cov(3.0, "pose"))],
            ),
            dict(
                cls=56,
                name="ghe",
                mode="ok",
                meas=[(np.array([-0.8, 2.5 - 0.05 * k]), meas_cov(2.5, "ok"))],
            ),
        ]
        out_py = sorted(kf_py.step(objs, t, ego=ego), key=lambda x: x["id"])
        out_rs = sorted(kf_rs.step(objs, t, ego=ego), key=lambda x: x["id"])
        assert len(out_py) == len(out_rs)
        for p, r in zip(out_py, out_rs):
            assert p["id"] == r["id"] and p["cls"] == r["cls"] and p["is_moving"] == r["is_moving"]
            for key in ("X", "Z", "vx", "vz", "closing_speed", "cpa"):
                assert abs(p[key] - r[key]) < 1e-4

        g_py, c_py = grid_py.step(
            ego[0], ego[1], ego[2], 0.1, 70.0, 1.15,
            obstacles=[(0.0, 2.2), (0.1, 2.2), (-0.1, 2.2)],
            dropoffs=[], overheads=[], tracks=out_py,
        )
        g_rs, c_rs = grid_rs.step(
            ego[0], ego[1], ego[2], 0.1, 70.0, 1.15,
            obstacles=[(0.0, 2.2), (0.1, 2.2), (-0.1, 2.2)],
            dropoffs=[], overheads=[], tracks=out_rs,
        )
        assert g_py["action"] == g_rs["action"]
        assert abs(g_py["target_deg"] - g_rs["target_deg"]) < 1e-4
        assert abs(g_py["clearance_m"] - g_rs["clearance_m"]) < 1e-4
        assert len(c_py) == len(c_rs)
