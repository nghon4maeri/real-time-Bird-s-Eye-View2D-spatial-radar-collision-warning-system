import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import depth as D
import radar as R
from core import accel


def _make_synth_depth(cam, grid=(84, 64), wall_z=3.0, drop_x_range=None):
    gw, gh = grid
    d = object.__new__(D.Depth)
    d.cam, d.fit, d.disp, d.n_anchor = cam, None, None, 0
    d._init_grid_state(gw, gh)

    u = (np.arange(gw) + 0.5) * cam.w / gw
    v = (np.arange(gh) + 0.5) * cam.h / gh
    gx = np.tile((u - cam.cx) / cam.f, (gh, 1))
    gy = np.tile(((v - cam.cy) / cam.f)[:, None], (1, gw))
    xr, yr = cam.unroll(gx, gy)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    ray = cp - yr * sp
    down = yr * cp + sp
    zc_floor = np.where(down > 1e-3, cam.height / np.maximum(down, 1e-3), np.inf)
    zc_wall = wall_z / np.maximum(ray, 1e-3) if wall_z is not None else np.full_like(zc_floor, np.inf)
    zc = np.minimum(zc_floor, zc_wall)

    if drop_x_range is not None:
        X_floor = zc_floor * xr
        Z_floor = zc_floor * ray
        mask_drop = (Z_floor > 1.2) & (Z_floor < 2.6) & (X_floor > drop_x_range[0]) & (X_floor < drop_x_range[1])
        # San tut xuong 0.7 m -> down tang them 0.7 m -> zc tang
        zc_drop = np.where(down > 1e-3, (cam.height + 0.7) / np.maximum(down, 1e-3), np.inf)
        zc = np.where(mask_drop, zc_drop, zc)

    disp = (2.0 / np.maximum(zc, 0.2) + 0.5).astype(np.float32)
    return d, disp, zc, ray


def test_kalman_smoothing_hungarian_ttl_and_joseph():
    rng = np.random.default_rng(0)
    kf = R.KalmanTracker()
    raw_err, kf_err = [], []
    for k in range(40):
        t = k * 0.1
        X, Z = 0.5, 4.0 - 1.0 * t
        zx, zz = X + rng.normal(0, 0.1), Z + rng.normal(0, 0.15 * Z)
        o = dict(cls=0, box=None, name="nguoi", mode="pose", meas=[(np.array([zx, zz]), R.meas_cov(zz, "pose"))])
        tr = kf.step([o], t)
        assert len(tr) == 1
        if k > 10:
            raw_err.append(abs(zz - Z))
            kf_err.append(abs(tr[0]["Z"] - Z))
    assert np.mean(kf_err) < np.mean(raw_err)
    assert abs(tr[0]["vz"] + 1.0) < 0.35
    assert tr[0]["ttc"] is not None and tr[0]["ttc"] < 1.5

    # Khong ghep nham giua cac lop khac nhau (chair 56 vs table 60), tach TTL nguoi vs vat tinh
    kf2 = R.KalmanTracker()
    for k in range(10):
        objs = [
            dict(cls=0, name="nguoi", mode="ok", meas=[(np.array([-0.5, 2.0]), R.meas_cov(2.0, "ok"))]),
            dict(cls=56, name="ghe", mode="ok", meas=[(np.array([-0.4, 2.1]), R.meas_cov(2.1, "ok"))]),
            dict(cls=60, name="ban", mode="ok", meas=[(np.array([-0.42, 2.12]), R.meas_cov(2.12, "ok"))]),
        ]
        tr2 = kf2.step(objs, k * 0.1, det_every=3, fps=10.0)
    assert len(tr2) == 3 and {o["cls"] for o in tr2} == {0, 56, 60}

    # Sau 0.8s mat dau: nguoi (TTL 0.7s) bien mat, ghe/ban (TTL thich ung >= 1.5s) van giu
    tr_mid = kf2.step([], 0.9 + 0.8, det_every=3, fps=10.0)
    assert {o["cls"] for o in tr_mid} == {56, 60}
    # Qua TTL vat tinh thi xoa het
    assert len(kf2.step([], 0.9 + 3.0, det_every=3, fps=10.0)) == 0

    z, _ = R.fuse([(np.array([0, 1.45]), R.meas_cov(1.45, "sat")), (np.array([0, 2.3]), R.meas_cov(2.3, "depth"))])
    assert 2.0 < z[1] < 2.3


def test_depth_fit_ransac_mask_and_wall_scan():
    cam = R.Camera(640, 480, 70, 1.5, 20)
    d, disp, zc, ray = _make_synth_depth(cam, grid=(84, 64), wall_z=3.0)

    box = (300, 150, 340, 330)
    r, c = d._cell(box)
    Zp = float(np.median(zc[r, c] * ray[r, c]))

    box2 = (440, 280, 500, 410)
    r2, c2 = d._cell(box2)
    Zp2 = float(np.median(zc[r2, c2] * ray[r2, c2]))

    # Them 1 neo outlier sai han + 1 box dung sat mep duoi de mask khoi san
    bad_box = (80, 140, 130, 300)
    near_box = (240, 380, 400, 478)
    for _ in range(15):
        d._fit(disp, [(box, Zp, "pose"), (box2, Zp2, "ok"), (bad_box, 0.35, "ok"), (near_box, 0.8, "sat")])
    d.disp = disp
    assert d.fit is not None
    assert abs(d.fit[0] - 0.5) < 0.08
    assert abs(d.fit[1] + 0.25) < 0.08

    _, Z = d.measure(box)
    assert abs(Z - Zp) < 0.15

    # Chu ky 1: chua xac nhan 2 khung -> chua tra ve
    pts1 = d.scan(5.0, min_confirm=2)
    assert pts1 == []
    # Chu ky 2: xac nhan tuong o Z = 3.0 m
    pts2 = d.scan(5.0, min_confirm=2)
    wall = [z for x, z in pts2 if abs(x) < 1.0]
    assert wall and abs(np.median(wall) - 3.0) < 0.15
    assert all(z > 0.2 for _, z in pts2)


def test_depth_cluster_noise_filter_and_roll_compensation():
    # Roll = 18 do: san van phai bi loai (khong nham san nghieng thanh vat can)
    cam = R.Camera(640, 480, 70, 1.5, 22.0, roll_deg=18.0)
    d, disp, _, _ = _make_synth_depth(cam, grid=(84, 64), wall_z=None)
    d.fit = (0.5, -0.25)
    d.disp = disp
    obst1, drop1 = d.scan_all(5.0, min_confirm=1)
    assert obst1 == [] and drop1 == []

    # Them nhieu don le (chi 1 cot) -> phai bi bo boi min_cluster_cols=3
    disp_noisy = disp.copy()
    disp_noisy[20:28, 42] = 2.0 / 1.8 + 0.5
    d.disp = disp_noisy
    obst_n1, _ = d.scan_all(5.0, min_confirm=1, min_cluster_cols=3)
    assert obst_n1 == []


def test_depth_dropoff_detection():
    cam = R.Camera(640, 480, 70, 1.5, 25.0)
    d, disp, _, _ = _make_synth_depth(cam, grid=(168, 128), wall_z=None, drop_x_range=(-0.6, 0.6))
    d.fit = (0.5, -0.25)
    d.disp = disp
    # 2 chu ky lien tiep de xac nhan bac thang / ho sau
    d.scan_all(5.0, min_confirm=2)
    obst, drop = d.scan_all(5.0, min_confirm=2)
    assert obst == []
    assert len(drop) >= 3
    assert all(1.0 < z < 2.8 for _, z in drop)
