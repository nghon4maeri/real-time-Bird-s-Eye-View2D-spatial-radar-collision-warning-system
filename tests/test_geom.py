import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import radar as R


def test_floor_point_round_trip():
    cam = R.Camera(640, 480, 70, 1.5, 25)
    for X, Z in [(0, 2), (1, 3), (-1.5, 4), (0.3, 1.2)]:
        u, v = cam.project(X, 1.5, Z)
        gx, gz = cam.floor_point(u, v)
        assert abs(gx - X) < 1e-6 and abs(gz - Z) < 1e-6
    assert 1.0 < cam.min_visible() < 1.3


def test_floor_point_with_roll_and_pitch():
    for roll in (-18.0, -7.5, 12.0, 22.0):
        cam = R.Camera(640, 480, 70, 1.5, 22.0, roll_deg=roll)
        for X, Z in [(0.0, 2.0), (0.8, 3.0), (-1.2, 4.0)]:
            u, v = cam.project(X, 1.5, Z)
            gx, gz = cam.floor_point(u, v)
            assert abs(gx - X) < 1e-5
            assert abs(gz - Z) < 1e-5
        p0, p1 = cam.horizon_line()
        assert p0[1] != p1[1]


def test_estimate_synthetic_boxes():
    cam = R.Camera(640, 480, 70, 1.5, 25)
    for cls, H in [(0, 1.65), (0, 1.25), (56, 0.85), (60, 0.75)]:
        for X, Z in [(0, 1.3), (0, 2), (0.8, 3), (-1, 4), (0, 6)]:
            u, vb = cam.project(X, 1.5, Z)
            _, vt = cam.project(X, 1.5 - H, Z)
            w = 0.25 * cam.f / Z
            box = [u - w, vt, u + w, vb]
            if vb >= cam.h:
                box = [u - w, vt, u + w, cam.h - 1]
            if vt < 0:
                box = [box[0], 0, box[2], box[3]]
            ex, ez, mode = R.estimate(cam, box, cls)
            if Z >= 1.5:
                assert abs(ez - Z) < 0.25, (cls, Z, ez, mode)


def test_calibrate_pitch():
    cam_true = R.Camera(640, 480, 70, 1.5, 25)
    u, v = cam_true.project(0, 1.5, 2.0)
    c2 = R.Camera(640, 480, 70, 1.5, 10)
    c2.calibrate_pitch(u, v, 2.0)
    assert abs(math.degrees(c2.pitch) - 25.0) < 0.05


def test_sector_static_and_ttc_cpa():
    t = R.SectorTracker(70)
    objs = [
        dict(X=-1, Z=1.2, r=math.hypot(-1, 1.2)),
        dict(X=0, Z=2.0, r=2.0),
        dict(X=2, Z=4.0, r=math.hypot(2, 4)),
    ]
    assert t.update(objs) == 1
    assert t.level == [1, 1, 0]

    # Vat o xa (3.8 m -> tinh la AN TOAN) nhung lao thang vao hanh lang |CPA| < 0.6 m voi TTC < 1.5 s -> NGUY HIEM
    t2 = R.SectorTracker(70)
    cs, xcpa, cpa, ttc = R.compute_ttc_cpa(0.2, 3.8, 0.0, -3.0)
    assert cs > 2.5 and cpa < 0.6 and ttc < 1.5
    lv = t2.update([dict(X=0.2, Z=3.8, r=math.hypot(0.2, 3.8), closing_speed=cs, cpa=cpa, ttc=ttc)])
    assert lv == 2
    assert t2.level[1] == 2

    # Nguoi di ngang qua mat o Z=3.0m -> CPA = 3.0m >= 0.6m -> khong kich hoat TTC alarm
    t3 = R.SectorTracker(70)
    cs2, _, cpa2, ttc2 = R.compute_ttc_cpa(-1.0, 3.0, 2.0, 0.0)
    assert cpa2 > 0.6
    lv2 = t3.update([dict(X=-1.0, Z=3.0, r=math.hypot(-1.0, 3.0), closing_speed=cs2, cpa=cpa2, ttc=ttc2)])
    assert lv2 == 0


def test_resolve_occlusion_multilevel_chain():
    # Chuoi 3 vat che nhau (dat nguoc thu tu de kiem tra sap xep Z tang dan)
    obj_far = dict(box=(280, 120, 360, 320), cls=0, mode="ok", X=0.0, Z=8.0, r=8.0)
    obj_mid = dict(box=(270, 180, 370, 380), cls=60, mode="ok", X=0.0, Z=5.0, r=5.0)
    obj_near = dict(box=(260, 340, 380, 440), cls=56, mode="ok", X=0.0, Z=1.5, r=1.5)

    res = R.resolve_occlusion([obj_far, obj_mid, obj_near])
    by_cls = {o["cls"]: o for o in res}
    assert abs(by_cls[56]["Z"] - 1.5) < 1e-6
    assert abs(by_cls[60]["Z"] - 2.0) < 1e-6 and by_cls[60]["mode"] == "che"
    assert abs(by_cls[0]["Z"] - 2.5) < 1e-6 and by_cls[0]["mode"] == "che"
