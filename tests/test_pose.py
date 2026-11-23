import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import radar as R


def person_kp(cam, X, Z, yaw=0.0):
    """Khung nguoi 3D (vai 1.40, hong 0.90, tai 1.60) dung tai (X, Z), xoay yaw do."""
    import math

    def proj(dx, H):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        x, z = X + dx * c, Z + dx * s
        return cam.project(x, cam.height - H, z)

    kp = np.zeros((17, 3))
    for i, dx, H in [
        (3, -0.075, 1.6),
        (4, 0.075, 1.6),
        (5, -0.19, 1.4),
        (6, 0.19, 1.4),
        (11, -0.19, 0.9),
        (12, 0.19, 0.9),
    ]:
        kp[i] = (*proj(dx, H), 0.9)
    bl, br = proj(-0.225, 1.4)[0], proj(0.225, 1.4)[0]
    return kp, (min(bl, br), proj(0, 1.7)[1], max(bl, br), cam.h - 1)


def test_person_distance_pitch_yaw():
    for pitch in (0, -12, 25):
        cam = R.Camera(640, 480, 70, 1.5, pitch)
        for X, Z, yaw in [(0, 1.5, 0), (0.5, 2.0, 0), (-1, 3.0, 0), (0, 2.0, 60), (0.3, 5.0, 30)]:
            kp, box = person_kp(cam, X, Z, yaw)
            ex, ez = R.person_distance(cam, box, kp)
            assert abs(ez - Z) < 0.10 * Z and abs(ex - X) < 0.15


def test_person_distance_with_roll():
    for roll in (-15.0, 15.0):
        cam = R.Camera(640, 480, 70, 1.5, 20.0, roll_deg=roll)
        for X, Z in [(0.0, 2.0), (0.5, 2.5), (-0.6, 3.0)]:
            kp, box = person_kp(cam, X, Z, yaw=0.0)
            ex, ez = R.person_distance(cam, box, kp)
            assert abs(ez - Z) < 0.12 * Z and abs(ex - X) < 0.20


def test_person_distance_edge_cases():
    cam = R.Camera(640, 480, 70, 1.5, -12)
    kp, box = person_kp(cam, 0, 2.0)
    kp[11, 1] = kp[5, 1] + 5
    kp[12, 1] = kp[6, 1] + 400
    assert R.person_distance(cam, box, kp) is not None

    kp2 = np.zeros((17, 3))
    kp2[5] = (300, 10, 0.9)
    kp2[6] = (340, 10, 0.9)
    cam2 = R.Camera(640, 480, 70, 1.5, 60)
    R.person_distance(cam2, (290, 0, 350, 479), kp2)
