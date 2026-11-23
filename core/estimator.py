# -*- coding: utf-8 -*-
"""Uoc luong khoang cach tu box/pose va kiem tra suc khoe camera."""
import math
import time

import cv2
import numpy as np

from core.camera import size_distance
from core.config import BOX_W, D_RED, HEAD, HEIGHTS, KP_CONF, SHOULDER, TORSO


def estimate(cam, box, cls):
    """Uoc luong vi tri (X, Z) cua vat, tra ve (X, Z, mode)."""
    x1, y1, x2, y2 = box
    u = (x1 + x2) / 2.0
    if y2 >= cam.h - 4:
        Z = min(cam.min_visible(), D_RED - 0.05)
        xr, _ = cam.unroll((u - cam.cx) / cam.f, (y2 - cam.cy) / cam.f)
        return xr * Z, Z, "sat"

    p = cam.floor_point(u, y2)
    Zg = p[1] if p else None

    best = None
    if y1 > 2:
        for H, tol in HEIGHTS.get(cls, []):
            zs = size_distance(cam, y1, y2, H, u_top=u, u_bot=u)
            lo = size_distance(cam, y1, y2, H - tol, u_top=u, u_bot=u)
            hi = size_distance(cam, y1, y2, H + tol, u_top=u, u_bot=u)
            if zs is None or lo is None or hi is None:
                continue
            gap = 0.0 if Zg is None or lo <= Zg <= hi else min(abs(Zg - lo), abs(Zg - hi))
            if best is None or gap < best[3]:
                best = (zs, lo, hi, gap)

    if best is None:
        if Zg is None:
            return None
        Z, mode = Zg, "san"
    elif Zg is None:
        Z, mode = best[0], "cao"
    elif best[3] == 0.0:
        Z, mode = (Zg + best[0]) / 2.0, "ok"
    else:
        Z, mode = min(max(Zg, best[1]), best[2]), "hc"

    if p:
        X = p[0] / p[1] * Z
    else:
        xr, _ = cam.unroll((u - cam.cx) / cam.f, (y2 - cam.cy) / cam.f)
        X = xr * Z
    return X, Z, mode


def person_distance(cam, box, kp):
    """Vi tri nguoi (X, Z) tu kich thuoc co the (co bu pitch + roll)."""
    ok = kp[:, 2] > KP_CONF
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    cues = []

    def horiz(a, b, L, s, u_mid, v_mid):
        px = math.hypot(*(a - b))
        if px > 3:
            yr = cam.unrolled_y(u_mid, v_mid)
            cues.append((cam.f * L / px * (cp - yr * sp), s))

    for sh, hp in ((5, 11), (6, 12)):
        if ok[sh] and ok[hp]:
            dy_unrolled = cam.unrolled_y(kp[hp, 0], kp[hp, 1]) - cam.unrolled_y(kp[sh, 0], kp[sh, 1])
            if dy_unrolled * cam.f > 3:
                z = size_distance(
                    cam, kp[sh, 1], kp[hp, 1], TORSO[0], u_top=kp[sh, 0], u_bot=kp[hp, 0]
                )
                if z:
                    cues.append((z, TORSO[1]))

    for (a, b), (L, s) in (((5, 6), SHOULDER), ((3, 4), HEAD)):
        if ok[a] and ok[b]:
            horiz(
                kp[a, :2],
                kp[b, :2],
                L,
                s,
                (kp[a, 0] + kp[b, 0]) / 2.0,
                (kp[a, 1] + kp[b, 1]) / 2.0,
            )

    x1, y1, x2, y2 = box
    if x1 > 4 and x2 < cam.w - 4:
        horiz(
            np.array([x1, 0.0]),
            np.array([x2, 0.0]),
            *BOX_W,
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    cues = [(z, s) for z, s in cues if z > 0.1]
    if not cues:
        return None

    n_torso = sum(1 for _, s in cues if s == TORSO[1])
    med = float(np.median([z for z, _ in (cues[:n_torso] or cues)]))
    good = (
        [(z, s) for z, s in cues if abs(z - med) <= 0.35 * med]
        or [min(cues, key=lambda c: abs(c[0] - med))]
    )
    w = [1.0 / (s * z) ** 2 for z, s in good]
    Z = sum(wi * z for wi, (z, _) in zip(w, good)) / sum(w)

    vis = kp[ok]
    u = float(vis[:, 0].mean()) if len(vis) else (x1 + x2) / 2.0
    v = float(vis[:, 1].mean()) if len(vis) else (y1 + y2) / 2.0
    xr, yr = cam.unroll((u - cam.cx) / cam.f, (v - cam.cy) / cam.f)
    ray = cp - yr * sp
    if ray <= 1e-3:
        return None
    return xr / ray * Z, Z


def check_frame_health(frame, brightness_thresh=14.0, blur_thresh=6.0):
    """Kiem tra do sang va do net khung hinh. Tra ve (healthy, reason, brightness, sharpness)."""
    if frame is None or frame.size == 0:
        return False, "empty", 0.0, 0.0
    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < brightness_thresh:
        return False, "dark", brightness, sharpness
    if sharpness < blur_thresh:
        return False, "blocked", brightness, sharpness
    return True, None, brightness, sharpness


class CameraHealthMonitor:
    """Theo doi suc khoe camera theo thoi gian; bao fail-safe neu che/toi > duration_thresh s."""

    def __init__(self, duration_thresh=1.0, brightness_thresh=14.0, blur_thresh=6.0):
        self.duration_thresh = float(duration_thresh)
        self.brightness_thresh = float(brightness_thresh)
        self.blur_thresh = float(blur_thresh)
        self.unhealthy_since = None
        self.alert = False
        self.reason = None
        self.brightness = 255.0
        self.sharpness = 100.0

    def update(self, frame, now=None):
        if now is None:
            now = time.time()
        ok, reason, bri, shp = check_frame_health(
            frame, self.brightness_thresh, self.blur_thresh
        )
        self.brightness, self.sharpness = bri, shp
        if ok:
            self.unhealthy_since = None
            self.alert = False
            self.reason = None
        else:
            if self.unhealthy_since is None:
                self.unhealthy_since = now
            self.reason = reason
            if now - self.unhealthy_since >= self.duration_thresh:
                self.alert = True
        return {
            "ok": not self.alert,
            "raw_ok": ok,
            "alert": self.alert,
            "reason": self.reason,
            "brightness": round(self.brightness, 1),
            "sharpness": round(self.sharpness, 1),
        }
