# -*- coding: utf-8 -*-
"""Gop phep do, phan tich TTC/CPA va theo doi canh bao theo 3 cung."""
import math
import numpy as np

from core.config import (
    CPA_CORRIDOR,
    D_RED,
    D_YELLOW,
    HYST,
    MEAS_REL,
    MIN_CLOSING_SPEED,
    SAT_SIGMA,
    TTC_RED,
    TTC_YELLOW,
)


def meas_cov(Z, mode):
    """Ma tran hiep phuong sai do (X, Z)."""
    sz = SAT_SIGMA if mode == "sat" else MEAS_REL.get(mode, 0.2) * Z + 0.05
    sx = 0.1 + 0.03 * Z
    return np.diag([sx * sx, sz * sz])


def fuse(meas):
    """Gop nhieu phep do [(z, R)] cua cung 1 vat theo dang thong tin."""
    info = sum(np.linalg.inv(R) for _, R in meas)
    R = np.linalg.inv(info)
    return R @ sum(np.linalg.inv(Ri) @ z for z, Ri in meas), R


def compute_ttc_cpa(X, Z, vx, vz):
    """Tinh closing_speed, x_cpa, cpa_dist va TTC cho vat co trang thai (X, Z, vx, vz)."""
    r = math.hypot(X, Z)
    if r < 1e-6:
        return 0.0, X, 0.0, None
    closing_speed = -(X * vx + Z * vz) / r
    v2 = vx * vx + vz * vz
    if closing_speed > 0.0 and v2 > 1e-6:
        t_cpa = -(X * vx + Z * vz) / v2
        x_cpa = X + vx * t_cpa
        z_cpa = Z + vz * t_cpa
        cpa_dist = math.hypot(x_cpa, z_cpa)
        ttc = t_cpa if (t_cpa > 0.0 and closing_speed > MIN_CLOSING_SPEED) else None
    else:
        x_cpa = X
        cpa_dist = r
        ttc = None
    return closing_speed, x_cpa, cpa_dist, ttc


def level_of(d, prev=0):
    """Phan vung tinh theo khoang cach co do tre HYST."""
    if d < D_RED or (prev == 2 and d < D_RED + HYST):
        return 2
    if d < D_YELLOW or (prev >= 1 and d < D_YELLOW + HYST):
        return 1
    return 0


def dynamic_level_of(obj):
    """Muc canh bao dong dua tren TTC va hanh lang CPA."""
    ttc = obj.get("ttc")
    if ttc is None:
        return 0
    if (
        obj.get("closing_speed", 0.0) > MIN_CLOSING_SPEED
        and obj.get("cpa", float("inf")) < CPA_CORRIDOR
    ):
        if ttc < TTC_RED:
            return 2
        if ttc < TTC_YELLOW:
            return 1
    return 0


class SectorTracker:
    """Giu khoang cach gan nhat + canh bao dong TTC/CPA + drop-off + overhead moi cung TRAI/GIUA/PHAI."""

    def __init__(self, hfov_deg):
        self.edge = float(hfov_deg) / 6.0
        self.d = [math.inf] * 3
        self.miss = [0] * 3
        self.level = [0] * 3
        self.dyn_level = [0] * 3
        self.min_ttc = [None] * 3
        self.dropoff = [False] * 3
        self.dropoff_d = [math.inf] * 3
        self.overhead = [False] * 3
        self.overhead_d = [math.inf] * 3

    def update_hfov(self, hfov_deg):
        self.edge = float(hfov_deg) / 6.0

    def sector_of(self, X, Z):
        a = math.degrees(math.atan2(X, max(Z, 1e-3)))
        return 0 if a < -self.edge else (2 if a > self.edge else 1)

    def update(self, objs, dropoffs=(), overheads=()):
        nearest = [math.inf] * 3
        dyn_lv = [0] * 3
        min_ttc = [None] * 3
        drop_near = [math.inf] * 3
        over_near = [math.inf] * 3

        for o in objs:
            s = self.sector_of(o["X"], o["Z"])
            nearest[s] = min(nearest[s], o["r"])
            dlv = dynamic_level_of(o)
            if dlv > dyn_lv[s]:
                dyn_lv[s] = dlv
            ttc = o.get("ttc")
            if ttc is not None and dlv > 0:
                if min_ttc[s] is None or ttc < min_ttc[s]:
                    min_ttc[s] = ttc

        for pt in dropoffs:
            if isinstance(pt, dict):
                dx, dz = pt["X"], pt["Z"]
                dr = pt.get("r", math.hypot(dx, dz))
            else:
                dx, dz = pt[0], pt[1]
                dr = math.hypot(dx, dz)
            s = self.sector_of(dx, dz)
            drop_near[s] = min(drop_near[s], dr)

        for pt in overheads:
            if isinstance(pt, dict):
                ox, oz = pt["X"], pt["Z"]
                orad = pt.get("r", math.hypot(ox, oz))
            else:
                ox, oz = pt[0], pt[1]
                orad = math.hypot(ox, oz)
            s = self.sector_of(ox, oz)
            over_near[s] = min(over_near[s], orad)

        for i in range(3):
            n = nearest[i]
            if n < math.inf:
                self.miss[i] = 0
                if self.d[i] == math.inf or n < self.d[i]:
                    self.d[i] = n
                else:
                    self.d[i] += 0.3 * (n - self.d[i])
            else:
                self.miss[i] += 1
                if self.miss[i] > 8:
                    self.d[i] = math.inf

            self.dyn_level[i] = dyn_lv[i]
            self.min_ttc[i] = min_ttc[i]
            self.dropoff_d[i] = drop_near[i]
            self.dropoff[i] = drop_near[i] < D_YELLOW + 0.5
            self.overhead_d[i] = over_near[i]
            self.overhead[i] = over_near[i] < D_YELLOW + 0.5

            static_lv = level_of(self.d[i], self.level[i])
            drop_lv = 2 if drop_near[i] < D_RED else (1 if drop_near[i] < D_YELLOW + 0.5 else 0)
            over_lv = 2 if over_near[i] < D_RED else (1 if over_near[i] < D_YELLOW + 0.5 else 0)
            self.level[i] = max(static_lv, dyn_lv[i], drop_lv, over_lv)

        return max(self.level)

    def worst_sector(self):
        """Tra ve chi so cung (0|1|2) nguy hiem nhat."""
        best_i = 1
        best_key = (-1, math.inf)
        for i in range(3):
            eff_d = min(self.d[i], self.dropoff_d[i], self.overhead_d[i])
            if self.level[i] > best_key[0] or (self.level[i] == best_key[0] and eff_d < best_key[1]):
                best_i = i
                best_key = (self.level[i], eff_d)
        return best_i
