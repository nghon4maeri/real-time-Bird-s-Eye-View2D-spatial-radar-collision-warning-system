# -*- coding: utf-8 -*-
"""Cau noi tang toc Rust (radar_rs) voi auto-fallback sang Python/NumPy thuan."""
import math
import numpy as np

from core.config import (
    ACCEL,
    GATE,
    MIN_CLOSING_SPEED,
    OCCLUDED_GAP,
    TRACK_TTL_PERSON,
    TRACK_TTL_STATIC,
)
from core.tracker import compute_ttc_cpa, fuse

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:
    _scipy_lsa = None

try:
    import radar_rs as _rust
except ImportError:
    _rust = None

COMPUTE_BACKEND = "rust" if _rust is not None else "python"
_ACTIVE_BACKEND = COMPUTE_BACKEND


def has_rust():
    return _rust is not None


def get_backend():
    return _ACTIVE_BACKEND


def set_backend(name):
    global _ACTIVE_BACKEND
    if name == "rust":
        if _rust is None:
            raise RuntimeError("radar_rs chua duoc cai dat.")
        _ACTIVE_BACKEND = "rust"
    elif name == "python":
        _ACTIVE_BACKEND = "python"
    else:
        raise ValueError(f"Backend khong hop le: {name}")


def _use_rust(backend=None):
    b = _ACTIVE_BACKEND if backend is None else backend
    return b == "rust" and _rust is not None


# ======================= 1. OCCLUSION =======================
def resolve_occlusion_py(objs, occluded_gap=OCCLUDED_GAP):
    """Sap xep objs theo Z tang dan va xu ly chuoi che khuat nhieu tang."""
    objs.sort(key=lambda o: o["Z"])
    for a in objs:
        if a["mode"] in ("sat", "pose"):
            continue
        ax1, _, ax2, ay2 = a["box"]
        au = 0.5 * (ax1 + ax2)
        for b in objs:
            if b is a or b["Z"] >= a["Z"]:
                continue
            bx1, by1, bx2, by2 = b["box"]
            if bx1 <= au <= bx2 and by1 <= ay2 < by2 - 8.0:
                z_new = b["Z"] + occluded_gap
                if z_new < a["Z"]:
                    if abs(a["Z"]) > 1e-9:
                        a["X"] *= z_new / a["Z"]
                    a["Z"] = z_new
                    a["mode"] = "che"
                    a["r"] = math.hypot(a["X"], a["Z"])
    return objs


def resolve_occlusion(objs, occluded_gap=OCCLUDED_GAP, backend=None):
    if not objs:
        return objs
    if _use_rust(backend):
        objs.sort(key=lambda o: o["Z"])
        boxes = np.asarray([o["box"] for o in objs], dtype=np.float64)
        xs = np.asarray([o["X"] for o in objs], dtype=np.float64)
        zs = np.asarray([o["Z"] for o in objs], dtype=np.float64)
        can_occ = np.asarray([o["mode"] not in ("sat", "pose") for o in objs], dtype=bool)
        new_x, new_z, che = _rust.resolve_occlusion_raw(boxes, xs, zs, can_occ, float(occluded_gap))
        for i, o in enumerate(objs):
            if che[i]:
                o["X"] = new_x[i]
                o["Z"] = new_z[i]
                o["mode"] = "che"
                o["r"] = math.hypot(new_x[i], new_z[i])
        return objs
    return resolve_occlusion_py(objs, occluded_gap=occluded_gap)


# ======================= 2. DEPTH FIT (MASK + RANSAC/HUBER) =======================
def _solve_wls_2x2(d, iz, w):
    s00 = float(np.sum(w * d * d))
    s01 = float(np.sum(w * d))
    s11 = float(np.sum(w))
    t0 = float(np.sum(w * d * iz))
    t1 = float(np.sum(w * iz))
    det = s00 * s11 - s01 * s01
    if abs(det) <= 1e-12:
        return None
    a = (t0 * s11 - t1 * s01) / det
    b = (s00 * t1 - s01 * t0) / det
    if math.isfinite(a) and math.isfinite(b):
        return a, b
    return None


def depth_fit_ransac_py(
    disp,
    anchor_boxes,
    anchor_zs,
    mask_boxes,
    cam_w,
    cam_h,
    cx,
    cy,
    f,
    cam_height,
    pitch,
    roll,
    min_visible,
    prev_fit=None,
):
    """Ban Python/NumPy cua depth_fit_ransac: mask vat can khoi san + RANSAC/Huber."""
    gh, gw = disp.shape
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    sx, sy = gw / cam_w, gh / cam_h

    d_pts, iz_pts, w_pts = [], [], []
    for i in range(len(anchor_zs)):
        x1, y1, x2, y2 = [float(v) for v in anchor_boxes[i]]
        z = float(anchor_zs[i])
        if z <= 0.05 or not math.isfinite(z):
            continue
        bw, bh = x2 - x1, y2 - y1
        r0 = max(0, min(gh - 1, int((y1 + 0.3 * bh) * sy)))
        r1 = min(gh, max(r0 + 1, int((y1 + 0.7 * bh) * sy)))
        c0 = max(0, min(gw - 1, int((x1 + 0.3 * bw) * sx)))
        c1 = min(gw, max(c0 + 1, int((x1 + 0.7 * bw) * sx)))

        u_mean = 0.5 * (c0 + c1) * cam_w / gw
        v_mean = 0.5 * (r0 + r1) * cam_h / gh
        yr = ((u_mean - cx) / f) * sr + ((v_mean - cy) / f) * cr
        ray = cp - yr * sp
        if ray > 0.05:
            patch = disp[r0:r1, c0:c1]
            valid = patch[np.isfinite(patch)]
            if valid.size > 0:
                d_pts.append(float(np.median(valid)))
                iz_pts.append(ray / z)
                w_pts.append(1.0)

    if min_visible < 4.0 and gh >= 4:
        c_start, c_end = gw // 4, (3 * gw) // 4
        m_arr = np.asarray(mask_boxes, dtype=np.float64) if len(mask_boxes) else np.empty((0, 4))
        for row in (gh - 2, gh - 3, gh - 4):
            v_r = (row + 0.5) * cam_h / gh
            d_floor, iz_floor = [], []
            for c in range(c_start, c_end):
                u_c = (c + 0.5) * cam_w / gw
                if m_arr.shape[0] > 0:
                    if np.any(
                        (u_c >= m_arr[:, 0] - 2.0)
                        & (u_c <= m_arr[:, 2] + 2.0)
                        & (v_r >= m_arr[:, 1] - 2.0)
                        & (v_r <= m_arr[:, 3] + 2.0)
                    ):
                        continue
                yr = ((u_c - cx) / f) * sr + ((v_r - cy) / f) * cr
                down = yr * cp + sp
                ray = cp - yr * sp
                if down > 1e-3 and ray > 0.05 * down:
                    val = float(disp[row, c])
                    if math.isfinite(val):
                        d_floor.append(val)
                        iz_floor.append(down / cam_height)
            if len(d_floor) >= 3:
                d_pts.append(float(np.median(d_floor)))
                iz_pts.append(float(np.median(iz_floor)))
                w_pts.append(0.5)
                break

    n = len(d_pts)
    if n == 0:
        return None

    d = np.asarray(d_pts, dtype=np.float64)
    iz = np.asarray(iz_pts, dtype=np.float64)
    w = np.asarray(w_pts, dtype=np.float64)
    ptp_ok = float(np.ptp(d)) > 1e-3 * max(float(np.abs(d).max()), 1e-6)

    if n >= 3 and ptp_ok:
        thresh = max(0.04, 0.18 * float(np.median(np.abs(iz))))
        best_score = -math.inf
        best_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                dd = d[j] - d[i]
                if abs(dd) <= 1e-6:
                    continue
                ac = (iz[j] - iz[i]) / dd
                if ac <= 0.0 or not math.isfinite(ac):
                    continue
                bc = iz[i] - ac * d[i]
                err = np.abs(ac * d + bc - iz)
                inlier_w = float(np.sum(w[err <= thresh]))
                loss = float(np.sum(w * np.minimum(err, thresh) ** 2))
                score = inlier_w - 0.01 * loss
                if score > best_score:
                    best_score = score
                    best_pair = (float(ac), float(bc))

        if best_pair is not None:
            ra, rb = best_pair
            mask = np.abs(ra * d + rb - iz) <= thresh
            if np.sum(mask) >= 2:
                fit0 = _solve_wls_2x2(d[mask], iz[mask], w[mask])
                cur_a, cur_b = fit0 if fit0 is not None else (ra, rb)
            else:
                cur_a, cur_b = ra, rb

            for _ in range(5):
                res = np.abs(cur_a * d + cur_b - iz)
                hw = np.where(
                    res <= thresh,
                    w,
                    np.where(res <= 3.0 * thresh, w * (thresh / np.maximum(res, 1e-9)), 0.0),
                )
                upd = _solve_wls_2x2(d, iz, hw)
                if upd is not None and upd[0] > 0.0:
                    cur_a, cur_b = upd
            a, b = cur_a, cur_b
        else:
            sol = _solve_wls_2x2(d, iz, w)
            if sol is None:
                return None
            a, b = sol
    elif n >= 2 and ptp_ok:
        sol = _solve_wls_2x2(d, iz, w)
        if sol is None:
            return None
        a, b = sol
    else:
        b = float(prev_fit[1]) if prev_fit is not None else 0.0
        den = max(float(np.sum(w * d * d)), 1e-9)
        a = float(np.sum(w * (iz - b) * d)) / den

    if a <= 0.0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    if prev_fit is not None:
        a = prev_fit[0] + 0.3 * (a - prev_fit[0])
        b = prev_fit[1] + 0.3 * (b - prev_fit[1])
    return float(a), float(b), int(n)


def depth_fit_ransac(
    disp,
    anchor_boxes,
    anchor_zs,
    mask_boxes,
    cam_w,
    cam_h,
    cx,
    cy,
    f,
    cam_height,
    pitch,
    roll,
    min_visible,
    prev_fit=None,
    backend=None,
):
    disp_arr = np.ascontiguousarray(disp, dtype=np.float32)
    ab_arr = (
        np.ascontiguousarray(anchor_boxes, dtype=np.float32)
        if len(anchor_boxes)
        else np.empty((0, 4), dtype=np.float32)
    )
    az_arr = (
        np.ascontiguousarray(anchor_zs, dtype=np.float32)
        if len(anchor_zs)
        else np.empty((0,), dtype=np.float32)
    )
    mb_arr = (
        np.ascontiguousarray(mask_boxes, dtype=np.float32)
        if len(mask_boxes)
        else np.empty((0, 4), dtype=np.float32)
    )
    pf = (float(prev_fit[0]), float(prev_fit[1])) if prev_fit is not None else None
    if _use_rust(backend):
        return _rust.depth_fit_ransac(
            disp_arr,
            ab_arr,
            az_arr,
            mb_arr,
            float(cam_w),
            float(cam_h),
            float(cx),
            float(cy),
            float(f),
            float(cam_height),
            float(pitch),
            float(roll),
            float(min_visible),
            pf,
        )
    return depth_fit_ransac_py(
        disp_arr,
        ab_arr,
        az_arr,
        mb_arr,
        float(cam_w),
        float(cam_h),
        float(cx),
        float(cy),
        float(f),
        float(cam_height),
        float(pitch),
        float(roll),
        float(min_visible),
        pf,
    )


# ======================= 3. DEPTH SCAN, CLUSTERING, DROP-OFF & OVERHEAD =======================
def _cluster_and_confirm_py(valid, xs, zs, prev_streak, prev_z, min_cluster_cols, min_confirm):
    gw = len(valid)
    clustered = np.zeros(gw, dtype=bool)
    c = 0
    while c < gw:
        if not valid[c]:
            c += 1
            continue
        start = c
        c += 1
        while c < gw and valid[c]:
            dz_tol = max(0.30, 0.18 * min(float(zs[c]), float(zs[c - 1])))
            if abs(float(zs[c]) - float(zs[c - 1])) <= dz_tol:
                c += 1
            else:
                break
        if c - start >= min_cluster_cols:
            clustered[start:c] = True

    streak = np.zeros(gw, dtype=np.int32)
    out_z = np.zeros(gw, dtype=np.float64)
    pts = []
    for col in range(gw):
        if not clustered[col]:
            continue
        zc = float(zs[col])
        out_z[col] = zc
        z_tol = max(0.35, 0.25 * zc)
        c_lo = max(0, col - 1)
        c_hi = min(gw - 1, col + 1)
        best_prev = 0
        for nb in range(c_lo, c_hi + 1):
            if prev_streak[nb] > 0 and prev_z[nb] > 0.0 and abs(zc - float(prev_z[nb])) <= z_tol:
                if int(prev_streak[nb]) > best_prev:
                    best_prev = int(prev_streak[nb])
        st = min(best_prev + 1, 1000)
        streak[col] = st
        if st >= min_confirm:
            pts.append((float(xs[col]), zc))
    return pts, streak, out_z


def depth_scan_and_cluster_py(
    disp,
    a,
    b,
    cam_w,
    cam_h,
    cx,
    cy,
    f,
    height,
    pitch,
    roll,
    max_r,
    prev_obst_streak,
    prev_drop_streak,
    prev_over_streak,
    prev_obst_z,
    prev_drop_z,
    prev_over_z,
    obst_h_min=0.15,
    obst_h_max=1.9,
    drop_h_thresh=-0.25,
    over_h_min=1.35,
    over_h_max=2.05,
    min_cluster_cols=3,
    min_confirm=2,
):
    gh, gw = disp.shape
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    drop_max_z = min(max_r, 3.5)
    over_max_z = min(max_r, 3.5)

    u = (np.arange(gw, dtype=np.float64) + 0.5) * cam_w / gw
    v = (np.arange(gh, dtype=np.float64) + 0.5) * cam_h / gh
    gx = np.tile((u - cx) / f, (gh, 1))
    gy = np.tile(((v - cy) / f)[:, None], (1, gw))
    xr = gx * cr - gy * sr
    yr = gx * sr + gy * cr

    inv = a * disp.astype(np.float64) + b
    valid_inv = np.isfinite(inv) & (inv > 1e-3)
    zc = np.where(valid_inv, 1.0 / np.maximum(inv, 1e-3), np.inf)

    down_dir = yr * cp + sp
    hgt = height - zc * down_dir
    X = zc * xr
    Z = zc * (cp - yr * sp)
    dist = np.hypot(X, Z)

    cols = np.arange(gw)

    # Vat can thong thuong
    obst_mask = valid_inv & (hgt > obst_h_min) & (hgt < obst_h_max) & (Z > 0.2) & (dist < max_r)
    r_obst = np.where(obst_mask, dist, np.inf)
    o_rows = r_obst.argmin(0)
    o_min_r = r_obst[o_rows, cols]
    col_valid = np.isfinite(o_min_r)
    col_x = X[o_rows, cols]
    col_z = Z[o_rows, cols]

    # Phan thap (0.15 < hgt <= 1.20) de loai tru nguoi/tuong khoi overhead
    low_mask = valid_inv & (hgt > obst_h_min) & (hgt <= 1.20) & (Z > 0.2) & (dist < max_r)
    low_z_arr = np.where(low_mask, Z, np.inf).min(axis=0)

    # Vat can ngang dau (treo lo lung, khong cham san ben duoi)
    over_mask = valid_inv & (hgt >= over_h_min) & (hgt < over_h_max) & (Z > 0.25) & (dist < over_max_z)
    r_over = np.where(over_mask, dist, np.inf)
    ov_rows = r_over.argmin(0)
    ov_min_r = r_over[ov_rows, cols]
    over_x = X[ov_rows, cols]
    over_z = Z[ov_rows, cols]
    over_valid = np.isfinite(ov_min_r) & ~(np.isfinite(low_z_arr) & (np.abs(low_z_arr - over_z) < 0.45))

    # Drop-off (bac thang / ho sau)
    drop_mask = (
        valid_inv
        & (~obst_mask)
        & (hgt > -3.0)
        & (hgt < drop_h_thresh)
        & (down_dir > 0.05)
        & (Z > 0.4)
        & (Z < drop_max_z)
    )
    r_drop = np.where(drop_mask, dist, np.inf)
    d_rows = r_drop.argmin(0)
    d_min_r = r_drop[d_rows, cols]
    drop_valid = np.isfinite(d_min_r) & (~col_valid | (d_min_r < o_min_r))
    drop_x = X[d_rows, cols]
    drop_z = Z[d_rows, cols]

    # Sai so san cho auto-pitch
    r_lo = (3 * gh) // 4
    c_lo, c_hi = gw // 4, (3 * gw) // 4
    fmask = (
        valid_inv[r_lo:, c_lo:c_hi]
        & (down_dir[r_lo:, c_lo:c_hi] > 0.1)
        & (Z[r_lo:, c_lo:c_hi] > 0.6)
        & (Z[r_lo:, c_lo:c_hi] < 3.2)
        & (np.abs(hgt[r_lo:, c_lo:c_hi]) < 0.20)
    )
    ferrs = hgt[r_lo:, c_lo:c_hi][fmask]
    floor_err = float(np.median(ferrs)) if ferrs.size >= 8 else math.nan

    obst_pts, new_os, new_oz = _cluster_and_confirm_py(
        col_valid, col_x, col_z, prev_obst_streak, prev_obst_z, min_cluster_cols, min_confirm
    )
    drop_pts, new_ds, new_dz = _cluster_and_confirm_py(
        drop_valid, drop_x, drop_z, prev_drop_streak, prev_drop_z, min_cluster_cols, min_confirm
    )
    over_pts, new_ovs, new_ovz = _cluster_and_confirm_py(
        over_valid, over_x, over_z, prev_over_streak, prev_over_z, min_cluster_cols, min_confirm
    )
    return obst_pts, drop_pts, over_pts, new_os, new_ds, new_ovs, new_oz, new_dz, new_ovz, floor_err


def depth_scan_and_cluster(
    disp,
    a,
    b,
    cam_w,
    cam_h,
    cx,
    cy,
    f,
    height,
    pitch,
    roll,
    max_r,
    prev_obst_streak,
    prev_drop_streak,
    prev_obst_z,
    prev_drop_z,
    obst_h_min=0.15,
    obst_h_max=1.9,
    drop_h_thresh=-0.25,
    min_cluster_cols=3,
    min_confirm=2,
    prev_over_streak=None,
    prev_over_z=None,
    over_h_min=1.35,
    over_h_max=2.05,
    backend=None,
):
    disp_arr = np.ascontiguousarray(disp, dtype=np.float32)
    gw = disp_arr.shape[1]
    pos = np.ascontiguousarray(prev_obst_streak, dtype=np.int32)
    pds = np.ascontiguousarray(prev_drop_streak, dtype=np.int32)
    povs = (
        np.ascontiguousarray(prev_over_streak, dtype=np.int32)
        if prev_over_streak is not None
        else np.zeros(gw, dtype=np.int32)
    )
    poz = np.ascontiguousarray(prev_obst_z, dtype=np.float64)
    pdz = np.ascontiguousarray(prev_drop_z, dtype=np.float64)
    povz = (
        np.ascontiguousarray(prev_over_z, dtype=np.float64)
        if prev_over_z is not None
        else np.zeros(gw, dtype=np.float64)
    )

    if _use_rust(backend):
        obst_pts, drop_pts, over_pts, nos, nds, novs, noz, ndz, novz, ferr = _rust.depth_scan_and_cluster(
            disp_arr,
            float(a),
            float(b),
            float(cam_w),
            float(cam_h),
            float(cx),
            float(cy),
            float(f),
            float(height),
            float(pitch),
            float(roll),
            float(max_r),
            pos,
            pds,
            povs,
            poz,
            pdz,
            povz,
            float(obst_h_min),
            float(obst_h_max),
            float(drop_h_thresh),
            float(over_h_min),
            float(over_h_max),
            int(min_cluster_cols),
            int(min_confirm),
        )
        return (
            obst_pts,
            drop_pts,
            over_pts,
            np.asarray(nos, dtype=np.int32),
            np.asarray(nds, dtype=np.int32),
            np.asarray(novs, dtype=np.int32),
            np.asarray(noz, dtype=np.float64),
            np.asarray(ndz, dtype=np.float64),
            np.asarray(novz, dtype=np.float64),
            float(ferr),
        )

    return depth_scan_and_cluster_py(
        disp_arr,
        float(a),
        float(b),
        float(cam_w),
        float(cam_h),
        float(cx),
        float(cy),
        float(f),
        float(height),
        float(pitch),
        float(roll),
        float(max_r),
        pos,
        pds,
        povs,
        poz,
        pdz,
        povz,
        float(obst_h_min),
        float(obst_h_max),
        float(drop_h_thresh),
        float(over_h_min),
        float(over_h_max),
        int(min_cluster_cols),
        int(min_confirm),
    )


# ======================= 4. KALMAN TRACKER (EGO + HUNGARIAN + JOSEPH + TTL + TTC) =======================
def _hungarian_py(cost):
    if _scipy_lsa is not None:
        r_ind, c_ind = _scipy_lsa(cost)
        return list(zip(r_ind.tolist(), c_ind.tolist()))
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    a = np.zeros((n + 1, n + 1), dtype=np.float64)
    a[1 : n_rows + 1, 1 : n_cols + 1] = cost
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, math.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(int(p[j] - 1), j - 1) for j in range(1, n + 1) if 0 < p[j] <= n_rows and j <= n_cols]


class Track:
    """Kalman van toc khong doi [X, Z, vX, vZ] voi ego-motion compensation va Joseph form."""

    def __init__(self, o, z, R, t, tid):
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([R[0, 0], R[1, 1], 1.0, 1.0]).astype(np.float64)
        self.t = self.seen = float(t)
        self.o, self.id = o, tid

    def apply_ego(self, d_yaw, d_x, d_z):
        if abs(d_yaw) <= 1e-6 and abs(d_x) <= 1e-6 and abs(d_z) <= 1e-6:
            return
        cy, sy = math.cos(d_yaw), math.sin(d_yaw)
        dx1, dz1 = self.x[0] - d_x, self.x[1] - d_z
        self.x[0] = dx1 * cy - dz1 * sy
        self.x[1] = dx1 * sy + dz1 * cy
        vx0, vz0 = self.x[2], self.x[3]
        self.x[2] = vx0 * cy - vz0 * sy
        self.x[3] = vx0 * sy + vz0 * cy
        rot = np.array(
            [[cy, -sy, 0.0, 0.0], [sy, cy, 0.0, 0.0], [0.0, 0.0, cy, -sy], [0.0, 0.0, sy, cy]],
            dtype=np.float64,
        )
        self.P = rot @ self.P @ rot.T

    def predict(self, t):
        dt, self.t = float(t) - self.t, float(t)
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = F[1, 3] = dt
        G = np.array(
            [[dt * dt / 2.0, 0.0], [0.0, dt * dt / 2.0], [dt, 0.0], [0.0, dt]],
            dtype=np.float64,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (G @ G.T) * (ACCEL ** 2)

    def innovation(self, z, R):
        return z - self.x[:2], self.P[:2, :2] + R

    def update(self, z, R):
        y, S = self.innovation(z, R)
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ y
        H = np.eye(2, 4, dtype=np.float64)
        I_KH = np.eye(4, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


class PythonKalmanTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def step(self, objs, t, det_every=1, fps=15.0, ego=None):
        t = float(t)
        d_yaw, d_x, d_z = ego if ego is not None else (0.0, 0.0, 0.0)
        for tr in self.tracks:
            tr.apply_ego(d_yaw, d_x, d_z)
            tr.predict(t)
        fused = [fuse(o["meas"]) for o in objs]

        n_tr, n_obj = len(self.tracks), len(objs)
        used_t, used_o = set(), set()
        if n_tr > 0 and n_obj > 0:
            BIG = 1e9
            cost = np.full((n_tr, n_obj), BIG, dtype=np.float64)
            for i, tr in enumerate(self.tracks):
                for j, (o, (z, R)) in enumerate(zip(objs, fused)):
                    if tr.o["cls"] != o["cls"]:
                        continue
                    y, S = tr.innovation(z, R)
                    d2 = float(y @ np.linalg.solve(S, y))
                    if d2 < GATE and math.isfinite(d2):
                        cost[i, j] = d2
            for i, j in _hungarian_py(cost):
                if cost[i, j] < GATE:
                    tr = self.tracks[i]
                    tr.update(*fused[j])
                    tr.o, tr.seen = objs[j], t
                    used_t.add(i)
                    used_o.add(j)

        for j, o in enumerate(objs):
            if j not in used_o:
                self.tracks.append(Track(o, *fused[j], t, self.next_id))
                self.next_id += 1

        static_ttl = max(TRACK_TTL_STATIC, (max(1, det_every) / max(fps, 1.0)) * 2.5)
        self.tracks = [
            tr
            for tr in self.tracks
            if t - tr.seen < (TRACK_TTL_PERSON if tr.o["cls"] == 0 else static_ttl)
        ]

        out = []
        for tr in self.tracks:
            X, Z, vx, vz = float(tr.x[0]), float(tr.x[1]), float(tr.x[2]), float(tr.x[3])
            if Z <= 0.05:
                continue
            cs, x_cpa, cpa_dist, ttc = compute_ttc_cpa(X, Z, vx, vz)
            out.append(
                dict(
                    tr.o,
                    X=X,
                    Z=Z,
                    r=math.hypot(X, Z),
                    vx=vx,
                    vz=vz,
                    id=tr.id,
                    fresh=(tr.seen == t),
                    closing_speed=cs,
                    x_cpa=x_cpa,
                    cpa=cpa_dist,
                    ttc=ttc,
                    is_moving=(math.hypot(vx, vz) > 0.35),
                )
            )
        return out


class KalmanTracker:
    """Wrapper chon Rust KalmanTrackerRs hoac PythonKalmanTracker theo backend."""

    def __init__(self, backend=None):
        self._backend = backend
        self._py = None
        self._rs = None
        self._meta = {}
        if _use_rust(backend):
            self._rs = _rust.KalmanTrackerRs()
        else:
            self._py = PythonKalmanTracker()

    @property
    def tracks(self):
        return self._py.tracks if self._py is not None else []

    def step(self, objs, t, det_every=1, fps=15.0, ego=None):
        if self._py is not None:
            return self._py.step(objs, t, det_every=det_every, fps=fps, ego=ego)

        d_yaw, d_x, d_z = ego if ego is not None else (0.0, 0.0, 0.0)
        obj_cls = np.asarray([int(o["cls"]) for o in objs], dtype=np.int32)
        m_idx, m_z, m_r = [], [], []
        for j, o in enumerate(objs):
            for z, R in o["meas"]:
                m_idx.append(j)
                m_z.append([float(z[0]), float(z[1])])
                m_r.append([float(R[0, 0]), float(R[0, 1]), float(R[1, 0]), float(R[1, 1])])

        idx_arr = np.asarray(m_idx, dtype=np.int32)
        z_arr = np.asarray(m_z, dtype=np.float64) if m_z else np.empty((0, 2), dtype=np.float64)
        r_arr = np.asarray(m_r, dtype=np.float64) if m_r else np.empty((0, 4), dtype=np.float64)

        rows = self._rs.step_raw(
            obj_cls,
            idx_arr,
            z_arr,
            r_arr,
            float(t),
            int(det_every),
            float(fps),
            float(GATE),
            float(ACCEL),
            float(TRACK_TTL_PERSON),
            float(TRACK_TTL_STATIC),
            float(MIN_CLOSING_SPEED),
            float(d_yaw),
            float(d_x),
            float(d_z),
        )

        alive_ids = set()
        out = []
        for (tid, obj_idx, cls_id, fresh), (X, Z, vx, vz, r, cs, x_cpa, cpa_dist, ttc_val) in rows:
            alive_ids.add(tid)
            if obj_idx >= 0:
                base = dict(objs[obj_idx])
                self._meta[tid] = base
            else:
                base = dict(self._meta.get(tid, {"cls": cls_id, "name": str(cls_id), "mode": "ok"}))
            ttc = None if math.isnan(ttc_val) else float(ttc_val)
            out.append(
                dict(
                    base,
                    X=float(X),
                    Z=float(Z),
                    r=float(r),
                    vx=float(vx),
                    vz=float(vz),
                    id=int(tid),
                    fresh=bool(fresh),
                    closing_speed=float(cs),
                    x_cpa=float(x_cpa),
                    cpa=float(cpa_dist),
                    ttc=ttc,
                    is_moving=(math.hypot(vx, vz) > 0.35),
                )
            )
        for tid in list(self._meta):
            if tid not in alive_ids:
                del self._meta[tid]
        return out


# ======================= 5. 2D OCCUPANCY GRID & SAFE STEERING =======================
class PythonOccupancyGrid2D:
    GRID_N = 60
    CELL_M = 0.10
    HALF_X = 3.0

    def __init__(self):
        self.occ = np.zeros((self.GRID_N, self.GRID_N), dtype=np.float32)
        self.kind = np.zeros((self.GRID_N, self.GRID_N), dtype=np.uint8)
        coords = (np.arange(self.GRID_N, dtype=np.float64) + 0.5) * self.CELL_M
        self._zn = np.tile(coords[:, None], (1, self.GRID_N))
        self._xn = np.tile((coords - self.HALF_X)[None, :], (self.GRID_N, 1))

    def reset(self):
        self.occ.fill(0.0)
        self.kind.fill(0)

    def _add_hit(self, x, z, delta, k):
        if -self.HALF_X <= x < self.HALF_X and 0.0 <= z < self.GRID_N * self.CELL_M:
            c = int(math.floor((x + self.HALF_X) / self.CELL_M))
            r = int(math.floor(z / self.CELL_M))
            if 0 <= r < self.GRID_N and 0 <= c < self.GRID_N:
                self.occ[r, c] = min(1.0, float(self.occ[r, c]) + delta)
                if k >= self.kind[r, c]:
                    self.kind[r, c] = k

    def step(
        self,
        d_yaw,
        d_x,
        d_z,
        dt,
        hfov_deg,
        min_visible,
        obstacles,
        dropoffs,
        overheads,
        tracks,
        decay_tau=1.8,
    ):
        dt_c = max(0.005, min(0.5, float(dt)))
        decay_blind = math.exp(-dt_c / max(0.2, float(decay_tau)))
        decay_vis = math.exp(-dt_c / 0.55)
        half_fov = math.radians(hfov_deg * 0.5)
        cy, sy = math.cos(d_yaw), math.sin(d_yaw)

        x_old = self._xn * cy + self._zn * sy + d_x
        z_old = -self._xn * sy + self._zn * cy + d_z
        co = np.floor((x_old + self.HALF_X) / self.CELL_M).astype(np.int32)
        ro = np.floor(z_old / self.CELL_M).astype(np.int32)
        valid = (co >= 0) & (co < self.GRID_N) & (ro >= 0) & (ro < self.GRID_N)

        next_occ = np.zeros_like(self.occ)
        next_kind = np.zeros_like(self.kind)
        if np.any(valid):
            val = self.occ[ro[valid], co[valid]]
            dist = np.hypot(self._xn[valid], self._zn[valid])
            ang = np.abs(np.arctan2(self._xn[valid], np.maximum(self._zn[valid], 1e-3)))
            in_vis = (dist >= min_visible) & (ang <= half_fov)
            dec = val * np.where(in_vis, decay_vis, decay_blind).astype(np.float32)
            keep = dec >= 0.04
            vr, vc = np.where(valid)
            kr, kc = vr[keep], vc[keep]
            next_occ[kr, kc] = dec[keep]
            next_kind[kr, kc] = self.kind[ro[kr, kc], co[kr, kc]]
        self.occ, self.kind = next_occ, next_kind

        for x, z in obstacles:
            self._add_hit(x, z, 0.55, 1)
        for x, z in tracks:
            for ox in (-0.15, 0.0, 0.15):
                for oz in (-0.15, 0.0, 0.15):
                    self._add_hit(x + ox, z + oz, 0.65, 1)
        for x, z in dropoffs:
            self._add_hit(x, z, 0.75, 2)
        for x, z in overheads:
            self._add_hit(x, z, 0.70, 3)

        action, target_deg, clearance_m = self.compute_steering(hfov_deg, 0.65, 5.0)
        mask = self.occ >= 0.30
        rr, cc = np.where(mask)
        cells = [
            (
                float((c + 0.5) * self.CELL_M - self.HALF_X),
                float((r + 0.5) * self.CELL_M),
                float(self.occ[r, c]),
                int(self.kind[r, c]),
            )
            for r, c in zip(rr.tolist(), cc.tolist())
        ]
        return action, target_deg, clearance_m, cells

    def compute_steering(self, hfov_deg, passage_width=0.65, max_range=5.0):
        half_fov = max(15.0, min(60.0, float(hfov_deg) * 0.5))
        half_w = passage_width * 0.5
        lat_samples = (-half_w, -half_w * 0.5, 0.0, half_w * 0.5, half_w)

        best_deg, best_score, best_clear = 0.0, -math.inf, 0.0
        straight_clear = max_range
        n_steps = int(round((half_fov * 2.0) / 2.0))
        for k in range(n_steps + 1):
            deg = -half_fov + k * 2.0
            rad = math.radians(deg)
            sin_t, cos_t = math.sin(rad), math.cos(rad)
            clear = max_range
            s = 0.25
            while s <= max_range:
                cx, cz = s * sin_t, s * cos_t
                hit = False
                for lw in lat_samples:
                    px = cx + lw * cos_t
                    pz = cz - lw * sin_t
                    if -self.HALF_X <= px < self.HALF_X and 0.0 <= pz < self.GRID_N * self.CELL_M:
                        c = int(math.floor((px + self.HALF_X) / self.CELL_M))
                        r = int(math.floor(pz / self.CELL_M))
                        if self.occ[r, c] >= 0.35:
                            hit = True
                            break
                if hit:
                    clear = s
                    break
                s += self.CELL_M

            if abs(deg) < 1.1:
                straight_clear = clear
            score = clear - 0.015 * abs(deg)
            if score > best_score:
                best_score, best_deg, best_clear = score, deg, clear

        if straight_clear >= 2.2:
            return "straight", 0.0, straight_clear
        if best_clear < 1.0:
            return "stop", best_deg, best_clear
        if best_clear < 1.5:
            return "caution", best_deg, best_clear
        if best_deg < -4.0:
            return "left", best_deg, best_clear
        if best_deg > 4.0:
            return "right", best_deg, best_clear
        return "straight", 0.0, straight_clear


class OccupancyGrid2D:
    """Wrapper chon Rust OccupancyGridRs hoac PythonOccupancyGrid2D."""

    def __init__(self, backend=None):
        self._rs = _rust.OccupancyGridRs() if _use_rust(backend) else None
        self._py = None if self._rs is not None else PythonOccupancyGrid2D()

    def reset(self):
        if self._rs is not None:
            self._rs.reset()
        else:
            self._py.reset()

    def step(
        self,
        d_yaw,
        d_x,
        d_z,
        dt,
        hfov_deg,
        min_visible,
        obstacles=(),
        dropoffs=(),
        overheads=(),
        tracks=(),
        decay_tau=1.8,
    ):
        obs_pts = [(float(x), float(z)) for x, z in obstacles]
        drp_pts = [(float(x), float(z)) for x, z in dropoffs]
        ov_pts = [(float(x), float(z)) for x, z in overheads]
        tr_pts = [(float(t["X"]), float(t["Z"])) if isinstance(t, dict) else (float(t[0]), float(t[1])) for t in tracks]

        if self._rs is not None:
            action, target_deg, clearance_m, cells = self._rs.step(
                float(d_yaw),
                float(d_x),
                float(d_z),
                float(dt),
                float(hfov_deg),
                float(min_visible),
                obs_pts,
                drp_pts,
                ov_pts,
                tr_pts,
                float(decay_tau),
            )
        else:
            action, target_deg, clearance_m, cells = self._py.step(
                float(d_yaw),
                float(d_x),
                float(d_z),
                float(dt),
                float(hfov_deg),
                float(min_visible),
                obs_pts,
                drp_pts,
                ov_pts,
                tr_pts,
                float(decay_tau),
            )

        guidance = {
            "action": action,
            "target_deg": round(float(target_deg), 1),
            "clearance_m": round(float(clearance_m), 2),
        }
        return guidance, cells

    def compute_steering(self, hfov_deg, passage_width=0.65, max_range=5.0):
        if self._rs is not None:
            return self._rs.compute_steering(float(hfov_deg), float(passage_width), float(max_range))
        return self._py.compute_steering(float(hfov_deg), float(passage_width), float(max_range))
