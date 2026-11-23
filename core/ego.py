# -*- coding: utf-8 -*-
"""Uoc luong chuyen dong nguoi cam camera (Ego-Motion) bang Shi-Tomasi + LK Optical Flow."""
import math

import cv2
import numpy as np


class EgoMotionEstimator:
    """Uoc luong (d_yaw, d_x, d_z) cua camera giua 2 khung lien tiep tren nen tinh (mask vat can)."""

    def __init__(self, max_corners=80, proc_w=320):
        self.max_corners = int(max_corners)
        self.proc_w = int(proc_w)
        self.prev_gray = None
        self.prev_pts = None
        self.prev_cam_snap = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.prev_cam_snap = None

    def _build_mask(self, gh, gw, scale, objs):
        mask = np.full((gh, gw), 255, dtype=np.uint8)
        for o in objs:
            box = o.get("box") if isinstance(o, dict) else o
            if box is None:
                continue
            x1, y1, x2, y2 = box
            mx1 = max(0, int((x1 - 6.0) * scale))
            my1 = max(0, int((y1 - 6.0) * scale))
            mx2 = min(gw, int((x2 + 6.0) * scale))
            my2 = min(gh, int((y2 + 6.0) * scale))
            if mx2 > mx1 and my2 > my1:
                mask[my1:my2, mx1:mx2] = 0
        return mask

    def estimate(self, frame, cam, objs=()):
        """Tra ve dict {'d_yaw', 'd_x', 'd_z', 'inliers', 'conf'} trong he met/radian."""
        if frame is None or frame.size == 0:
            return {"d_yaw": 0.0, "d_x": 0.0, "d_z": 0.0, "inliers": 0, "conf": 0.0}

        scale = min(1.0, self.proc_w / float(cam.w))
        gw = max(64, int(round(cam.w * scale)))
        gh = max(48, int(round(cam.h * scale)))

        small = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        mask = self._build_mask(gh, gw, scale, objs)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 12:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=14, mask=mask
            )
            self.prev_cam_snap = (cam.pitch, getattr(cam, "roll", 0.0))
            return {"d_yaw": 0.0, "d_x": 0.0, "d_z": 0.0, "inliers": 0, "conf": 0.0}

        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )

        d_yaw, d_x, d_z, n_inliers = 0.0, 0.0, 0.0, 0
        if nxt is not None and st is not None:
            ok_mask = (st.ravel() == 1) & (err.ravel() < 14.0)
            p0 = self.prev_pts[ok_mask].reshape(-1, 2) / scale
            p1 = nxt[ok_mask].reshape(-1, 2) / scale

            # Loai bo diem nam trong box vat can o khung hien tai
            keep_bg = []
            for idx, (u1, v1) in enumerate(p1):
                iu, iv = int(u1 * scale), int(v1 * scale)
                if 0 <= iu < gw and 0 <= iv < gh and mask[iv, iu] > 0:
                    keep_bg.append(idx)

            if len(keep_bg) >= 6:
                p0 = p0[keep_bg]
                p1 = p1[keep_bg]
                # Tinh goc tia ngang sau khi bu roll hien tai
                yaws = []
                for (u0, v0), (u1, v1) in zip(p0, p1):
                    xr0, _ = cam.unroll((u0 - cam.cx) / cam.f, (v0 - cam.cy) / cam.f)
                    xr1, _ = cam.unroll((u1 - cam.cx) / cam.f, (v1 - cam.cy) / cam.f)
                    yaws.append(math.atan2(xr0, 1.0) - math.atan2(xr1, 1.0))

                yaws_arr = np.asarray(yaws, dtype=np.float64)
                med_yaw = float(np.median(yaws_arr))
                mad = float(np.median(np.abs(yaws_arr - med_yaw)))
                inliers = np.abs(yaws_arr - med_yaw) <= max(0.015, 2.5 * mad)
                n_inliers = int(np.sum(inliers))
                if n_inliers >= 5:
                    d_yaw = float(np.median(yaws_arr[inliers]))
                    if abs(d_yaw) < 0.002:
                        d_yaw = 0.0
                    d_yaw = max(-0.35, min(0.35, d_yaw))

                    # Uoc luong tinh tien (d_x, d_z) tu cac diem nam tren mat san
                    cy, sy = math.cos(d_yaw), math.sin(d_yaw)
                    dx_list, dz_list = [], []
                    for idx in np.where(inliers)[0]:
                        fp0 = cam.floor_point(float(p0[idx, 0]), float(p0[idx, 1]))
                        fp1 = cam.floor_point(float(p1[idx, 0]), float(p1[idx, 1]))
                        if fp0 is None or fp1 is None:
                            continue
                        x0, z0 = fp0
                        x1, z1 = fp1
                        if 0.6 <= z0 <= 4.5 and 0.6 <= z1 <= 4.5:
                            x1_rot = x1 * cy + z1 * sy
                            z1_rot = -x1 * sy + z1 * cy
                            dx_list.append(x0 - x1_rot)
                            dz_list.append(z0 - z1_rot)
                    if len(dx_list) >= 4:
                        d_x = float(np.median(dx_list))
                        d_z = float(np.median(dz_list))
                        if abs(d_x) < 0.005:
                            d_x = 0.0
                        if abs(d_z) < 0.005:
                            d_z = 0.0
                        d_x = max(-0.45, min(0.45, d_x))
                        d_z = max(-0.60, min(0.60, d_z))

        self.prev_gray = gray
        if n_inliers >= 35 and nxt is not None:
            self.prev_pts = nxt[ok_mask][keep_bg].reshape(-1, 1, 2).astype(np.float32)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=14, mask=mask
            )
        self.prev_cam_snap = (cam.pitch, getattr(cam, "roll", 0.0))

        conf = min(1.0, n_inliers / 25.0)
        return {
            "d_yaw": float(d_yaw),
            "d_x": float(d_x),
            "d_z": float(d_z),
            "inliers": n_inliers,
            "conf": round(conf, 2),
        }
