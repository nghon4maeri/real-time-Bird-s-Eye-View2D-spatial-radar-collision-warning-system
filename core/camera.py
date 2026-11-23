# -*- coding: utf-8 -*-
"""Mo hinh hinh hoc camera pinhole co bu 2 truc (pitch + roll)."""
import math


class Camera:
    """Mo hinh camera pinhole cam o do cao h, cui xuong goc pitch va nghieng ngang roll."""

    def __init__(self, w, h, hfov_deg, height, pitch_deg, roll_deg=0.0):
        self.w, self.h = int(w), int(h)
        self.hfov_deg = float(hfov_deg)
        self.f = (self.w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.height = float(height)
        self.pitch = math.radians(pitch_deg)
        self.roll = math.radians(roll_deg)

    def set_hfov(self, hfov_deg):
        self.hfov_deg = max(30.0, min(140.0, float(hfov_deg)))
        self.f = (self.w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def unroll(self, x, y):
        """Xoay nguoc tia camera chuan hoa (x, y) theo goc roll ve he thang."""
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        return x * cr - y * sr, x * sr + y * cr

    def unrolled_y(self, u, v):
        """Toa do y chuan hoa sau khi khu roll tai pixel (u, v)."""
        x = (u - self.cx) / self.f
        y = (v - self.cy) / self.f
        return x * math.sin(self.roll) + y * math.cos(self.roll)

    def project(self, X, down, Z):
        """Chieu diem 3D (X: phai, down: xuong tu camera, Z: truoc) len pixel (u, v)."""
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        xp = X
        yp = down * cp - Z * sp
        zc = down * sp + Z * cp
        xc = xp * cr + yp * sr
        yc = -xp * sr + yp * cr
        return self.cx + self.f * xc / zc, self.cy + self.f * yc / zc

    def floor_point(self, u, v):
        """Pixel (u, v) -> diem tren san (X: ngang, Z: phia truoc), don vi met.
        Tra ve None neu tia nhin nam tren duong chan troi (khong cat san)."""
        x = (u - self.cx) / self.f
        y = (v - self.cy) / self.f
        xr, yr = self.unroll(x, y)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        down = yr * cp + sp
        if down <= 1e-3:
            return None
        t = self.height / down
        return t * xr, t * (cp - yr * sp)

    def min_visible(self):
        """Khoang cach san gan nhat camera con nhin thay (mep duoi anh)."""
        p = self.floor_point(self.cx, self.h - 1)
        return p[1] if p else 0.0

    def horizon_v(self, u=None):
        """Dong anh v cua duong chan troi tai cot u (mac dinh tam anh cx)."""
        if u is None:
            u = self.cx
        x = (u - self.cx) / self.f
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        if abs(cr) < 1e-4:
            return float("inf")
        y = (-math.tan(self.pitch) - x * sr) / cr
        return self.cy + self.f * y

    def horizon_line(self):
        """Hai diem dau mut duong chan troi ((0, v0), (w-1, v1)) co tinh roll."""
        return (0, int(round(self.horizon_v(0)))), (self.w - 1, int(round(self.horizon_v(self.w - 1))))

    def calibrate_pitch(self, u, v, dist):
        """Tim goc cui sao cho diem (u, v) tren san cach camera dung `dist` m."""
        lo, hi = math.radians(-20.0), math.radians(75.0)
        for _ in range(60):
            self.pitch = (lo + hi) / 2.0
            p = self.floor_point(u, v)
            if p is None or math.hypot(*p) > dist:
                lo = self.pitch
            else:
                hi = self.pitch
        self.pitch = (lo + hi) / 2.0


def size_distance(cam, v_top, v_bot, H, u_top=None, u_bot=None):
    """Khoang cach den vat cao H dung tren san, co bu roll neu truyen u_top/u_bot."""
    ut = cam.cx if u_top is None else u_top
    ub = cam.cx if u_bot is None else u_bot
    yb = cam.unrolled_y(ub, v_bot)
    yt = cam.unrolled_y(ut, v_top)
    tb = math.tan(cam.pitch + math.atan(yb))
    tt = math.tan(cam.pitch + math.atan(yt))
    diff = tb - tt
    return H / diff if diff > 1e-4 else None
