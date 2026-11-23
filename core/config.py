# -*- coding: utf-8 -*-
"""Cau hinh mac dinh va quan ly luu/nap config.json."""
import json
import os
import tempfile

# ======================= CAU HINH MAC DINH =======================
HFOV_DEG = 70.0       # goc nhin ngang cua webcam (webcam laptop thuong 60-80)
CAM_HEIGHT = 1.5      # chieu cao webcam so voi san (m)
PITCH_DEG = 25.0      # goc cui xuong (do)
ROLL_DEG = 0.0        # goc nghieng ngang (do)
D_RED = 1.5           # < 1.5 m  -> DO
D_YELLOW = 2.5        # < 2.5 m  -> VANG
RADAR_RANGE = 5.0     # ban kinh hien thi radar (m)
CALIB_DIST = 2.0      # khoang cach vat mau khi hieu chinh (m)
HYST = 0.15           # do tre chuyen vung (m), chong nhap nhay
MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"
IMGSZ = 416
CONF = 0.35

# Lop COCO coi la vat can tren san (co ban + mo rong)
BASE_CLASSES = {
    0: "nguoi",
    13: "ghe dai",
    24: "ba lo",
    28: "vali",
    56: "ghe",
    57: "sofa",
    58: "chau cay",
    60: "ban",
}

EXTRA_CLASSES = {
    1: "xe dap",
    3: "xe may",
    10: "tru nuoc",
    11: "bien bao",
    15: "meo",
    16: "cho",
}

CLASSES = {**BASE_CLASSES, **EXTRA_CLASSES}

# Chieu cao trung binh (m) +- dung sai cho tung lop
HEIGHTS = {
    0:  [(1.65, 0.15), (1.25, 0.15)],   # nguoi: dung, ngoi
    1:  [(1.05, 0.20)],                 # xe dap
    3:  [(1.15, 0.20)],                 # xe may
    10: [(0.75, 0.20)],                 # tru nuoc
    11: [(1.80, 0.35)],                 # bien bao
    13: [(0.45, 0.10)],                 # ghe dai
    15: [(0.28, 0.10)],                 # meo
    16: [(0.50, 0.20)],                 # cho
    24: [(0.45, 0.10)],                 # ba lo
    28: [(0.65, 0.15)],                 # vali
    56: [(0.85, 0.15)],                 # ghe
    57: [(0.85, 0.15)],                 # sofa
    58: [(0.60, 0.20)],                 # chau cay
    60: [(0.75, 0.20)],                 # ban
}

# Kich thuoc co the trung binh (m) + sai so tuong doi (pose)
TORSO = (0.50, 0.12)     # vai -> hong, doan DOC
SHOULDER = (0.38, 0.10)  # vai trai - vai phai, doan NGANG
HEAD = (0.15, 0.15)      # tai trai - tai phai
BOX_W = (0.45, 0.35)     # be rong box nguoi
KP_CONF = 0.5

OCCLUDED_GAP = 0.5       # vat bi che coi nhu nam sau vat che 0.5 m

# Kalman & tracking
MEAS_REL = {
    "ok": 0.08,
    "pose": 0.08,
    "san": 0.12,
    "cao": 0.15,
    "depth": 0.15,
    "hc": 0.25,
    "che": 0.30,
}
SAT_SIGMA = 0.6
ACCEL = 2.0
GATE = 9.21              # chi^2 2 dof, 99%
TRACK_TTL = 0.7          # giu tuong thich nguoc
TRACK_TTL_PERSON = 0.7
TRACK_TTL_STATIC = 1.5

# TTC & CPA canh bao dong
TTC_RED = 1.5            # s -> NGUY HIEM
TTC_YELLOW = 3.0         # s -> CHU Y
CPA_CORRIDOR = 0.6       # m hanh lang va cham
MIN_CLOSING_SPEED = 0.3  # m/s

LEVEL_NAME = ["AN TOAN", "CHU Y", "NGUY HIEM"]
LEVEL_COLOR = [(80, 200, 80), (0, 200, 255), (0, 0, 255)]  # BGR
SECTOR_NAME = ["TRAI", "GIUA", "PHAI"]

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)

PERSISTENT_DEFAULTS = {
    "pitch": PITCH_DEG,
    "offset": 0.0,
    "height": CAM_HEIGHT,
    "hfov": HFOV_DEG,
    "calib_dist": CALIB_DIST,
    "flip": False,
    "muted": False,
    "voice_enabled": False,
    "spatial_audio": True,
    "radar_range": RADAR_RANGE,
}


def load_config(path=DEFAULT_CONFIG_PATH):
    """Nap cau hinh tu JSON, hop nhat voi gia tri mac dinh."""
    cfg = dict(PERSISTENT_DEFAULTS)
    if not path or not os.path.exists(path):
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, default in PERSISTENT_DEFAULTS.items():
                if k in data and data[k] is not None:
                    try:
                        if isinstance(default, bool):
                            cfg[k] = bool(data[k])
                        elif isinstance(default, float):
                            cfg[k] = float(data[k])
                        elif isinstance(default, int):
                            cfg[k] = int(data[k])
                        else:
                            cfg[k] = data[k]
                    except (TypeError, ValueError):
                        pass
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    """Luu cau hinh ra JSON an toan (ghi tam roi replace)."""
    if not path:
        return
    data = {k: cfg.get(k, PERSISTENT_DEFAULTS[k]) for k in PERSISTENT_DEFAULTS}
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cfg_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
