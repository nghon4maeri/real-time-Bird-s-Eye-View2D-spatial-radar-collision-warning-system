#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RADAR 2D TU WEBCAM
------------------
Dung chung engine core/ voi web.py, ho tro bat Pose, Depth, Kalman, Sensor qua co CLI.
Chay:      python radar.py                   (webcam mac dinh)
           python radar.py --cam 1           (webcam khac)
           python radar.py --video demo.mp4 --pose --depth --kalman

Phim tat:
  c       : che do hieu chinh -> dat 1 vat cach dung CALIB_DIST met,
            click chuot vao diem vat cham san tren anh camera
  + / -   : tang / giam goc cui camera 1 do
  ] / [   : tang / giam chieu cao camera 5 cm
  f       : lat anh trai-phai
  m       : bat / tat am thanh
  q / ESC : thoat
"""
import argparse
import math
import time

import cv2
import numpy as np

from core import accel
from core.accel import KalmanTracker, Track, resolve_occlusion
from core.camera import Camera, size_distance
from core.config import (
    ACCEL,
    BOX_W,
    CALIB_DIST,
    CAM_HEIGHT,
    CLASSES,
    CONF,
    CPA_CORRIDOR,
    D_RED,
    D_YELLOW,
    GATE,
    HEAD,
    HEIGHTS,
    HFOV_DEG,
    HYST,
    IMGSZ,
    KP_CONF,
    LEVEL_COLOR,
    LEVEL_NAME,
    MEAS_REL,
    MIN_CLOSING_SPEED,
    MODEL,
    OCCLUDED_GAP,
    PITCH_DEG,
    POSE_MODEL,
    RADAR_RANGE,
    SAT_SIGMA,
    SECTOR_NAME,
    SHOULDER,
    TORSO,
    TRACK_TTL,
    TTC_RED,
    TTC_YELLOW,
)
from core.estimator import estimate, person_distance
from core.pipeline import Beeper, FrameReader, RadarEngine, draw_camera, draw_radar, put
from core.tracker import SectorTracker, compute_ttc_cpa, dynamic_level_of, fuse, level_of, meas_cov

WIN = "Radar 2D - webcam"

__all__ = [
    "ACCEL", "BOX_W", "CALIB_DIST", "CAM_HEIGHT", "CLASSES", "CONF", "CPA_CORRIDOR",
    "D_RED", "D_YELLOW", "GATE", "HEAD", "HEIGHTS", "HFOV_DEG", "HYST", "IMGSZ",
    "KP_CONF", "LEVEL_COLOR", "LEVEL_NAME", "MEAS_REL", "MIN_CLOSING_SPEED", "MODEL",
    "OCCLUDED_GAP", "PITCH_DEG", "POSE_MODEL", "RADAR_RANGE", "SAT_SIGMA", "SECTOR_NAME",
    "SHOULDER", "TORSO", "TRACK_TTL", "TTC_RED", "TTC_YELLOW", "WIN",
    "Beeper", "Camera", "KalmanTracker", "SectorTracker", "Track",
    "compute_ttc_cpa", "draw_camera", "draw_radar", "dynamic_level_of", "estimate",
    "fuse", "level_of", "meas_cov", "person_distance", "put", "resolve_occlusion",
    "size_distance",
]


def compose(cam_img, radar_img, cam, fps, beeper, flip, backend_info=""):
    h = cam_img.shape[0]
    bar = np.full((44, cam_img.shape[1] + radar_img.shape[1], 3), 35, np.uint8)
    info = (
        f"FPS {fps:4.1f} | cao {cam.height:.2f}m  cui {math.degrees(cam.pitch):.1f}deg  "
        f"roll {math.degrees(cam.roll):.1f}deg  FOV {cam.hfov_deg:.0f}deg | "
        f"am thanh {'TAT' if beeper.muted else 'BAT'}"
        f"{' | LAT' if flip else ''}"
        f"{' | ' + backend_info if backend_info else ''}"
    )
    put(bar, info, (10, 18), (220, 220, 220), 0.48)
    put(
        bar,
        "[c] hieu chinh  [+/-] goc cui  [ [/] ] chieu cao  [f] lat  [m] am thanh  [q] thoat",
        (10, 37),
        (150, 150, 150),
        0.45,
    )
    return np.vstack([np.hstack([cam_img, radar_img[:h]]), bar])


def main():
    ap = argparse.ArgumentParser(description="Radar 2D tu webcam")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--hfov", type=float, default=HFOV_DEG)
    ap.add_argument("--height", type=float, default=CAM_HEIGHT)
    ap.add_argument("--pitch", type=float, default=PITCH_DEG)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--pose-model", type=str, default=POSE_MODEL)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--det-every", type=int, default=3)
    ap.add_argument("--pose", action="store_true", help="bat model pose")
    ap.add_argument("--depth", action="store_true", help="bat Depth Anything V2")
    ap.add_argument("--kalman", action="store_true", help="bat Kalman tracking + TTC")
    ap.add_argument("--sensor", action="store_true", help="bat cam bien nghieng")
    ap.add_argument("--no-ego", action="store_true", help="tat bu chuyen dong camera")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx", "openvino"])
    ap.add_argument("--depth-backend", type=str, default=None, choices=["auto", "pytorch", "onnx", "openvino"])
    args = ap.parse_args()

    reader = FrameReader(args.video if args.video else args.cam, realtime=True)
    reader.start()

    engine = RadarEngine(
        reader.w,
        reader.h,
        args.hfov,
        args.height,
        args.pitch,
        model_path=args.model,
        pose_model_path=args.pose_model,
        imgsz=args.imgsz,
        det_every=args.det_every,
        use_pose=args.pose,
        use_depth=args.depth,
        use_kalman=args.kalman,
        use_sensor=args.sensor,
        use_ego=not args.no_ego,
        backend=args.backend,
        depth_backend=args.depth_backend,
    )
    print(f"[Radar] compute={accel.get_backend()} ai={engine.ai_backend}", flush=True)

    beeper = Beeper()
    beeper.start()
    state = {"calib": False, "click": None}
    flip = False
    last_seq = 0

    def on_mouse(event, x, y, flags, st):
        if event == cv2.EVENT_LBUTTONDOWN and st["calib"] and x < reader.w and y < reader.h:
            st["click"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse, state)

    try:
        while True:
            ok, frame, _, seq = reader.read_latest(last_seq)
            if not ok or frame is None:
                beeper.level = 2
                err = np.zeros((reader.h, reader.w + reader.h, 3), np.uint8)
                put(err, "MAT TIN HIEU CAMERA - DUNG!", (20, reader.h // 2), (0, 0, 255), 1.0, 2)
                cv2.imshow(WIN, err)
                if cv2.waitKey(200) & 0xFF in (ord("q"), 27):
                    break
                continue
            if seq == last_seq:
                time.sleep(0.005)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_seq = seq
            frame = frame.copy()
            if flip:
                frame = cv2.flip(frame, 1)

            vis, objs, scan, dropoffs, overall, _, health = engine.step(frame, draw_keypoints=args.pose)
            beeper.level = overall

            if state["click"]:
                engine.calibrate_click(*state["click"])
                state["click"], state["calib"] = None, False

            out = compose(
                draw_camera(vis, engine.cam, objs, state["calib"], engine.calib_dist, health=health),
                draw_radar(
                    reader.h,
                    engine.cam,
                    objs,
                    engine.tracker,
                    overall,
                    scan=scan,
                    dropoffs=dropoffs,
                    overheads=engine.last_overheads,
                    guidance=engine.last_guidance,
                ),
                engine.cam,
                engine.fps,
                beeper,
                flip,
                backend_info=f"{accel.get_backend().upper()}/{engine.ai_backend}",
            )
            cv2.imshow(WIN, out)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("c"):
                state["calib"] = not state["calib"]
            elif k in (ord("+"), ord("=")):
                if engine.sensor.tilt is not None:
                    engine.offset += 1.0
                else:
                    engine.cam.pitch += math.radians(1.0)
            elif k == ord("-"):
                if engine.sensor.tilt is not None:
                    engine.offset -= 1.0
                else:
                    engine.cam.pitch -= math.radians(1.0)
            elif k == ord("]"):
                engine.cam.height += 0.05
            elif k == ord("["):
                engine.cam.height = max(0.2, engine.cam.height - 0.05)
            elif k == ord("f"):
                flip = not flip
            elif k == ord("m"):
                beeper.muted = not beeper.muted
    finally:
        reader.stop()
        engine.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
