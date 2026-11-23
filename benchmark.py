#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cong cu Benchmark, Record & Replay: so sanh Rust vs Python, stage profiling, va Ego-Motion A/B."""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from core import accel
from core.camera import Camera
from core.config import CAM_HEIGHT, HFOV_DEG, MODEL, PITCH_DEG, POSE_MODEL
from core.pipeline import RadarEngine, SensorReader
from core.tracker import meas_cov


def bench_core(iters=300):
    cam = Camera(640, 480, 70.0, 1.5, 22.0, roll_deg=8.0)
    gw, gh = 168, 128
    disp = np.full((gh, gw), 1.2, dtype=np.float32)
    disp[20:70, 40:130] = 1.6
    os_buf = np.zeros(gw, dtype=np.int32)
    ds_buf = np.zeros(gw, dtype=np.int32)
    ovs_buf = np.zeros(gw, dtype=np.int32)
    oz_buf = np.zeros(gw, dtype=np.float64)
    dz_buf = np.zeros(gw, dtype=np.float64)
    ovz_buf = np.zeros(gw, dtype=np.float64)

    sample_objs = [
        dict(
            box=(100 + i * 30, 150, 140 + i * 30, 350),
            cls=0 if i % 2 == 0 else 56,
            name="obj",
            mode="ok",
            X=-1.0 + i * 0.4,
            Z=2.0 + i * 0.3,
            r=2.5,
            meas=[(np.array([-1.0 + i * 0.4, 2.0 + i * 0.3]), meas_cov(2.5, "ok"))],
        )
        for i in range(8)
    ]

    results = {}
    backends = ["python", "rust"] if accel.has_rust() else ["python"]
    for bk in backends:
        kf = accel.KalmanTracker(backend=bk)
        grid = accel.OccupancyGrid2D(backend=bk)
        t0 = time.perf_counter()
        for k in range(iters):
            obst, drop, over, *_ = accel.depth_scan_and_cluster(
                disp, 0.5, -0.25, cam.w, cam.h, cam.cx, cam.cy, cam.f, cam.height, cam.pitch, cam.roll,
                5.0, os_buf, ds_buf, oz_buf, dz_buf,
                prev_over_streak=ovs_buf, prev_over_z=ovz_buf, backend=bk,
            )
            batch = [dict(o) for o in sample_objs]
            accel.resolve_occlusion(batch, backend=bk)
            tr = kf.step(batch, k * 0.033, ego=(0.01, 0.0, 0.02))
            grid.step(0.01, 0.0, 0.02, 0.033, 70.0, 1.15, obst, drop, over, tr)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / iters
        results[bk] = dt_ms
    return results


def record_session(source, out_dir, max_frames=300, use_sensor=True):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Khong mo duoc nguon ghi: {source}")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fps = min(max(float(fps), 5.0), 60.0)

    vid_path = os.path.join(out_dir, "video.mp4")
    meta_path = os.path.join(out_dir, "meta.jsonl")
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sensor = SensorReader(enabled=use_sensor)

    count = 0
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            while count < max_frames:
                if count > 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                writer.write(frame)
                rec = {
                    "frame": count,
                    "t": time.time(),
                    "pitch": sensor.tilt,
                    "roll": sensor.roll,
                }
                mf.write(json.dumps(rec) + "\n")
                count += 1
    finally:
        sensor.stop()
        writer.release()
        cap.release()
    print(f"Da ghi {count} khung hinh vao {out_dir} ({vid_path}, {meta_path})")


def run_video_eval(
    video_path,
    meta_list=None,
    max_frames=200,
    use_pose=True,
    use_depth=True,
    use_ego=True,
    backend="auto",
    depth_backend=None,
    label="BENCHMARK",
):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Khong mo duoc video: {video_path}")
    h, w = frame.shape[:2]

    engine = RadarEngine(
        w,
        h,
        HFOV_DEG,
        CAM_HEIGHT,
        PITCH_DEG,
        model_path=MODEL,
        pose_model_path=POSE_MODEL,
        imgsz=320,
        det_every=3,
        use_pose=use_pose,
        use_depth=use_depth,
        use_kalman=True,
        use_sensor=False,
        use_ego=use_ego,
        backend=backend,
        depth_backend=depth_backend,
    )

    latencies = []
    track_counts = []
    level_counts = {0: 0, 1: 0, 2: 0}
    moving_false_ttc = 0
    frames = 0

    try:
        for _ in range(3):
            engine.step(frame)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()

        t_start = time.perf_counter()
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if meta_list and frames < len(meta_list):
                m = meta_list[frames]
                if m.get("pitch") is not None:
                    engine.cam.pitch = math.radians(m["pitch"])
                if m.get("roll") is not None:
                    engine.cam.roll = math.radians(m["roll"])

            t0 = time.perf_counter()
            _, tracks, _, _, overall, _, _ = engine.step(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            track_counts.append(len(tracks))
            level_counts[overall] = level_counts.get(overall, 0) + 1
            if any(t.get("ttc") is not None and t.get("ttc") < 3.0 and t.get("cls") != 0 for t in tracks):
                moving_false_ttc += 1
            frames += 1
        total_s = time.perf_counter() - t_start
    finally:
        cap.release()
        engine.stop()

    arr = np.array(latencies) if latencies else np.array([0.0])
    tm = engine.timings
    print(f"\n=== {label} (ego={use_ego}) ===")
    print(f"Video          : {video_path} ({frames} frames)")
    print(f"Compute / AI   : {accel.get_backend().upper()} | {engine.ai_backend} | Depth={engine.depth.runtime_label if engine.depth else 'OFF'}")
    print(f"Avg FPS        : {frames / max(total_s, 1e-6):.2f} FPS")
    print(
        f"Latency (ms)   : p50={np.percentile(arr, 50):.2f} | p95={np.percentile(arr, 95):.2f} | "
        f"p99={np.percentile(arr, 99):.2f} | mean={arr.mean():.2f}"
    )
    print(
        f"Stage EMA (ms) : yolo={tm['yolo_ms']:.1f} | pose={tm['pose_ms']:.1f} | "
        f"depth={tm['depth_ms']:.1f} | ego={tm['ego_ms']:.1f} | core={tm['core_ms']:.2f}"
    )
    print(f"Avg Tracks     : {np.mean(track_counts):.2f} | Static-obj TTC alerts: {moving_false_ttc}/{max(frames,1)}")
    print(
        f"Alert Levels   : AN TOAN={level_counts[0]/max(frames,1)*100:.1f}% | "
        f"CHU Y={level_counts[1]/max(frames,1)*100:.1f}% | NGUY HIEM={level_counts[2]/max(frames,1)*100:.1f}%"
    )
    return {
        "fps": frames / max(total_s, 1e-6),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "levels": level_counts,
        "static_ttc": moving_false_ttc,
    }


def main():
    default_vid = "demo.mp4" if os.path.exists("demo.mp4") else "demo_street.mp4"
    ap = argparse.ArgumentParser(description="Benchmark, Record & Replay Webcam Radar")
    ap.add_argument("--video", type=str, default=default_vid)
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-ego", action="store_true")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "onnx", "openvino"])
    ap.add_argument("--depth-backend", type=str, default=None, choices=["auto", "pytorch", "onnx", "openvino"])
    ap.add_argument("--record", type=str, default=None, help="Thu muc ghi lai video + meta.jsonl")
    ap.add_argument("--replay", type=str, default=None, help="Thu muc replay de A/B test ego-motion")
    args = ap.parse_args()

    if args.record:
        src = args.cam if args.cam is not None else args.video
        record_session(src, args.record, max_frames=args.frames)
        return

    core_res = bench_core()
    print("=== CORE MICRO-BENCHMARK (Depth 168x128 + Overheads + Occlusion + Kalman + Grid/Steering) ===")
    for k, ms in core_res.items():
        print(f"  {k:8s}: {ms:.3f} ms / iter")
    if "rust" in core_res and "python" in core_res:
        print(f"  Speedup : {core_res['python'] / max(core_res['rust'], 1e-6):.1f}x")

    if args.replay:
        vid_path = os.path.join(args.replay, "video.mp4")
        meta_path = os.path.join(args.replay, "meta.jsonl")
        meta = []
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = [json.loads(line) for line in f if line.strip()]
        r_no_ego = run_video_eval(
            vid_path, meta, args.frames, not args.no_pose, not args.no_depth, False,
            args.backend, args.depth_backend, label="REPLAY A/B (NO EGO)"
        )
        r_ego = run_video_eval(
            vid_path, meta, args.frames, not args.no_pose, not args.no_depth, True,
            args.backend, args.depth_backend, label="REPLAY A/B (WITH EGO)"
        )
        print(
            f"\nA/B Delta static-obj TTC false alerts: {r_no_ego['static_ttc']} -> {r_ego['static_ttc']}"
        )
        return

    run_video_eval(
        args.video,
        None,
        max_frames=args.frames,
        use_pose=not args.no_pose,
        use_depth=not args.no_depth,
        use_ego=not args.no_ego,
        backend=args.backend,
        depth_backend=args.depth_backend,
        label="BAO CAO BENCHMARK PIPELINE",
    )


if __name__ == "__main__":
    main()
