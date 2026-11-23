use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

fn median_f64(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        0.5 * (vals[n / 2 - 1] + vals[n / 2])
    }
}

fn solve_wls_2x2(d: &[f64], iz: &[f64], w: &[f64]) -> Option<(f64, f64)> {
    let mut s00 = 0.0_f64;
    let mut s01 = 0.0_f64;
    let mut s11 = 0.0_f64;
    let mut t0 = 0.0_f64;
    let mut t1 = 0.0_f64;
    for i in 0..d.len() {
        let wi = w[i];
        let di = d[i];
        let yi = iz[i];
        s00 += wi * di * di;
        s01 += wi * di;
        s11 += wi;
        t0 += wi * di * yi;
        t1 += wi * yi;
    }
    let det = s00 * s11 - s01 * s01;
    if det.abs() <= 1e-12 {
        return None;
    }
    let a = (t0 * s11 - t1 * s01) / det;
    let b = (s00 * t1 - s01 * t0) / det;
    if a.is_finite() && b.is_finite() {
        Some((a, b))
    } else {
        None
    }
}

/// Mask bounding box khoi san + hoi quy RANSAC / Huber IRLS tim (a, b, n_anchor).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn depth_fit_ransac(
    py: Python<'_>,
    disp: PyReadonlyArray2<'_, f32>,
    anchor_boxes: PyReadonlyArray2<'_, f32>,
    anchor_zs: PyReadonlyArray1<'_, f32>,
    mask_boxes: PyReadonlyArray2<'_, f32>,
    cam_w: f64,
    cam_h: f64,
    cx: f64,
    cy: f64,
    f: f64,
    cam_height: f64,
    pitch: f64,
    roll: f64,
    min_visible: f64,
    prev_fit: Option<(f64, f64)>,
) -> PyResult<Option<(f64, f64, usize)>> {
    let shape = disp.shape();
    let (gh, gw) = (shape[0], shape[1]);
    let disp_vec = disp.as_slice()?.to_vec();
    let anc_boxes_vec = anchor_boxes.as_slice()?.to_vec();
    let anc_zs_vec = anchor_zs.as_slice()?.to_vec();
    let mask_boxes_vec = mask_boxes.as_slice()?.to_vec();
    let n_anc = anc_zs_vec.len();
    let n_mask = mask_boxes.shape()[0];

    let res = py.allow_threads(move || {
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cr = roll.cos();
        let sr = roll.sin();
        let sx = gw as f64 / cam_w;
        let sy = gh as f64 / cam_h;

        let mut d_pts: Vec<f64> = Vec::with_capacity(n_anc + 1);
        let mut iz_pts: Vec<f64> = Vec::with_capacity(n_anc + 1);
        let mut w_pts: Vec<f64> = Vec::with_capacity(n_anc + 1);

        for i in 0..n_anc {
            let x1 = anc_boxes_vec[i * 4] as f64;
            let y1 = anc_boxes_vec[i * 4 + 1] as f64;
            let x2 = anc_boxes_vec[i * 4 + 2] as f64;
            let y2 = anc_boxes_vec[i * 4 + 3] as f64;
            let z = anc_zs_vec[i] as f64;
            if z <= 0.05 || !z.is_finite() {
                continue;
            }
            let bw = x2 - x1;
            let bh = y2 - y1;
            let r0 = (((y1 + 0.3 * bh) * sy) as isize).clamp(0, gh as isize - 1) as usize;
            let r1 = (((y1 + 0.7 * bh) * sy) as isize)
                .max(r0 as isize + 1)
                .min(gh as isize) as usize;
            let c0 = (((x1 + 0.3 * bw) * sx) as isize).clamp(0, gw as isize - 1) as usize;
            let c1 = (((x1 + 0.7 * bw) * sx) as isize)
                .max(c0 as isize + 1)
                .min(gw as isize) as usize;

            let u_mean = 0.5 * (c0 + c1) as f64 * cam_w / gw as f64;
            let v_mean = 0.5 * (r0 + r1) as f64 * cam_h / gh as f64;
            let yr = ((u_mean - cx) / f) * sr + ((v_mean - cy) / f) * cr;
            let ray = cp - yr * sp;
            if ray > 0.05 {
                let mut cell_vals = Vec::with_capacity((r1 - r0) * (c1 - c0));
                for r in r0..r1 {
                    let row_off = r * gw;
                    for c in c0..c1 {
                        let v = disp_vec[row_off + c] as f64;
                        if v.is_finite() {
                            cell_vals.push(v);
                        }
                    }
                }
                if !cell_vals.is_empty() {
                    let dm = median_f64(&mut cell_vals);
                    d_pts.push(dm);
                    iz_pts.push(ray / z);
                    w_pts.push(1.0);
                }
            }
        }

        // Lay mau san o mep duoi, loai tru tat ca bounding box trong mask_boxes
        if min_visible < 4.0 && gh >= 4 {
            let c_start = gw / 4;
            let c_end = (3 * gw) / 4;
            let cand_rows = [gh - 2, gh - 3, gh - 4];
            for &row in &cand_rows {
                let v_r = (row as f64 + 0.5) * cam_h / gh as f64;
                let mut d_floor = Vec::new();
                let mut iz_floor = Vec::new();
                for c in c_start..c_end {
                    let u_c = (c as f64 + 0.5) * cam_w / gw as f64;
                    let mut masked = false;
                    for m in 0..n_mask {
                        let mx1 = mask_boxes_vec[m * 4] as f64 - 2.0;
                        let my1 = mask_boxes_vec[m * 4 + 1] as f64 - 2.0;
                        let mx2 = mask_boxes_vec[m * 4 + 2] as f64 + 2.0;
                        let my2 = mask_boxes_vec[m * 4 + 3] as f64 + 2.0;
                        if u_c >= mx1 && u_c <= mx2 && v_r >= my1 && v_r <= my2 {
                            masked = true;
                            break;
                        }
                    }
                    if masked {
                        continue;
                    }
                    let yr = ((u_c - cx) / f) * sr + ((v_r - cy) / f) * cr;
                    let down = yr * cp + sp;
                    let ray = cp - yr * sp;
                    if down > 1e-3 && ray > 0.05 * down {
                        let val = disp_vec[row * gw + c] as f64;
                        if val.is_finite() {
                            d_floor.push(val);
                            iz_floor.push(down / cam_height);
                        }
                    }
                }
                if d_floor.len() >= 3 {
                    d_pts.push(median_f64(&mut d_floor));
                    iz_pts.push(median_f64(&mut iz_floor));
                    w_pts.push(0.5);
                    break;
                }
            }
        }

        let n = d_pts.len();
        if n == 0 {
            return None;
        }

        let d_min = d_pts.iter().copied().fold(f64::INFINITY, f64::min);
        let d_max = d_pts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let d_abs_max = d_pts.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let ptp_ok = (d_max - d_min) > 1e-3 * d_abs_max.max(1e-6);

        let (mut a, mut b) = if n >= 3 && ptp_ok {
            let mut abs_iz: Vec<f64> = iz_pts.iter().map(|v| v.abs()).collect();
            let thresh = (0.18 * median_f64(&mut abs_iz)).max(0.04);

            let mut best_score = f64::NEG_INFINITY;
            let mut best_pair: Option<(f64, f64)> = None;

            for i in 0..n {
                for j in (i + 1)..n {
                    let dd = d_pts[j] - d_pts[i];
                    if dd.abs() <= 1e-6 {
                        continue;
                    }
                    let ac = (iz_pts[j] - iz_pts[i]) / dd;
                    if ac <= 0.0 || !ac.is_finite() {
                        continue;
                    }
                    let bc = iz_pts[i] - ac * d_pts[i];
                    let mut inlier_w = 0.0;
                    let mut loss = 0.0;
                    for k in 0..n {
                        let err = (ac * d_pts[k] + bc - iz_pts[k]).abs();
                        if err <= thresh {
                            inlier_w += w_pts[k];
                        }
                        let c_err = err.min(thresh);
                        loss += w_pts[k] * c_err * c_err;
                    }
                    let score = inlier_w - 0.01 * loss;
                    if score > best_score {
                        best_score = score;
                        best_pair = Some((ac, bc));
                    }
                }
            }

            if let Some((ra, rb)) = best_pair {
                // Khoi tao WLS tren cac diem inlier
                let (mut id, mut iiz, mut iw) = (Vec::new(), Vec::new(), Vec::new());
                for k in 0..n {
                    if (ra * d_pts[k] + rb - iz_pts[k]).abs() <= thresh {
                        id.push(d_pts[k]);
                        iiz.push(iz_pts[k]);
                        iw.push(w_pts[k]);
                    }
                }
                let (mut cur_a, mut cur_b) = if id.len() >= 2 {
                    solve_wls_2x2(&id, &iiz, &iw).unwrap_or((ra, rb))
                } else {
                    (ra, rb)
                };

                // Huber IRLS 5 vong lap
                let mut hw = vec![0.0_f64; n];
                for _ in 0..5 {
                    for k in 0..n {
                        let res = (cur_a * d_pts[k] + cur_b - iz_pts[k]).abs();
                        hw[k] = if res <= thresh {
                            w_pts[k]
                        } else if res <= 3.0 * thresh {
                            w_pts[k] * (thresh / res)
                        } else {
                            0.0
                        };
                    }
                    if let Some((na, nb)) = solve_wls_2x2(&d_pts, &iz_pts, &hw) {
                        if na > 0.0 {
                            cur_a = na;
                            cur_b = nb;
                        }
                    }
                }
                (cur_a, cur_b)
            } else {
                solve_wls_2x2(&d_pts, &iz_pts, &w_pts)?
            }
        } else if n >= 2 && ptp_ok {
            solve_wls_2x2(&d_pts, &iz_pts, &w_pts)?
        } else {
            let b_fix = prev_fit.map(|p| p.1).unwrap_or(0.0);
            let num: f64 = (0..n)
                .map(|k| w_pts[k] * (iz_pts[k] - b_fix) * d_pts[k])
                .sum();
            let den: f64 = (0..n).map(|k| w_pts[k] * d_pts[k] * d_pts[k]).sum();
            (num / den.max(1e-9), b_fix)
        };

        if a <= 0.0 || !a.is_finite() || !b.is_finite() {
            return None;
        }
        if let Some((pa, pb)) = prev_fit {
            const FIT_EMA: f64 = 0.3;
            a = pa + FIT_EMA * (a - pa);
            b = pb + FIT_EMA * (b - pb);
        }
        Some((a, b, n))
    });

    Ok(res)
}

#[allow(clippy::too_many_arguments)]
fn cluster_and_confirm(
    valid: &[bool],
    xs: &[f64],
    zs: &[f64],
    prev_streak: &[i32],
    prev_z: &[f64],
    min_cluster_cols: usize,
    min_confirm: i32,
) -> (Vec<(f64, f64)>, Vec<i32>, Vec<f64>) {
    let gw = valid.len();
    let mut clustered = vec![false; gw];
    let mut c = 0;
    while c < gw {
        if !valid[c] {
            c += 1;
            continue;
        }
        let start = c;
        c += 1;
        while c < gw && valid[c] {
            let dz_tol = (0.18 * zs[c].min(zs[c - 1])).max(0.30);
            if (zs[c] - zs[c - 1]).abs() <= dz_tol {
                c += 1;
            } else {
                break;
            }
        }
        if c - start >= min_cluster_cols {
            for k in start..c {
                clustered[k] = true;
            }
        }
    }

    let mut streak = vec![0_i32; gw];
    let mut out_z = vec![0.0_f64; gw];
    let mut pts = Vec::new();
    for col in 0..gw {
        if !clustered[col] {
            continue;
        }
        let zc = zs[col];
        out_z[col] = zc;
        let z_tol = (0.25 * zc).max(0.35);
        let c_lo = col.saturating_sub(1);
        let c_hi = (col + 1).min(gw - 1);
        let mut best_prev = 0;
        for nb in c_lo..=c_hi {
            if prev_streak[nb] > 0 && prev_z[nb] > 0.0 && (zc - prev_z[nb]).abs() <= z_tol {
                if prev_streak[nb] > best_prev {
                    best_prev = prev_streak[nb];
                }
            }
        }
        let st = (best_prev + 1).min(1000);
        streak[col] = st;
        if st >= min_confirm {
            pts.push((xs[col], zc));
        }
    }
    (pts, streak, out_z)
}

/// Quet luoi depth voi bu (pitch, roll), loc cum nhieu, xac nhan da khung,
/// phat hien bac thang (dropoffs) va vat can ngang dau (overheads) + sai so san.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn depth_scan_and_cluster(
    py: Python<'_>,
    disp: PyReadonlyArray2<'_, f32>,
    a: f64,
    b: f64,
    cam_w: f64,
    cam_h: f64,
    cx: f64,
    cy: f64,
    f: f64,
    height: f64,
    pitch: f64,
    roll: f64,
    max_r: f64,
    prev_obst_streak: PyReadonlyArray1<'_, i32>,
    prev_drop_streak: PyReadonlyArray1<'_, i32>,
    prev_over_streak: PyReadonlyArray1<'_, i32>,
    prev_obst_z: PyReadonlyArray1<'_, f64>,
    prev_drop_z: PyReadonlyArray1<'_, f64>,
    prev_over_z: PyReadonlyArray1<'_, f64>,
    obst_h_min: f64,
    obst_h_max: f64,
    drop_h_thresh: f64,
    over_h_min: f64,
    over_h_max: f64,
    min_cluster_cols: usize,
    min_confirm: i32,
) -> PyResult<(
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
)> {
    let shape = disp.shape();
    let (gh, gw) = (shape[0], shape[1]);
    let disp_vec = disp.as_slice()?.to_vec();
    let pos_vec = prev_obst_streak.as_slice()?.to_vec();
    let pds_vec = prev_drop_streak.as_slice()?.to_vec();
    let pov_vec = prev_over_streak.as_slice()?.to_vec();
    let poz_vec = prev_obst_z.as_slice()?.to_vec();
    let pdz_vec = prev_drop_z.as_slice()?.to_vec();
    let povz_vec = prev_over_z.as_slice()?.to_vec();

    let res = py.allow_threads(move || {
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cr = roll.cos();
        let sr = roll.sin();
        let drop_max_z = max_r.min(3.5);
        let over_max_z = max_r.min(3.5);

        let mut col_x = vec![0.0_f64; gw];
        let mut col_z = vec![0.0_f64; gw];
        let mut col_r = vec![f64::INFINITY; gw];
        let mut col_valid = vec![false; gw];

        let mut drop_x = vec![0.0_f64; gw];
        let mut drop_z = vec![0.0_f64; gw];
        let mut drop_r = vec![f64::INFINITY; gw];
        let mut drop_valid = vec![false; gw];

        let mut over_x = vec![0.0_f64; gw];
        let mut over_z = vec![0.0_f64; gw];
        let mut over_r = vec![f64::INFINITY; gw];
        let mut over_valid = vec![false; gw];
        let mut low_z = vec![f64::INFINITY; gw];

        let mut floor_errs: Vec<f64> = Vec::new();

        let mut x_norm = Vec::with_capacity(gw);
        for c in 0..gw {
            let u = (c as f64 + 0.5) * cam_w / gw as f64;
            x_norm.push((u - cx) / f);
        }

        for r in 0..gh {
            let v = (r as f64 + 0.5) * cam_h / gh as f64;
            let y = (v - cy) / f;
            let row_off = r * gw;
            for c in 0..gw {
                let inv = a * (disp_vec[row_off + c] as f64) + b;
                if inv <= 1e-3 || !inv.is_finite() {
                    continue;
                }
                let zc = 1.0 / inv;
                let x = x_norm[c];
                let xr = x * cr - y * sr;
                let yr = x * sr + y * cr;
                let down_dir = yr * cp + sp;
                let hgt = height - zc * down_dir;
                let xx = zc * xr;
                let zz = zc * (cp - yr * sp);
                let dist = xx.hypot(zz);

                // Thu thap sai so san gan de uoc luong auto-pitch
                if r >= (3 * gh) / 4
                    && c >= gw / 4
                    && c < (3 * gw) / 4
                    && down_dir > 0.1
                    && zz > 0.6
                    && zz < 3.2
                    && hgt.abs() < 0.20
                {
                    floor_errs.push(hgt);
                }

                if hgt > obst_h_min && hgt <= 1.20 && zz > 0.2 && dist < max_r {
                    if zz < low_z[c] {
                        low_z[c] = zz;
                    }
                }

                if hgt >= over_h_min && hgt < over_h_max && zz > 0.25 && dist < over_max_z {
                    if dist < over_r[c] {
                        over_r[c] = dist;
                        over_x[c] = xx;
                        over_z[c] = zz;
                        over_valid[c] = true;
                    }
                }

                if hgt > obst_h_min && hgt < obst_h_max && zz > 0.2 && dist < max_r {
                    if dist < col_r[c] {
                        col_r[c] = dist;
                        col_x[c] = xx;
                        col_z[c] = zz;
                        col_valid[c] = true;
                    }
                } else if hgt > -3.0
                    && hgt < drop_h_thresh
                    && down_dir > 0.05
                    && zz > 0.4
                    && zz < drop_max_z
                    && dist < drop_r[c]
                {
                    drop_r[c] = dist;
                    drop_x[c] = xx;
                    drop_z[c] = zz;
                    drop_valid[c] = true;
                }
            }
        }

        // Overhead chi giu neu ben duoi KHONG co phan chan dung cung do sau (tranh nham dau nguoi/tuong)
        for c in 0..gw {
            if over_valid[c] && low_z[c].is_finite() && (low_z[c] - over_z[c]).abs() < 0.45 {
                over_valid[c] = false;
            }
            if drop_valid[c] && col_valid[c] && col_r[c] <= drop_r[c] {
                drop_valid[c] = false;
            }
        }

        let (obst_pts, new_os, new_oz) = cluster_and_confirm(
            &col_valid,
            &col_x,
            &col_z,
            &pos_vec,
            &poz_vec,
            min_cluster_cols,
            min_confirm,
        );
        let (drop_pts, new_ds, new_dz) = cluster_and_confirm(
            &drop_valid,
            &drop_x,
            &drop_z,
            &pds_vec,
            &pdz_vec,
            min_cluster_cols,
            min_confirm,
        );
        let (over_pts, new_ovs, new_ovz) = cluster_and_confirm(
            &over_valid,
            &over_x,
            &over_z,
            &pov_vec,
            &povz_vec,
            min_cluster_cols,
            min_confirm,
        );
        let floor_err = if floor_errs.len() >= 8 {
            median_f64(&mut floor_errs)
        } else {
            f64::NAN
        };

        (
            obst_pts,
            drop_pts,
            over_pts,
            new_os,
            new_ds,
            new_ovs,
            new_oz,
            new_dz,
            new_ovz,
            floor_err,
        )
    });

    Ok(res)
}
