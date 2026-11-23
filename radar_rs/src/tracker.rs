use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[derive(Clone)]
struct TrackRs {
    id: u64,
    cls: i32,
    x: [f64; 4],
    p: [f64; 16],
    t: f64,
    seen: f64,
}

/// Phan bo toi uu Hungarian (Jonker-Volgenant / Munkres O(n^3)) cho ma tran chi phi n x m.
fn hungarian(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n_rows = cost.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let n_cols = cost[0].len();
    if n_cols == 0 {
        return Vec::new();
    }
    let n = n_rows.max(n_cols);
    let mut a = vec![vec![0.0_f64; n + 1]; n + 1];
    for r in 0..n_rows {
        for c in 0..n_cols {
            a[r + 1][c + 1] = cost[r][c];
        }
    }

    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    let mut p = vec![0_usize; n + 1];
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0_usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0_usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = a[i0][j] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut res = Vec::new();
    for j in 1..=n {
        if p[j] > 0 && p[j] <= n_rows && j <= n_cols {
            res.push((p[j] - 1, j - 1));
        }
    }
    res
}

fn inv2x2(m: [f64; 4]) -> Option<[f64; 4]> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() <= 1e-12 {
        return None;
    }
    let idet = 1.0 / det;
    Some([m[3] * idet, -m[1] * idet, -m[2] * idet, m[0] * idet])
}

fn matmul4(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut c = [0.0_f64; 16];
    for i in 0..4 {
        for k in 0..4 {
            let aik = a[i * 4 + k];
            for j in 0..4 {
                c[i * 4 + j] += aik * b[k * 4 + j];
            }
        }
    }
    c
}

#[pyclass]
pub struct KalmanTrackerRs {
    tracks: Vec<TrackRs>,
    next_id: u64,
}

#[pymethods]
impl KalmanTrackerRs {
    #[new]
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 0,
        }
    }

    /// Cap nhat Kalman + Ego-motion compensation + Hungarian + Joseph form + adaptive TTL + TTC/CPA.
    /// Tra ve: ((id, obj_idx, cls, fresh), (X, Z, vx, vz, r, closing_speed, x_cpa, cpa, ttc_or_nan)).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn step_raw(
        &mut self,
        py: Python<'_>,
        obj_cls: PyReadonlyArray1<'_, i32>,
        meas_obj_idx: PyReadonlyArray1<'_, i32>,
        meas_z: PyReadonlyArray2<'_, f64>,
        meas_r: PyReadonlyArray2<'_, f64>,
        t: f64,
        det_every: i32,
        fps: f64,
        gate: f64,
        accel: f64,
        ttl_person: f64,
        ttl_static: f64,
        min_closing_speed: f64,
        ego_d_yaw: f64,
        ego_d_x: f64,
        ego_d_z: f64,
    ) -> PyResult<Vec<((u64, i32, i32, bool), (f64, f64, f64, f64, f64, f64, f64, f64, f64))>> {
        let cls_vec = obj_cls.as_slice()?.to_vec();
        let m_idx_vec = meas_obj_idx.as_slice()?.to_vec();
        let m_z_vec = meas_z.as_slice()?.to_vec();
        let m_r_vec = meas_r.as_slice()?.to_vec();

        let out = py.allow_threads(move || {
            let n_objs = cls_vec.len();

            // 1) Gop phep do (information form) cho tung vat
            let mut info_sum = vec![[0.0_f64; 4]; n_objs];
            let mut info_z = vec![[0.0_f64; 2]; n_objs];
            for k in 0..m_idx_vec.len() {
                let oid = m_idx_vec[k] as usize;
                if oid >= n_objs {
                    continue;
                }
                let rmat = [
                    m_r_vec[k * 4],
                    m_r_vec[k * 4 + 1],
                    m_r_vec[k * 4 + 2],
                    m_r_vec[k * 4 + 3],
                ];
                if let Some(ir) = inv2x2(rmat) {
                    let zx = m_z_vec[k * 2];
                    let zz = m_z_vec[k * 2 + 1];
                    for u in 0..4 {
                        info_sum[oid][u] += ir[u];
                    }
                    info_z[oid][0] += ir[0] * zx + ir[1] * zz;
                    info_z[oid][1] += ir[2] * zx + ir[3] * zz;
                }
            }
            let mut fused_z = vec![[0.0_f64; 2]; n_objs];
            let mut fused_r = vec![[0.0_f64; 4]; n_objs];
            for j in 0..n_objs {
                if let Some(rf) = inv2x2(info_sum[j]) {
                    fused_r[j] = rf;
                    fused_z[j][0] = rf[0] * info_z[j][0] + rf[1] * info_z[j][1];
                    fused_z[j][1] = rf[2] * info_z[j][0] + rf[3] * info_z[j][1];
                }
            }

            // 1.5) Ego-motion compensation: bien doi track cu theo (-d_yaw, -d_x, -d_z)
            if ego_d_yaw.abs() > 1e-6 || ego_d_x.abs() > 1e-6 || ego_d_z.abs() > 1e-6 {
                let cy = ego_d_yaw.cos();
                let sy = ego_d_yaw.sin();
                let mut rot = [0.0_f64; 16];
                rot[0] = cy;
                rot[1] = -sy;
                rot[4] = sy;
                rot[5] = cy;
                rot[10] = cy;
                rot[11] = -sy;
                rot[14] = sy;
                rot[15] = cy;
                let mut rot_t = [0.0_f64; 16];
                for rr in 0..4 {
                    for cc in 0..4 {
                        rot_t[rr * 4 + cc] = rot[cc * 4 + rr];
                    }
                }
                for tr in &mut self.tracks {
                    let dx1 = tr.x[0] - ego_d_x;
                    let dz1 = tr.x[1] - ego_d_z;
                    tr.x[0] = dx1 * cy - dz1 * sy;
                    tr.x[1] = dx1 * sy + dz1 * cy;
                    let vx0 = tr.x[2];
                    let vz0 = tr.x[3];
                    tr.x[2] = vx0 * cy - vz0 * sy;
                    tr.x[3] = vx0 * sy + vz0 * cy;
                    let rp = matmul4(&rot, &tr.p);
                    tr.p = matmul4(&rp, &rot_t);
                }
            }

            // 2) Predict
            let q_acc = accel * accel;
            for tr in &mut self.tracks {
                let dt = t - tr.t;
                tr.t = t;
                let mut fmat = [0.0_f64; 16];
                fmat[0] = 1.0;
                fmat[5] = 1.0;
                fmat[10] = 1.0;
                fmat[15] = 1.0;
                fmat[2] = dt;
                fmat[7] = dt;

                tr.x[0] += tr.x[2] * dt;
                tr.x[1] += tr.x[3] * dt;

                let fp = matmul4(&fmat, &tr.p);
                let mut f_t = [0.0_f64; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        f_t[r * 4 + c] = fmat[c * 4 + r];
                    }
                }
                let mut p_next = matmul4(&fp, &f_t);

                let dt2 = dt * dt;
                let dt3 = dt2 * dt * 0.5;
                let dt4 = dt2 * dt2 * 0.25;
                p_next[0] += dt4 * q_acc;
                p_next[2] += dt3 * q_acc;
                p_next[5] += dt4 * q_acc;
                p_next[7] += dt3 * q_acc;
                p_next[8] += dt3 * q_acc;
                p_next[10] += dt2 * q_acc;
                p_next[13] += dt3 * q_acc;
                p_next[15] += dt2 * q_acc;
                tr.p = p_next;
            }

            // 3) Mahalanobis + Hungarian assignment (khop dung cls)
            let n_tr = self.tracks.len();
            let mut matched_obj_for_tr: Vec<Option<usize>> = vec![None; n_tr];
            let mut used_obj = vec![false; n_objs];

            if n_tr > 0 && n_objs > 0 {
                const BIG: f64 = 1e9;
                let mut cost = vec![vec![BIG; n_objs]; n_tr];
                for i in 0..n_tr {
                    let tr = &self.tracks[i];
                    for j in 0..n_objs {
                        if tr.cls != cls_vec[j] {
                            continue;
                        }
                        let y0 = fused_z[j][0] - tr.x[0];
                        let y1 = fused_z[j][1] - tr.x[1];
                        let s = [
                            tr.p[0] + fused_r[j][0],
                            tr.p[1] + fused_r[j][1],
                            tr.p[4] + fused_r[j][2],
                            tr.p[5] + fused_r[j][3],
                        ];
                        if let Some(is) = inv2x2(s) {
                            let sy0 = is[0] * y0 + is[1] * y1;
                            let sy1 = is[2] * y0 + is[3] * y1;
                            let d2 = y0 * sy0 + y1 * sy1;
                            if d2 < gate && d2.is_finite() {
                                cost[i][j] = d2;
                            }
                        }
                    }
                }
                for (i, j) in hungarian(&cost) {
                    if cost[i][j] < gate {
                        matched_obj_for_tr[i] = Some(j);
                        used_obj[j] = true;
                    }
                }
            }

            // 4) Joseph-form update
            let mut tr_obj_idx = vec![-1_i32; n_tr];
            for i in 0..n_tr {
                if let Some(j) = matched_obj_for_tr[i] {
                    let tr = &mut self.tracks[i];
                    let y0 = fused_z[j][0] - tr.x[0];
                    let y1 = fused_z[j][1] - tr.x[1];
                    let r = fused_r[j];
                    let s = [
                        tr.p[0] + r[0],
                        tr.p[1] + r[1],
                        tr.p[4] + r[2],
                        tr.p[5] + r[3],
                    ];
                    if let Some(is) = inv2x2(s) {
                        // K = P[:, :2] @ S^-1 (4x2)
                        let mut kmat = [0.0_f64; 8];
                        for row in 0..4 {
                            let p0 = tr.p[row * 4];
                            let p1 = tr.p[row * 4 + 1];
                            kmat[row * 2] = p0 * is[0] + p1 * is[2];
                            kmat[row * 2 + 1] = p0 * is[1] + p1 * is[3];
                        }
                        for row in 0..4 {
                            tr.x[row] += kmat[row * 2] * y0 + kmat[row * 2 + 1] * y1;
                        }
                        // I_KH = I4 - K * H (H = [I2, 0])
                        let mut ikh = [0.0_f64; 16];
                        ikh[0] = 1.0 - kmat[0];
                        ikh[1] = -kmat[1];
                        ikh[4] = -kmat[2];
                        ikh[5] = 1.0 - kmat[3];
                        ikh[8] = -kmat[4];
                        ikh[9] = -kmat[5];
                        ikh[10] = 1.0;
                        ikh[12] = -kmat[6];
                        ikh[13] = -kmat[7];
                        ikh[15] = 1.0;

                        let ikhp = matmul4(&ikh, &tr.p);
                        let mut ikh_t = [0.0_f64; 16];
                        for rr in 0..4 {
                            for cc in 0..4 {
                                ikh_t[rr * 4 + cc] = ikh[cc * 4 + rr];
                            }
                        }
                        let mut p_upd = matmul4(&ikhp, &ikh_t);

                        // + K R K^T
                        for r_i in 0..4 {
                            let kr0 = kmat[r_i * 2] * r[0] + kmat[r_i * 2 + 1] * r[2];
                            let kr1 = kmat[r_i * 2] * r[1] + kmat[r_i * 2 + 1] * r[3];
                            for c_i in 0..4 {
                                p_upd[r_i * 4 + c_i] +=
                                    kr0 * kmat[c_i * 2] + kr1 * kmat[c_i * 2 + 1];
                            }
                        }
                        tr.p = p_upd;
                    }
                    tr.seen = t;
                    tr_obj_idx[i] = j as i32;
                }
            }

            // 5) Tao track moi cho vat chua ghep
            for j in 0..n_objs {
                if !used_obj[j] {
                    let mut p0 = [0.0_f64; 16];
                    p0[0] = fused_r[j][0];
                    p0[1] = fused_r[j][1];
                    p0[4] = fused_r[j][2];
                    p0[5] = fused_r[j][3];
                    p0[10] = 1.0;
                    p0[15] = 1.0;
                    self.tracks.push(TrackRs {
                        id: self.next_id,
                        cls: cls_vec[j],
                        x: [fused_z[j][0], fused_z[j][1], 0.0, 0.0],
                        p: p0,
                        t,
                        seen: t,
                    });
                    self.next_id += 1;
                    tr_obj_idx.push(j as i32);
                }
            }

            // 6) Xoa track het han theo TTL tach nguoi vs vat tinh thich ung det_every / fps
            let adaptive_static_ttl =
                ttl_static.max((det_every.max(1) as f64 / fps.max(1.0)) * 2.5);
            let mut keep_tracks = Vec::with_capacity(self.tracks.len());
            let mut keep_obj_idx = Vec::with_capacity(self.tracks.len());
            for (idx, tr) in self.tracks.drain(..).enumerate() {
                let ttl = if tr.cls == 0 {
                    ttl_person
                } else {
                    adaptive_static_ttl
                };
                if t - tr.seen < ttl {
                    keep_tracks.push(tr);
                    keep_obj_idx.push(tr_obj_idx[idx]);
                }
            }
            self.tracks = keep_tracks;

            // 7) Xuat ket qua + tinh TTC/CPA
            let mut results = Vec::with_capacity(self.tracks.len());
            for (idx, tr) in self.tracks.iter().enumerate() {
                let xx = tr.x[0];
                let zz = tr.x[1];
                if zz <= 0.05 {
                    continue;
                }
                let vx = tr.x[2];
                let vz = tr.x[3];
                let r = xx.hypot(zz);
                let fresh = (tr.seen - t).abs() < 1e-12;
                let closing_speed = if r >= 1e-6 {
                    -(xx * vx + zz * vz) / r
                } else {
                    0.0
                };
                let v2 = vx * vx + vz * vz;
                let (x_cpa, cpa_dist, ttc) = if closing_speed > 0.0 && v2 > 1e-6 {
                    let t_cpa = -(xx * vx + zz * vz) / v2;
                    let xc = xx + vx * t_cpa;
                    let zc = zz + vz * t_cpa;
                    let cd = xc.hypot(zc);
                    let tv = if t_cpa > 0.0 && closing_speed > min_closing_speed {
                        t_cpa
                    } else {
                        f64::NAN
                    };
                    (xc, cd, tv)
                } else {
                    (xx, r, f64::NAN)
                };
                results.push((
                    (tr.id, keep_obj_idx[idx], tr.cls, fresh),
                    (xx, zz, vx, vz, r, closing_speed, x_cpa, cpa_dist, ttc),
                ));
            }
            results
        });

        Ok(out)
    }
}
