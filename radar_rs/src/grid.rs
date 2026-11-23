use pyo3::prelude::*;

const GRID_N: usize = 60;
const CELL_M: f64 = 0.10;
const HALF_X: f64 = 3.0; // X in [-3.0, +3.0], Z in [0.0, 6.0]

#[pyclass]
pub struct OccupancyGridRs {
    occ: Vec<f32>,
    kind: Vec<u8>,
}

impl OccupancyGridRs {
    #[inline]
    fn to_cell(x: f64, z: f64) -> Option<(usize, usize)> {
        if !(x >= -HALF_X && x < HALF_X && z >= 0.0 && z < (GRID_N as f64) * CELL_M) {
            return None;
        }
        let c = ((x + HALF_X) / CELL_M).floor() as isize;
        let r = (z / CELL_M).floor() as isize;
        if c >= 0 && c < GRID_N as isize && r >= 0 && r < GRID_N as isize {
            Some((r as usize, c as usize))
        } else {
            None
        }
    }

    #[inline]
    fn add_hit(&mut self, x: f64, z: f64, delta: f32, k: u8) {
        if let Some((r, c)) = Self::to_cell(x, z) {
            let idx = r * GRID_N + c;
            self.occ[idx] = (self.occ[idx] + delta).min(1.0);
            if k >= self.kind[idx] {
                self.kind[idx] = k;
            }
        }
    }
}

#[pymethods]
impl OccupancyGridRs {
    #[new]
    pub fn new() -> Self {
        Self {
            occ: vec![0.0; GRID_N * GRID_N],
            kind: vec![0; GRID_N * GRID_N],
        }
    }

    pub fn reset(&mut self) {
        self.occ.fill(0.0);
        self.kind.fill(0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        py: Python<'_>,
        d_yaw: f64,
        d_x: f64,
        d_z: f64,
        dt: f64,
        hfov_deg: f64,
        min_visible: f64,
        obstacles: Vec<(f64, f64)>,
        dropoffs: Vec<(f64, f64)>,
        overheads: Vec<(f64, f64)>,
        tracks: Vec<(f64, f64)>,
        decay_tau: f64,
    ) -> PyResult<(String, f64, f64, Vec<(f64, f64, f32, u8)>)> {
        let out = py.allow_threads(move || {
            let dt_clamped = dt.clamp(0.005, 0.5);
            let decay_blind = (-dt_clamped / decay_tau.max(0.2)).exp() as f32;
            let decay_vis = (-dt_clamped / 0.55).exp() as f32;
            let half_fov_rad = (hfov_deg * 0.5).to_radians();

            let cy = d_yaw.cos();
            let sy = d_yaw.sin();

            let mut next_occ = vec![0.0_f32; GRID_N * GRID_N];
            let mut next_kind = vec![0_u8; GRID_N * GRID_N];

            // 1) Ego-motion warp (nghich dao: o moi -> o cu) + adaptive temporal decay
            for r in 0..GRID_N {
                let zn = (r as f64 + 0.5) * CELL_M;
                for c in 0..GRID_N {
                    let xn = (c as f64 + 0.5) * CELL_M - HALF_X;
                    // Nghich dao phep bien doi: X' = (X-dx)*cy - (Z-dz)*sy, Z' = (X-dx)*sy + (Z-dz)*cy
                    let x_old = xn * cy + zn * sy + d_x;
                    let z_old = -xn * sy + zn * cy + d_z;
                    if let Some((ro, co)) = Self::to_cell(x_old, z_old) {
                        let o_idx = ro * GRID_N + co;
                        let val = self.occ[o_idx];
                        if val > 0.02 {
                            let dist = xn.hypot(zn);
                            let ang = xn.atan2(zn.max(1e-3)).abs();
                            let in_vis = dist >= min_visible && ang <= half_fov_rad;
                            let decayed = val * if in_vis { decay_vis } else { decay_blind };
                            if decayed >= 0.04 {
                                let n_idx = r * GRID_N + c;
                                next_occ[n_idx] = decayed;
                                next_kind[n_idx] = self.kind[o_idx];
                            }
                        }
                    }
                }
            }
            self.occ = next_occ;
            self.kind = next_kind;

            // 2) Tich hop quan sat moi
            for (x, z) in obstacles {
                self.add_hit(x, z, 0.55, 1);
            }
            for (x, z) in tracks {
                for ox in [-0.15, 0.0, 0.15] {
                    for oz in [-0.15, 0.0, 0.15] {
                        self.add_hit(x + ox, z + oz, 0.65, 1);
                    }
                }
            }
            for (x, z) in dropoffs {
                self.add_hit(x, z, 0.75, 2);
            }
            for (x, z) in overheads {
                self.add_hit(x, z, 0.70, 3);
            }

            // 3) Tinh guidance (VFH+ nhe)
            let (action, target_deg, clearance_m) =
                self.compute_steering_inner(hfov_deg, 0.65, 5.0);

            // 4) Xuat cac o dang chiem dung de ve
            let mut cells = Vec::new();
            for r in 0..GRID_N {
                let z = (r as f64 + 0.5) * CELL_M;
                for c in 0..GRID_N {
                    let idx = r * GRID_N + c;
                    let o = self.occ[idx];
                    if o >= 0.30 {
                        let x = (c as f64 + 0.5) * CELL_M - HALF_X;
                        cells.push((x, z, o, self.kind[idx]));
                    }
                }
            }
            (action, target_deg, clearance_m, cells)
        });
        Ok(out)
    }

    pub fn compute_steering(
        &self,
        hfov_deg: f64,
        passage_width: f64,
        max_range: f64,
    ) -> (String, f64, f64) {
        self.compute_steering_inner(hfov_deg, passage_width, max_range)
    }
}

impl OccupancyGridRs {
    fn compute_steering_inner(
        &self,
        hfov_deg: f64,
        passage_width: f64,
        max_range: f64,
    ) -> (String, f64, f64) {
        let half_fov = (hfov_deg * 0.5).clamp(15.0, 60.0);
        let half_w = passage_width * 0.5;
        let lat_samples = [-half_w, -half_w * 0.5, 0.0, half_w * 0.5, half_w];

        let mut best_deg = 0.0_f64;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_clear = 0.0_f64;
        let mut straight_clear = max_range;

        let n_steps = ((half_fov * 2.0) / 2.0).round() as i32;
        for k in 0..=n_steps {
            let deg = -half_fov + (k as f64) * 2.0;
            let rad = deg.to_radians();
            let sin_t = rad.sin();
            let cos_t = rad.cos();

            let mut clear = max_range;
            let mut s = 0.25_f64;
            while s <= max_range {
                let cx = s * sin_t;
                let cz = s * cos_t;
                let mut hit = false;
                for &lw in &lat_samples {
                    let px = cx + lw * cos_t;
                    let pz = cz - lw * sin_t;
                    if let Some((r, c)) = Self::to_cell(px, pz) {
                        if self.occ[r * GRID_N + c] >= 0.35 {
                            hit = true;
                            break;
                        }
                    }
                }
                if hit {
                    clear = s;
                    break;
                }
                s += CELL_M;
            }

            if deg.abs() < 1.1 {
                straight_clear = clear;
            }
            let score = clear - 0.015 * deg.abs();
            if score > best_score {
                best_score = score;
                best_deg = deg;
                best_clear = clear;
            }
        }

        let (action, out_deg, out_clear) = if straight_clear >= 2.2 {
            ("straight".to_string(), 0.0, straight_clear)
        } else if best_clear < 1.0 {
            ("stop".to_string(), best_deg, best_clear)
        } else if best_clear < 1.5 {
            ("caution".to_string(), best_deg, best_clear)
        } else if best_deg < -4.0 {
            ("left".to_string(), best_deg, best_clear)
        } else if best_deg > 4.0 {
            ("right".to_string(), best_deg, best_clear)
        } else {
            ("straight".to_string(), 0.0, straight_clear)
        };
        (action, out_deg, out_clear)
    }
}
