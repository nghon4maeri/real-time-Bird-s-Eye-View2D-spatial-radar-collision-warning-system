use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Sap xep theo Z tang dan va xu ly che khuat nhieu tang (chuoi >= 3 vat).
#[pyfunction]
pub fn resolve_occlusion_raw(
    py: Python<'_>,
    boxes: PyReadonlyArray2<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    zs: PyReadonlyArray1<'_, f64>,
    can_occlude: PyReadonlyArray1<'_, bool>,
    occluded_gap: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<bool>)> {
    let shape = boxes.shape();
    let n = shape[0];
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let boxes_vec = boxes.as_slice()?.to_vec();
    let mut x_vec = xs.as_slice()?.to_vec();
    let mut z_vec = zs.as_slice()?.to_vec();
    let occ_vec = can_occlude.as_slice()?.to_vec();

    let che_vec = py.allow_threads(move || {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| z_vec[i].total_cmp(&z_vec[j]));

        let mut che = vec![false; n];
        for pos_a in 0..n {
            let a = order[pos_a];
            if !occ_vec[a] {
                continue;
            }
            let ax1 = boxes_vec[a * 4];
            let ax2 = boxes_vec[a * 4 + 2];
            let ay2 = boxes_vec[a * 4 + 3];
            let au = 0.5 * (ax1 + ax2);

            for &b in &order {
                if b == a {
                    continue;
                }
                if z_vec[b] >= z_vec[a] {
                    continue;
                }
                let bx1 = boxes_vec[b * 4];
                let by1 = boxes_vec[b * 4 + 1];
                let bx2 = boxes_vec[b * 4 + 2];
                let by2 = boxes_vec[b * 4 + 3];

                if bx1 <= au && au <= bx2 && by1 <= ay2 && ay2 < by2 - 8.0 {
                    let z_new = z_vec[b] + occluded_gap;
                    if z_new < z_vec[a] {
                        if z_vec[a].abs() > 1e-9 {
                            x_vec[a] *= z_new / z_vec[a];
                        }
                        z_vec[a] = z_new;
                        che[a] = true;
                    }
                }
            }
        }
        (x_vec, z_vec, che)
    });

    Ok(che_vec)
}
