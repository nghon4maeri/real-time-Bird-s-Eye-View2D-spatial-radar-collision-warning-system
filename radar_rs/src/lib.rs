mod depth_proc;
mod geom;
mod grid;
mod sensor;
mod tracker;

use pyo3::prelude::*;

#[pymodule]
fn radar_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(geom::resolve_occlusion_raw, m)?)?;
    m.add_function(wrap_pyfunction!(depth_proc::depth_fit_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(depth_proc::depth_scan_and_cluster, m)?)?;
    m.add_class::<tracker::KalmanTrackerRs>()?;
    m.add_class::<grid::OccupancyGridRs>()?;
    #[cfg(all(windows, feature = "win-sensor"))]
    m.add_class::<sensor::WinAccelerometer>()?;
    Ok(())
}
