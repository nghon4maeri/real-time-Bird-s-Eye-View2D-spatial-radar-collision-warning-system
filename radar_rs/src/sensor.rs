#[cfg(all(windows, feature = "win-sensor"))]
mod win_impl {
    use pyo3::prelude::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    use std::thread;
    use std::time::Duration;
    use windows::Devices::Sensors::Accelerometer;

    #[pyclass]
    pub struct WinAccelerometer {
        state: Arc<Mutex<(Option<f64>, f64, f64)>>,
        running: Arc<AtomicBool>,
    }

    #[pymethods]
    impl WinAccelerometer {
        #[new]
        pub fn new() -> PyResult<Self> {
            let accel = Accelerometer::GetDefault().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Khong mo duoc gia toc ke: {e}"))
            })?;
            let state = Arc::new(Mutex::new((None::<f64>, 0.0_f64, 0.0_f64)));
            let running = Arc::new(AtomicBool::new(true));
            let s_clone = Arc::clone(&state);
            let r_clone = Arc::clone(&running);

            thread::spawn(move || {
                let mut count = 0_u64;
                let mut t0 = std::time::Instant::now();
                let mut hz = 0.0_f64;
                while r_clone.load(Ordering::Relaxed) {
                    if let Ok(reading) = accel.GetCurrentReading() {
                        let x = reading.AccelerationX().unwrap_or(0.0);
                        let y = reading.AccelerationY().unwrap_or(-1.0);
                        let z = reading.AccelerationZ().unwrap_or(0.0);
                        let pitch = z.atan2(-y).to_degrees();
                        let roll = x.atan2(-y).to_degrees();
                        count += 1;
                        let elapsed = t0.elapsed().as_secs_f64();
                        if elapsed >= 1.0 {
                            hz = count as f64 / elapsed;
                            count = 0;
                            t0 = std::time::Instant::now();
                        }
                        if let Ok(mut guard) = s_clone.lock() {
                            guard.0 = Some(match guard.0 {
                                Some(prev) => prev + 0.15 * (pitch - prev),
                                None => pitch,
                            });
                            guard.1 += 0.15 * (roll - guard.1);
                            guard.2 = hz;
                        }
                    }
                    thread::sleep(Duration::from_millis(15));
                }
            });

            Ok(Self { state, running })
        }

        pub fn read(&self) -> (Option<f64>, f64, f64) {
            self.state
                .lock()
                .map(|g| *g)
                .unwrap_or((None, 0.0, 0.0))
        }

        pub fn stop(&self) {
            self.running.store(false, Ordering::Relaxed);
        }
    }
}
