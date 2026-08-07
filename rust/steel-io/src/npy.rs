//! `.npy` read/write, matching the on-disk layout STEEL's Python side
//! already produces under `Data/Model/Output/RunFiles/RunParam_.../`, so
//! existing plotting/notebooks can read Rust-produced runs unmodified
//! during the transition.
//!
//! Only the generic array round-trip lives here for now; the
//! `RunParam_<params>/<Figure_N>_<field>.npy` naming convention (the
//! Rust equivalent of `Functions.py`'s `SaveData_*`/`LoadData_*`
//! functions) is added in Milestone 5 once the orchestrator's output
//! schema exists.

use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;
use std::io::BufWriter;

pub fn write_npy_1d(path: &Path, data: ArrayView1<f64>) -> Result<()> {
    let file = BufWriter::new(File::create(path)?);
    data.write_npy(file)?;
    Ok(())
}

pub fn write_npy_2d(path: &Path, data: ArrayView2<f64>) -> Result<()> {
    let file = BufWriter::new(File::create(path)?);
    data.write_npy(file)?;
    Ok(())
}

pub fn read_npy_1d(path: &Path) -> Result<Array1<f64>> {
    let file = File::open(path)?;
    Ok(Array1::<f64>::read_npy(file)?)
}

pub fn read_npy_2d(path: &Path) -> Result<Array2<f64>> {
    let file = File::open(path)?;
    Ok(Array2::<f64>::read_npy(file)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn round_trips_1d_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test1d.npy");
        let data = array![1.0, 2.0, 3.5, -4.25];
        write_npy_1d(&path, data.view()).unwrap();
        let loaded = read_npy_1d(&path).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn round_trips_2d_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test2d.npy");
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        write_npy_2d(&path, data.view()).unwrap();
        let loaded = read_npy_2d(&path).unwrap();
        assert_eq!(data, loaded);
    }
}
