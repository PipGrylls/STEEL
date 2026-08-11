//! `.npy` read/write, matching the on-disk layout STEEL's Python side
//! already produces under `Data/Model/Output/RunFiles/RunParam_.../`, so
//! existing plotting/notebooks can read Rust-produced runs unmodified.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView, ArrayView1, ArrayView2, Dimension};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;
use std::io::BufWriter;

/// Writes an array of any rank. `.npy` is rank-agnostic, and STEEL's
/// outputs run from 1-D mass grids to 3-D `(z, host, stellar mass)`
/// cubes, so the per-rank helpers below are all thin wrappers over this.
pub fn write_npy<D: Dimension>(path: &Path, data: ArrayView<f64, D>) -> Result<()> {
    let file = BufWriter::new(
        File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    data.write_npy(file).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn write_npy_1d(path: &Path, data: ArrayView1<f64>) -> Result<()> {
    write_npy(path, data)
}

pub fn write_npy_2d(path: &Path, data: ArrayView2<f64>) -> Result<()> {
    write_npy(path, data)
}

/// Convenience for the many plain `Vec<f64>` axis arrays.
pub fn write_npy_slice(path: &Path, data: &[f64]) -> Result<()> {
    write_npy_1d(path, data.into())
}

pub fn read_npy_1d(path: &Path) -> Result<Array1<f64>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(Array1::<f64>::read_npy(file)?)
}

pub fn read_npy_2d(path: &Path) -> Result<Array2<f64>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(Array2::<f64>::read_npy(file)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

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

    #[test]
    fn round_trips_3d_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test3d.npy");
        let data = Array3::<f64>::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 100 + j * 10 + k) as f64);
        write_npy(&path, data.view()).unwrap();
        let file = File::open(&path).unwrap();
        let loaded = Array3::<f64>::read_npy(file).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn write_reports_the_path_it_failed_on() {
        let path = Path::new("/nonexistent-directory-for-steel-test/x.npy");
        let err = write_npy_slice(path, &[1.0]).unwrap_err();
        assert!(
            format!("{err:#}").contains("x.npy"),
            "error should name the file, got: {err:#}"
        );
    }
}
