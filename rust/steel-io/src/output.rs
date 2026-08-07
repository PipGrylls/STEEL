//! Writes a [`steel_core::RunOutput`] to disk, matching the on-disk
//! layout `Functions.py::SaveData_3` already produces
//! (`Data/Model/Output/RunFiles/RunParam_<params>/Figure3_*.npy`) so
//! existing Python plotting scripts keep working against Rust-produced
//! runs unmodified.

use std::path::{Path, PathBuf};

use anyhow::Result;
use ndarray::Array2;

use steel_core::RunOutput;

use crate::npy::{write_npy_1d, write_npy_2d};

/// Builds the `RunParam_<params>/` directory name the way
/// `Functions.py`'s `SaveData_*` family does: `"_".join(str(p) for p in
/// RunParam)` with a trailing underscore after every field.
pub fn run_param_dir_name(params: &[&str]) -> String {
    let mut s = String::from("RunParam_");
    for p in params {
        s.push_str(p);
        s.push('_');
    }
    s
}

/// Writes the `Figure3_*` fields (`AvaHaloMass`, `AnalyticalModel_SMF`,
/// `Surviving_Sat_SMF_MassRange`) into `output_root/<run_param_dir>/`.
pub fn write_figure3(output_root: &Path, run_param_dir: &str, output: &RunOutput) -> Result<PathBuf> {
    let dir = output_root.join(run_param_dir);
    std::fs::create_dir_all(&dir)?;

    let n_z = output.host_halo_mass.len();
    let n_host = output.host_halo_mass.first().map_or(0, |row| row.len());
    let mut host_mass_flat = Array2::<f64>::zeros((n_z, n_host));
    for (i, row) in output.host_halo_mass.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            host_mass_flat[[i, j]] = v;
        }
    }

    write_npy_2d(&dir.join("Figure3_AvaHaloMass.npy"), host_mass_flat.view())?;
    write_npy_1d(&dir.join("Figure3_AnalyticalModel_SMF.npy"), (&output.surviving_sat_smf[..]).into())?;
    write_npy_1d(&dir.join("Figure3_Surviving_Sat_SMF_MassRange.npy"), (&output.sat_sm_range[..]).into())?;
    write_npy_1d(&dir.join("Figure3_z.npy"), (&output.z[..]).into())?;

    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_param_dir_name_matches_python_convention() {
        let name = run_param_dir_name(&["1.0", "False", "False", "True", "CE", "G19_SE"]);
        assert_eq!(name, "RunParam_1.0_False_False_True_CE_G19_SE_");
    }

    #[test]
    fn write_figure3_creates_expected_files() {
        let dir = tempfile::tempdir().unwrap();
        let output = RunOutput {
            z: vec![0.1, 0.2],
            host_halo_mass: vec![vec![11.0, 12.0], vec![11.1, 12.1]],
            sat_sm_range: vec![9.0, 9.1, 9.2],
            surviving_sat_smf: vec![1.0, 2.0, 3.0],
        };
        let out_dir = write_figure3(dir.path(), "RunParam_test_", &output).unwrap();
        assert!(out_dir.join("Figure3_AvaHaloMass.npy").exists());
        assert!(out_dir.join("Figure3_AnalyticalModel_SMF.npy").exists());
        assert!(out_dir.join("Figure3_Surviving_Sat_SMF_MassRange.npy").exists());
        assert!(out_dir.join("Figure3_z.npy").exists());
    }
}
