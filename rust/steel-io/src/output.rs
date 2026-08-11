//! Writes a [`steel_core::RunOutput`] to disk in the exact on-disk
//! layout `Functions.py`'s `SaveData_*` family produces
//! (`Data/Model/Output/RunFiles/RunParam_<params>/<Family>_<field>.npy`),
//! so the existing Python plotting and post-processing — which read
//! those paths literally via `LoadData_*` — work unmodified against
//! Rust-produced runs.
//!
//! Every file name below is a compatibility contract copied from the
//! Python; the `SaveData_*` function each group corresponds to is named
//! in the writer's doc comment. Two families the Python emits are
//! deliberately not reproduced: `Figure5`/`Figure7`/`Figure8`/`Figure9`
//! (never written by `OneRealization` — the `SaveData_5/7/8/9` helpers
//! are dead code, and `P_Elliptical`/`Analyticalmodel_SI` are allocated
//! and never touched), and the `.dat`/`.png` companions of
//! `Surviving_Subhalos`, which are a text mirror and a debug plot of
//! data already written as `.npy`.

use std::path::{Path, PathBuf};

use anyhow::Result;

use steel_core::RunOutput;

use crate::npy::{write_npy, write_npy_slice};

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

/// The satellite stellar-mass bin *edges*
/// (`Surviving_Sat_SMF_MassRange`, length `n_sm + 1`).
///
/// `OneRealization` passes the trimmed `[:-1]` left-edge array to most
/// `SaveData_*` calls but the full edge array to `SaveData_10` and
/// `SaveData_Raw_Richness`; both forms are reproduced below so the
/// shapes `LoadData_*` expects are unchanged.
fn sat_sm_edges(output: &RunOutput) -> Vec<f64> {
    let mut edges = output.sat_sm_range.clone();
    match output.sat_sm_range.len() {
        0 => {}
        1 => edges.push(output.sat_sm_range[0]),
        n => {
            let width = output.sat_sm_range[1] - output.sat_sm_range[0];
            edges.push(output.sat_sm_range[n - 1] + width);
        }
    }
    edges
}

/// Writes every output family into `output_root/<run_param_dir>/`,
/// skipping any whose accumulator was disabled by
/// [`steel_core::OutputSelection`] (those arrive here empty).
///
/// Returns the directory written to.
pub fn write_run(output_root: &Path, run_param_dir: &str, output: &RunOutput) -> Result<PathBuf> {
    let dir = output_root.join(run_param_dir);
    std::fs::create_dir_all(&dir)?;

    let edges = sat_sm_edges(output);

    // --- SaveData_3: the headline z=0 satellite SMF ---
    write_npy(&dir.join("Figure3_AvaHaloMass.npy"), output.host_halo_mass.view())?;
    write_npy_slice(&dir.join("Figure3_AnalyticalModel_SMF.npy"), &output.surviving_sat_smf)?;
    write_npy_slice(&dir.join("Figure3_Surviving_Sat_SMF_MassRange.npy"), &output.sat_sm_range)?;
    // Not written by the Python (`LoadData_3` reconstructs z from
    // elsewhere); kept because it costs nothing and makes a Figure3
    // directory self-describing.
    write_npy_slice(&dir.join("Figure3_z.npy"), &output.z)?;

    // --- SaveData_4_6: richness above each stellar-mass cut ---
    write_npy(&dir.join("Figure4_6_AvaHaloMass.npy"), output.host_halo_mass.view())?;
    write_npy(&dir.join("Figure4_6_AnalyticalModelFrac_.npy"), output.cuts_frac.view())?;
    write_npy(&dir.join("Figure4_6_AnalyticalModelNoFrac_.npy"), output.cuts_nofrac.view())?;
    write_npy_slice(&dir.join("Figure4_6_SM_Cuts.npy"), &output.sm_cuts)?;

    // --- SaveData_10: the z=0 satellite SMF split by host halo bin ---
    write_npy(&dir.join("Figure10_AvaHaloMass.npy"), output.host_halo_mass.view())?;
    write_npy(
        &dir.join("Figure10_AnalyticalModel_SMF.npy"),
        output.surviving_sat_smf_by_host.view(),
    )?;
    write_npy_slice(&dir.join("Figure10_Surviving_Sat_SMF_MassRange.npy"), &edges)?;

    // --- SaveData_z_infall ---
    write_npy_slice(&dir.join("z_infall_Surviving_Sat_SMF_MassRange.npy"), &output.sat_sm_range)?;
    write_npy_slice(&dir.join("z_infall_z.npy"), &output.z)?;
    write_npy(&dir.join("z_infall.npy"), output.z_infall.view())?;

    // --- SaveData_MultiEpoch_SubHalos + the Surviving_Subhalos pair ---
    if !output.surviving_subhalos_z_z.is_empty() {
        write_npy_slice(&dir.join("MultiEpoch_SubHalos_z.npy"), &output.z)?;
        write_npy_slice(&dir.join("MultiEpoch_SatHaloMass.npy"), &output.sat_halo_mass)?;
        write_npy(
            &dir.join("MultiEpoch_SurvivingSubhalos_z_z.npy"),
            output.surviving_subhalos_z_z.view(),
        )?;
        // Not part of `SaveData_*`: the Python writes these to
        // `Data/Model/Output/Other/SubHaloes/` as a `.dat` text mirror
        // plus a debug `.png`. Same data, kept here as `.npy` beside
        // everything else.
        write_npy(&dir.join("Surviving_Subhalos.npy"), output.surviving_subhalos.view())?;
        write_npy(
            &dir.join("Surviving_Subhalos_ByParent.npy"),
            output.surviving_subhalos_by_parent.view(),
        )?;
    }

    // --- SaveData_SMFhz + SaveData_Sat_Env_Highz + SaveData_Raw_Richness ---
    if !output.surviving_sat_smf_highz.is_empty() {
        write_npy(&dir.join("SMFhz_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy(
            &dir.join("SMFhz_AnalyticalModel_SMF_Highz.npy"),
            output.surviving_sat_smf_highz.view(),
        )?;
        write_npy_slice(&dir.join("SMFhz_Surviving_Sat_SMF_MassRange.npy"), &output.sat_sm_range)?;

        write_npy(&dir.join("Sat_Env_Highz_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(&dir.join("Sat_Env_Highz_z.npy"), &output.z)?;
        write_npy(
            &dir.join("Sat_Env_Highz_AnalyticalModelFracHighz.npy"),
            output.cuts_frac_highz.view(),
        )?;
        write_npy(
            &dir.join("Sat_Env_Highz_AnalyticalModelNoFracHighz.npy"),
            output.cuts_nofrac_highz.view(),
        )?;
        write_npy_slice(&dir.join("Sat_Env_Highz_SM_Cuts.npy"), &output.sm_cuts)?;

        write_npy(&dir.join("Raw_Richness_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(&dir.join("Raw_Richness_Highz_z.npy"), &output.z)?;
        write_npy_slice(&dir.join("Raw_Richness_Surviving_Sat_SMF_MassRange.npy"), &edges)?;
        write_npy(
            &dir.join("Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy"),
            output.surviving_sat_smf_by_host_highz.view(),
        )?;
    }

    // --- SaveData_Sat_SMHM ---
    if !output.sat_smhm.is_empty() {
        write_npy_slice(&dir.join("Sat_SMHM_z.npy"), &output.z)?;
        write_npy_slice(&dir.join("Sat_SMHM_SatHaloMass.npy"), &output.sat_halo_mass)?;
        write_npy(&dir.join("Sat_SMHM_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(
            &dir.join("Sat_SMHM_Surviving_Sat_SMF_MassRange.npy"),
            &output.sat_sm_range,
        )?;
        write_npy(&dir.join("Sat_SMHM_Sat_SMHM.npy"), output.sat_smhm.view())?;
        write_npy(&dir.join("Sat_SMHM_Sat_SMHM_Host.npy"), output.sat_smhm_host.view())?;
    }

    // --- SaveData_sSFR ---
    if !output.satellite_ssfr.is_empty() {
        write_npy_slice(&dir.join("sSFR_Surviving_Sat_SMF_MassRange.npy"), &output.sat_sm_range)?;
        write_npy_slice(&dir.join("sSFR_Range.npy"), &output.ssfr_range)?;
        write_npy(&dir.join("Satellite_sSFR.npy"), output.satellite_ssfr.view())?;
    }

    // --- SaveData_Mergers + SaveData_Pair_Frac + SaveData_Pair_Frac_Halo ---
    if !output.accretion_history.is_empty() {
        write_npy(&dir.join("Mergers_Accretion_History.npy"), output.accretion_history.view())?;
        write_npy_slice(&dir.join("Mergers_z.npy"), &output.z)?;
        write_npy(&dir.join("Mergers_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(
            &dir.join("Mergers_Surviving_Sat_SMF_MassRange.npy"),
            &output.sat_sm_range,
        )?;

        write_npy(&dir.join("Pair_Frac_Pair_Frac.npy"), output.pair_frac.view())?;
        write_npy_slice(&dir.join("Pair_Frac_z.npy"), &output.z)?;
        write_npy(&dir.join("Pair_Frac_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(
            &dir.join("Pair_Frac_Surviving_Sat_SMF_MassRange.npy"),
            &output.sat_sm_range,
        )?;

        write_npy_slice(&dir.join("Pair_Frac_Halo_z.npy"), &output.z)?;
        write_npy(&dir.join("Pair_Frac_Halo_Pair_Frac_Halo.npy"), output.pair_frac_halo.view())?;
        write_npy(
            &dir.join("Pair_Frac_Halo_Accretion_History_Halo.npy"),
            output.accretion_history_halo.view(),
        )?;
        write_npy(&dir.join("Pair_Frac_Halo_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(&dir.join("Pair_Frac_Halo_SatHaloMass.npy"), &output.sat_halo_mass)?;
    }

    // --- SaveData_Total_Starformation ---
    if !output.total_star_formation_mean.is_empty() {
        write_npy(&dir.join("Total_Starformation_AvaHaloMass.npy"), output.host_halo_mass.view())?;
        write_npy_slice(
            &dir.join("Total_Starformation_Surviving_Sat_SMF_MassRange.npy"),
            &output.sat_sm_range,
        )?;
        write_npy(
            &dir.join("Total_Starformation_Total_StarFormation_Means.npy"),
            output.total_star_formation_mean.view(),
        )?;
        write_npy(
            &dir.join("Total_Starformation_Total_StarFormation_Std.npy"),
            output.total_star_formation_std.view(),
        )?;
        write_npy_slice(&dir.join("Total_Starformation_z.npy"), &output.z)?;
    }

    Ok(dir)
}

/// Writes only the `Figure3_*` files. Retained as the narrow entry point
/// used by callers that run with a reduced [`steel_core::OutputSelection`].
pub fn write_figure3(output_root: &Path, run_param_dir: &str, output: &RunOutput) -> Result<PathBuf> {
    let dir = output_root.join(run_param_dir);
    std::fs::create_dir_all(&dir)?;
    write_npy(&dir.join("Figure3_AvaHaloMass.npy"), output.host_halo_mass.view())?;
    write_npy_slice(&dir.join("Figure3_AnalyticalModel_SMF.npy"), &output.surviving_sat_smf)?;
    write_npy_slice(&dir.join("Figure3_Surviving_Sat_SMF_MassRange.npy"), &output.sat_sm_range)?;
    write_npy_slice(&dir.join("Figure3_z.npy"), &output.z)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn tiny_output(with_optional: bool) -> RunOutput {
        let (n_z, n_host, n_sat, n_sm, n_ssfr, n_cuts) = (3, 2, 4, 5, 6, 2);
        let opt3 = |shape: (usize, usize, usize)| {
            if with_optional {
                Array3::<f64>::zeros(shape)
            } else {
                Array3::<f64>::zeros((0, 0, 0))
            }
        };
        let opt2 = |shape: (usize, usize)| {
            if with_optional {
                Array2::<f64>::zeros(shape)
            } else {
                Array2::<f64>::zeros((0, 0))
            }
        };
        RunOutput {
            z: vec![0.1, 0.2, 0.3],
            host_halo_mass: Array2::zeros((n_z, n_host)),
            sat_halo_mass: vec![10.0, 10.1, 10.2, 10.3],
            sat_sm_range: (0..n_sm).map(|b| 9.0 + b as f64 * 0.1).collect(),
            ssfr_range: (0..n_ssfr).map(|b| -14.0 + b as f64 * 0.1).collect(),
            sm_cuts: vec![9.0, 10.0],
            surviving_subhalos: Array2::zeros((n_z, n_sat)),
            surviving_subhalos_by_parent: opt3((n_z, n_host, n_sat)),
            surviving_subhalos_z_z: opt3((n_z, n_z, n_sat)),
            surviving_sat_smf: vec![1.0; n_sm],
            surviving_sat_smf_by_host: Array2::zeros((n_host, n_sm)),
            surviving_sat_smf_highz: opt2((n_z, n_sm)),
            surviving_sat_smf_by_host_highz: opt3((n_z, n_host, n_sm)),
            sat_smhm: opt3((n_z, n_sat, n_sm)),
            sat_smhm_host: opt3((n_z, n_host, n_sm)),
            satellite_ssfr: opt2((n_sm, n_ssfr)),
            z_infall: Array2::zeros((n_z, n_sm)),
            accretion_history: opt3((n_z, n_host, n_sm)),
            accretion_history_halo: opt3((n_z, n_host, n_sat)),
            pair_frac: opt3((n_z, n_host, n_sm)),
            pair_frac_halo: opt3((n_z, n_host, n_sat)),
            cuts_frac: Array2::zeros((n_cuts, n_host)),
            cuts_nofrac: Array2::zeros((n_cuts, n_host)),
            cuts_frac_highz: opt3((n_cuts, n_z, n_host)),
            cuts_nofrac_highz: opt3((n_cuts, n_z, n_host)),
            total_star_formation_mean: opt3((n_z, n_host, n_sm)),
            total_star_formation_std: opt3((n_z, n_host, n_sm)),
        }
    }

    #[test]
    fn run_param_dir_name_matches_python_convention() {
        // The exact directory `STEEL.py` produces for
        // `Factor_Stripping_SF = ('1.0', False, False, True, 'CE', 'G19_SE')`
        // -- the existing Python plotting scripts read this path
        // literally, so the format is a compatibility contract, not a
        // cosmetic choice.
        let name = run_param_dir_name(&["1.0", "False", "False", "True", "CE", "G19_SE"]);
        assert_eq!(name, "RunParam_1.0_False_False_True_CE_G19_SE_");
    }

    #[test]
    fn write_run_produces_every_file_the_python_loaders_open() {
        let dir = tempfile::tempdir().unwrap();
        let out_dir = write_run(dir.path(), "RunParam_test_", &tiny_output(true)).unwrap();

        // Exactly the file names `Functions.py`'s `LoadData_*` family
        // reads. If a rename ever drifts, the Python post-processing
        // breaks silently at load time -- this is the guard against that.
        let expected = [
            "Figure3_AvaHaloMass.npy",
            "Figure3_AnalyticalModel_SMF.npy",
            "Figure3_Surviving_Sat_SMF_MassRange.npy",
            "Figure4_6_AvaHaloMass.npy",
            "Figure4_6_AnalyticalModelFrac_.npy",
            "Figure4_6_AnalyticalModelNoFrac_.npy",
            "Figure4_6_SM_Cuts.npy",
            "Figure10_AvaHaloMass.npy",
            "Figure10_AnalyticalModel_SMF.npy",
            "Figure10_Surviving_Sat_SMF_MassRange.npy",
            "SMFhz_AvaHaloMass.npy",
            "SMFhz_AnalyticalModel_SMF_Highz.npy",
            "SMFhz_Surviving_Sat_SMF_MassRange.npy",
            "z_infall_Surviving_Sat_SMF_MassRange.npy",
            "z_infall_z.npy",
            "z_infall.npy",
            "sSFR_Surviving_Sat_SMF_MassRange.npy",
            "sSFR_Range.npy",
            "Satellite_sSFR.npy",
            "Sat_SMHM_z.npy",
            "Sat_SMHM_SatHaloMass.npy",
            "Sat_SMHM_AvaHaloMass.npy",
            "Sat_SMHM_Surviving_Sat_SMF_MassRange.npy",
            "Sat_SMHM_Sat_SMHM.npy",
            "Sat_SMHM_Sat_SMHM_Host.npy",
            "Mergers_Accretion_History.npy",
            "Mergers_z.npy",
            "Mergers_AvaHaloMass.npy",
            "Mergers_Surviving_Sat_SMF_MassRange.npy",
            "Pair_Frac_Pair_Frac.npy",
            "Pair_Frac_z.npy",
            "Pair_Frac_AvaHaloMass.npy",
            "Pair_Frac_Surviving_Sat_SMF_MassRange.npy",
            "Sat_Env_Highz_AvaHaloMass.npy",
            "Sat_Env_Highz_z.npy",
            "Sat_Env_Highz_AnalyticalModelFracHighz.npy",
            "Sat_Env_Highz_AnalyticalModelNoFracHighz.npy",
            "Sat_Env_Highz_SM_Cuts.npy",
            "Raw_Richness_AvaHaloMass.npy",
            "Raw_Richness_Highz_z.npy",
            "Raw_Richness_Surviving_Sat_SMF_MassRange.npy",
            "Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy",
            "MultiEpoch_SubHalos_z.npy",
            "MultiEpoch_SatHaloMass.npy",
            "MultiEpoch_SurvivingSubhalos_z_z.npy",
            "Pair_Frac_Halo_z.npy",
            "Pair_Frac_Halo_Pair_Frac_Halo.npy",
            "Pair_Frac_Halo_Accretion_History_Halo.npy",
            "Pair_Frac_Halo_AvaHaloMass.npy",
            "Pair_Frac_Halo_SatHaloMass.npy",
            "Total_Starformation_AvaHaloMass.npy",
            "Total_Starformation_Surviving_Sat_SMF_MassRange.npy",
            "Total_Starformation_Total_StarFormation_Means.npy",
            "Total_Starformation_Total_StarFormation_Std.npy",
            "Total_Starformation_z.npy",
        ];
        for name in expected {
            assert!(out_dir.join(name).exists(), "missing output file {name}");
        }
    }

    #[test]
    fn write_run_skips_disabled_families_but_still_writes_the_core() {
        let dir = tempfile::tempdir().unwrap();
        let out_dir = write_run(dir.path(), "RunParam_reduced_", &tiny_output(false)).unwrap();
        assert!(out_dir.join("Figure3_AnalyticalModel_SMF.npy").exists());
        assert!(out_dir.join("Figure4_6_AnalyticalModelFrac_.npy").exists());
        assert!(!out_dir.join("Mergers_Accretion_History.npy").exists());
        assert!(!out_dir.join("Satellite_sSFR.npy").exists());
        assert!(!out_dir.join("MultiEpoch_SurvivingSubhalos_z_z.npy").exists());
    }

    #[test]
    fn sat_sm_edges_appends_one_bin_width() {
        let out = tiny_output(false);
        let edges = sat_sm_edges(&out);
        assert_eq!(edges.len(), out.sat_sm_range.len() + 1);
        assert!((edges[edges.len() - 1] - 9.5).abs() < 1e-12, "got {:?}", edges);
    }

    #[test]
    fn write_figure3_creates_expected_files() {
        let dir = tempfile::tempdir().unwrap();
        let out_dir = write_figure3(dir.path(), "RunParam_test_", &tiny_output(false)).unwrap();
        assert!(out_dir.join("Figure3_AvaHaloMass.npy").exists());
        assert!(out_dir.join("Figure3_AnalyticalModel_SMF.npy").exists());
        assert!(out_dir.join("Figure3_Surviving_Sat_SMF_MassRange.npy").exists());
        assert!(out_dir.join("Figure3_z.npy").exists());
    }
}
