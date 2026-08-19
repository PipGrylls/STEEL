//! STEEL orchestrator binary.
//!
//! Wires plugin implementations chosen by a TOML runfile into a
//! `steel_core::Simulation`, runs it, and writes the surviving-satellite
//! SMF output. Usage: `steel <runfile.toml> [output_dir]`.

mod registry;

use std::path::PathBuf;

use anyhow::{Context, Result};

use steel_io::RunFile;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let runfile_path = args.next().context("usage: steel <runfile.toml> [output_dir]")?;
    let output_dir = args.next().unwrap_or_else(|| "Data/Model/Output/RunFiles".to_string());

    let runfile = RunFile::from_path(&PathBuf::from(&runfile_path))?;
    let (simulation, config) = registry::build_simulation(&runfile)?;

    eprintln!("Running STEEL...");
    let output = simulation.run(&config);

    // Reproduce `STEEL.py`'s output directory name exactly: it joins the
    // raw `Factor_Stripping_SF` tuple
    // `(Tdyn_Factor, Stripping, SF, z_Evo, SFR_Model, AbnMtch)`, giving
    // e.g. `RunParam_1.0_False_False_True_G19_DPL_G19_SE_`. Matching it
    // is the whole point of keeping the `.npy` layout — the existing
    // Python plotting reads these paths directly, so Rust's native
    // `to_string()` formatting (`1`, `false`) and internal plugin names
    // would put the output somewhere those scripts never look.
    let py_bool = |b: bool| if b { "True" } else { "False" };
    // `Paramaters['PreProcessing']` is carried in STEEL.py's run tuple
    // as a `_PP` suffix on the SFR model name, and so appears in the
    // output directory. Without it a pre-processed run would overwrite
    // its non-pre-processed twin.
    // `[sfr]` is absent for every `[stellar_growth]` runfile today (see
    // `RunFile::sfr`'s doc); there is no `STEEL.py` precedent for that
    // case either, so it gets its own literal marker rather than a
    // fabricated `AbnMtch`-style name.
    let sfr_base_name = match &runfile.sfr {
        Some(cfg) => registry::sfr_legacy_name(cfg).to_string(),
        None => "NoSFR".to_string(),
    };
    let sfr_name =
        if runfile.run.pre_processing { format!("{sfr_base_name}_PP") } else { sfr_base_name };
    // `[smhm]` and `[stellar_growth]` are mutually exclusive suppliers
    // of stellar mass (see `RunFile::smhm`'s doc); `build_simulation`
    // already enforces exactly one is present, so this mirrors that
    // same choice for the output directory name. `z_evo` has no
    // rate-based analogue -- EMERGE/UniverseMachine have no redshift-
    // evolution toggle -- so a `[stellar_growth]`-driven run always
    // reports `True` here, matching STEEL's own default.
    let (z_evo, model_name) = match (&runfile.smhm, &runfile.stellar_growth) {
        (Some(smhm), _) => (smhm.z_evo, registry::smhm_legacy_name(smhm).to_string()),
        (None, Some(sg)) => (true, registry::stellar_growth_legacy_name(sg)),
        (None, None) => unreachable!("build_simulation rejects this combination before main.rs gets here"),
    };
    let run_param_dir = steel_io::run_param_dir_name(&[
        &format!("{:.1}", runfile.merger_time.dynamical_time_factor),
        py_bool(runfile.run.stellar_stripping),
        py_bool(runfile.run.star_formation),
        py_bool(z_evo),
        &sfr_name,
        &model_name,
    ]);

    let written_to = steel_io::write_run(&PathBuf::from(&output_dir), &run_param_dir, &output)?;
    eprintln!("Wrote output to {}", written_to.display());
    println!(
        "z steps: {}, host bins: {}, subhalo bins: {}, SMF bins: {}",
        output.z.len(),
        output.host_halo_mass.ncols(),
        output.sat_halo_mass.len(),
        output.surviving_sat_smf.len()
    );

    Ok(())
}
