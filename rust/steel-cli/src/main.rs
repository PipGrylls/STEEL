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
    let run_param_dir = steel_io::run_param_dir_name(&[
        &format!("{:.1}", runfile.merger_time.dynamical_time_factor),
        py_bool(runfile.run.stellar_stripping),
        py_bool(runfile.run.star_formation),
        py_bool(runfile.smhm.z_evo),
        registry::sfr_legacy_name(&runfile.sfr),
        registry::smhm_legacy_name(&runfile.smhm),
    ]);

    let written_to = steel_io::write_figure3(&PathBuf::from(&output_dir), &run_param_dir, &output)?;
    eprintln!("Wrote output to {}", written_to.display());
    println!(
        "z steps: {}, host bins: {}, SMF bins: {}",
        output.z.len(),
        output.host_halo_mass.first().map_or(0, |r| r.len()),
        output.surviving_sat_smf.len()
    );

    Ok(())
}
