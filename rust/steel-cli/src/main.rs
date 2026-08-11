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

    let run_param_dir = steel_io::run_param_dir_name(&[
        &runfile.merger_time.dynamical_time_factor.to_string(),
        &runfile.run.stellar_stripping.to_string(),
        &runfile.run.star_formation.to_string(),
        &runfile.smhm.model,
        &runfile.smhm.preset,
        &runfile.sfr.model,
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
