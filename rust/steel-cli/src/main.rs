//! STEEL orchestrator binary.
//!
//! Wires plugin implementations chosen by a TOML runfile into a
//! `steel_core::Simulation` and runs it. The registry, runfile schema,
//! and full orchestrator loop are Milestone 5; today this just proves
//! the workspace links together end to end.

use steel_core::cosmology::Cosmology;
use steel_plugins::Planck15;

fn main() -> anyhow::Result<()> {
    let cosmology = Planck15::new();
    println!(
        "steel-cli skeleton: Planck15 age(z=0) = {:.3} Gyr, E(z=1) = {:.4}",
        cosmology.age(0.0),
        cosmology.e_z(1.0)
    );
    println!("Orchestrator (Milestone 5) not yet implemented.");
    Ok(())
}
