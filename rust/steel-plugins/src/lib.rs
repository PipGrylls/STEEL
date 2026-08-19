//! Built-in implementations of the `steel-core` plugin traits.
//!
//! Each module corresponds to one physical process; see `steel-core` for
//! the trait each implements and why it's a plugin.

pub mod cosmology;
pub mod gas;
pub mod growth;
pub mod growth_models;
pub mod halo_growth;
pub mod harmonise;
pub mod hmf;
pub mod merger_time;
mod numerics;
pub mod quenching;
pub mod sfr;
pub mod shmf;
pub mod smhm;
pub mod stripping;
#[cfg(test)]
mod test_support;
pub mod variance;

pub use cosmology::Planck15;
pub use gas::StewartScaling;
pub use growth_models::{EmergeGrowth, UniverseMachineGrowth};
pub use halo_growth::VandenBosch14;
pub use harmonise::{ConcentrationMassRelation, DuttonMaccio14, HConvention, Imf};
pub use hmf::Despali16;
pub use merger_time::McCavanaBK08;
pub use quenching::Wetzel13;
pub use sfr::{DoublePowerLawSfr, SchreiberFormSfr, TomczakFormSfr};
pub use shmf::Jiang16;
pub use smhm::{BehrooziFormSmhm, MosterFormSmhm, RodriguezPuebla17};
pub use stripping::{Cattaneo11, HaloStrippingVdb05};
