//! Test-only fixtures shared across plugin unit tests: a flat,
//! single-point `GrowthTrack` and a concrete `Cosmology` to build an
//! `AccretionContext` from. The memoryless SMHM/SFR plugins exercised
//! by these tests never dereference either — the content is arbitrary,
//! only that a well-typed context can be constructed.

use steel_core::halo_growth::GrowthTrack;

use crate::cosmology::Planck15;

/// A single-point track and a concrete cosmology, ready to build an
/// `AccretionContext::central(&track, &cosmo, MassDefinition::Vir)` from.
pub(crate) fn flat_ctx() -> (GrowthTrack, Planck15) {
    (GrowthTrack { z: vec![0.0], log_mass: vec![12.0] }, Planck15::new())
}
