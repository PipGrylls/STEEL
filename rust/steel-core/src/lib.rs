//! Trait definitions, shared context, and the orchestrator skeleton for
//! the STEEL semi-empirical galaxy model.
//!
//! Every physical process in the model (halo growth, halo/subhalo mass
//! functions, merger timescales, abundance matching, star formation,
//! quenching, gas supply, stripping) is a trait here; concrete
//! implementations live in `steel-plugins`, and `steel-cli` wires a
//! chosen implementation of each into a [`context::Simulation`] from a
//! TOML runfile.

pub mod accretion;
pub mod baryonic;
pub mod compat;
pub mod context;
pub mod cosmology;
pub mod gas;
pub mod halo_growth;
pub mod hmf;
pub mod merger_time;
mod numerics;
pub mod quenching;
pub mod sfr;
pub mod shmf;
pub mod smhm;
pub mod stellar_growth;
pub mod stripping;

pub use accretion::AccretionContext;
pub use baryonic::{BaryonicPipeline, EvolutionHistory, SatelliteState, Timeline};
pub use compat::{
    validate_composition, Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, Incompatibility,
    PluginDescriptor,
};
pub use context::{ModelContext, OutputSelection, RunConfig, RunOutput, Simulation};
pub use cosmology::{Cosmology, MassDefinition};
pub use gas::GasMassModel;
pub use halo_growth::{GrowthTrack, HaloGrowthModel};
pub use hmf::HaloMassFunctionModel;
pub use merger_time::MergerTimescaleModel;
pub use quenching::{QuenchTimescales, QuenchingModel};
pub use sfr::SfrModel;
pub use shmf::SubhaloMassFunctionModel;
pub use smhm::SmhmModel;
pub use stellar_growth::{integrate_stellar_mass, StellarGrowthAsSmhm, StellarGrowthModel};
pub use stripping::{HaloStrippingModel, HaloStrippingTrack, StellarStrippingModel};
