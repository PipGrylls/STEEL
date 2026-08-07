//! Shared run context and the top-level orchestrator skeleton.

use std::sync::Arc;

use crate::baryonic::BaryonicPipeline;
use crate::cosmology::Cosmology;
use crate::halo_growth::HaloGrowthModel;
use crate::hmf::HaloMassFunctionModel;
use crate::merger_time::MergerTimescaleModel;
use crate::shmf::SubhaloMassFunctionModel;
use crate::smhm::SmhmModel;
use crate::stripping::HaloStrippingModel;

/// Values shared read-only across every plugin call in a run.
pub struct ModelContext {
    pub cosmology: Arc<dyn Cosmology>,
    /// Seed for the run's random number generator. Threaded explicitly
    /// rather than an ambient/reseeded-per-call global (unlike the Python
    /// original's `np.random.seed(...)` inside `DarkMatterToStellarMass`),
    /// so runs are reproducible.
    pub rng_seed: u64,
}

/// The independently-injected plugins for one STEEL run, plus the single
/// composed [`BaryonicPipeline`] for per-timestep satellite evolution.
///
/// This is the dependency-injection container: every field is a trait
/// object chosen at startup (from a TOML runfile, via `steel-cli`'s
/// plugin registry) and never branched on again — the orchestrator only
/// ever calls the trait methods.
pub struct Simulation {
    pub context: ModelContext,
    pub halo_growth: Arc<dyn HaloGrowthModel>,
    pub hmf: Arc<dyn HaloMassFunctionModel>,
    pub shmf: Arc<dyn SubhaloMassFunctionModel>,
    pub merger_time: Arc<dyn MergerTimescaleModel>,
    pub halo_stripping: Option<Arc<dyn HaloStrippingModel>>,
    pub smhm: Arc<dyn SmhmModel>,
    pub baryonic: BaryonicPipeline,
}

impl Simulation {
    /// Run the full statistical dark-matter-accretion-history pipeline
    /// (the Rust equivalent of `STEEL.py::OneRealization`'s triple loop
    /// over redshift / host-halo bin / subhalo bin).
    ///
    /// Implemented in Milestone 5 once the plugins it depends on
    /// (Milestones 2-4) exist; the container shape is fixed now so
    /// `steel-cli`'s registry can be written against it.
    pub fn run(&self) {
        unimplemented!(
            "Simulation::run is implemented in Milestone 5 \
             (port of STEEL.py::OneRealization)"
        )
    }
}
