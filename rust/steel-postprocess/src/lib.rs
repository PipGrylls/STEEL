//! Numeric (non-plotting) parts of `Scripts/CentralPostprocessing.py`.
//!
//! **Scope note:** `CentralPostprocessing.py::PairFractionData`'s
//! `Return_PF_Plot` (pair fraction vs. host mass cut), `Return_Merger_Plot`
//! (major-merger rate vs. mass), and `Return_Morph_Plot`
//! (elliptical-fraction accumulation) are still **not ported here**.
//! They were originally blocked on `Accretion_History` and `Pair_Frac`,
//! which the orchestrator did not produce; it produces both now (the
//! `mergers` output family), so the blocker is gone and these are a
//! mechanical follow-up rather than an open design question.
//! [`central_assembly`] is the first consumer of that data.
//!
//! What *is* self-contained and ported: central-galaxy stellar mass
//! growth (`Starformation_Centrals`, the central-galaxy sibling of the
//! satellite pipeline's `Starformation_c`), which only needs a halo
//! mass growth history, an SMHM relation, and an external accretion
//! rate — all things Milestones 1-4 already provide.

pub mod central_assembly;
pub mod central_evolution;

pub use central_assembly::{accretion_rate_msun_per_yr, merged_mass_per_central};
pub use central_evolution::{CentralEvolution, CentralHistory};
