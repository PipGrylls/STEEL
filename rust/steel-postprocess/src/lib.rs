//! Numeric (non-plotting) parts of `Scripts/CentralPostprocessing.py`.
//!
//! **Scope note:** `CentralPostprocessing.py::PairFractionData`'s
//! `Return_PF_Plot` (pair fraction vs. host mass cut), `Return_Merger_Plot`
//! (major-merger rate vs. mass), and `Return_Morph_Plot`
//! (elliptical-fraction accumulation) all consume `Accretion_History`
//! and `Pair_Frac` arrays that `steel-cli`'s orchestrator (Milestone 5)
//! doesn't produce yet — that milestone deliberately scoped down to
//! the surviving-satellite SMF (`Figure3`) as the one output it fully
//! validated, and documented the other ~13 `SaveData_*` accumulator
//! arrays as a mechanical follow-up rather than building all of them.
//! Those three methods are consequently **not ported here** — porting
//! them without the data they consume would mean either fabricating
//! that data or writing untestable code, neither of which is better
//! than being explicit about the dependency.
//!
//! What *is* self-contained and ported: central-galaxy stellar mass
//! growth (`Starformation_Centrals`, the central-galaxy sibling of the
//! satellite pipeline's `Starformation_c`), which only needs a halo
//! mass growth history, an SMHM relation, and an external accretion
//! rate — all things Milestones 1-4 already provide.

pub mod central_evolution;

pub use central_evolution::{CentralEvolution, CentralHistory};
