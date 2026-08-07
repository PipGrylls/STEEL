//! Grid-search fitting of SMHM parameters against a target stellar
//! mass function — a port of `Scripts/SMHM_Fit.py`.
//!
//! No new physics: this reuses `steel_core::SmhmModel` and
//! `steel_plugins::MosterFormSmhm` unchanged; it's an optimization
//! driver, not a new model. SDSS/Davidzon observational data loading
//! (`pandas.read_csv` in the Python's `SDSS_Plots.py`) isn't ported —
//! callers supply their own target SMF and halo mass function arrays.

pub mod grid_search;
pub mod smf;

pub use grid_search::{fit_low_z, Bound, GridSearchResult, MosterFormParams};
pub use smf::{dm_to_sm, rms_distance};
