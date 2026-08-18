//! Rate-based stellar mass assembly models.
//!
//! These implement `steel_core::StellarGrowthModel` rather than
//! `SmhmModel`: they specify dM*/dt and M* is obtained by integration
//! along the growth track. See `steel_core::stellar_growth` for why the
//! distinction is load-bearing.

mod emerge;
// mod universe_machine; // Task 11 restores this.

pub use emerge::EmergeGrowth;
// pub use universe_machine::UniverseMachineGrowth; // Task 11 restores this.
