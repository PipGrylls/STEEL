//! Stellar-mass-halo-mass (abundance matching) plugins, split into the
//! two genuinely different functional-form families found in
//! `Functions.py`.

mod behroozi;
mod moster;

pub use behroozi::BehrooziFormSmhm;
pub use moster::{MosterFormSmhm, ZEvo};
