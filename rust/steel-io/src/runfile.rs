//! TOML runfile schema, replacing the hardcoded `AbnMtch`/`Paramaters`
//! dicts and the `Tdyn_Factors` tuple list at the bottom of `STEEL.py`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RunFile {
    #[serde(default)]
    pub merger_time: MergerTimeConfig,
    pub smhm: SmhmConfig,
    pub sfr: SfrConfig,
    #[serde(default)]
    pub gas: GasConfig,
    #[serde(default)]
    pub stripping: StrippingConfig,
    #[serde(default)]
    pub run: RunSection,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MergerTimeConfig {
    pub dynamical_time_factor: f64,
    pub redshift_correction: bool,
}

impl Default for MergerTimeConfig {
    fn default() -> Self {
        Self { dynamical_time_factor: 1.0, redshift_correction: false }
    }
}

/// `model`: `"moster_form"` or `"behroozi_form"`.
/// `preset`: for `moster_form` one of `moster13`, `moster10`, `g18`,
/// `g18_not_se`, `g19_se`, `g19_c_mod`, `illustris`; for
/// `behroozi_form` one of `b18c`, `b18t`, `behroozi13`, `lorenzo18`.
#[derive(Debug, Deserialize)]
pub struct SmhmConfig {
    pub model: String,
    pub preset: String,
    #[serde(default = "default_true")]
    pub z_evo: bool,
}

/// `model`: `"tomczak_form"` (needs `preset`: `t16`, `ce`, `illustris`),
/// `"schreiber_form"` (needs `preset`: `s15`, `s16ce`), or
/// `"double_power_law"` (no preset).
#[derive(Debug, Deserialize)]
pub struct SfrConfig {
    pub model: String,
    #[serde(default)]
    pub preset: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GasConfig {
    pub model: String,
}

impl Default for GasConfig {
    fn default() -> Self {
        Self { model: "stewart_scaling".to_string() }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct StrippingConfig {
    pub model: String,
}

impl Default for StrippingConfig {
    fn default() -> Self {
        Self { model: "cattaneo11".to_string() }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct RunSection {
    pub log_m_min: f64,
    pub log_m_max: f64,
    pub log_m_bin: f64,
    pub sat_min_offset: f64,
    pub z_reference_min: f64,
    pub star_formation: bool,
    pub stellar_stripping: bool,
    pub n_realizations: usize,
    pub sat_sm_min: f64,
    pub sat_sm_max: f64,
    pub sat_sm_bin: f64,
    pub rng_seed: u64,
}

impl Default for RunSection {
    fn default() -> Self {
        Self {
            log_m_min: 11.0,
            log_m_max: 16.6,
            log_m_bin: 0.1,
            sat_min_offset: -1.0,
            z_reference_min: 0.1,
            star_formation: false,
            stellar_stripping: false,
            n_realizations: 5,
            sat_sm_min: 9.0,
            sat_sm_max: 13.0,
            sat_sm_bin: 0.1,
            rng_seed: 42,
        }
    }
}

fn default_true() -> bool {
    true
}

impl RunFile {
    pub fn parse(s: &str) -> Result<Self> {
        toml::from_str(s).context("parsing TOML runfile")
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_runfile() {
        let toml = r#"
            [smhm]
            model = "moster_form"
            preset = "g19_se"

            [sfr]
            model = "double_power_law"
        "#;
        let run = RunFile::parse(toml).unwrap();
        assert_eq!(run.smhm.model, "moster_form");
        assert_eq!(run.smhm.preset, "g19_se");
        assert!(run.smhm.z_evo);
        assert_eq!(run.sfr.model, "double_power_law");
        assert_eq!(run.run.n_realizations, 5); // default
        assert_eq!(run.merger_time.dynamical_time_factor, 1.0); // default
    }

    #[test]
    fn parses_a_full_runfile() {
        let toml = r#"
            [merger_time]
            dynamical_time_factor = 0.8
            redshift_correction = true

            [smhm]
            model = "behroozi_form"
            preset = "b18c"
            z_evo = false

            [sfr]
            model = "tomczak_form"
            preset = "ce"

            [gas]
            model = "stewart_scaling"

            [stripping]
            model = "cattaneo11"

            [run]
            star_formation = true
            stellar_stripping = true
            n_realizations = 10
            rng_seed = 7
        "#;
        let run = RunFile::parse(toml).unwrap();
        assert_eq!(run.merger_time.dynamical_time_factor, 0.8);
        assert!(run.merger_time.redshift_correction);
        assert_eq!(run.smhm.model, "behroozi_form");
        assert!(!run.smhm.z_evo);
        assert_eq!(run.sfr.preset.as_deref(), Some("ce"));
        assert!(run.run.star_formation);
        assert_eq!(run.run.n_realizations, 10);
        assert_eq!(run.run.rng_seed, 7);
    }
}
