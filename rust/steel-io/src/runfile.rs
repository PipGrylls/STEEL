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
    #[serde(default)]
    pub outputs: OutputsSection,
}

/// Which output families to accumulate — the runfile face of
/// `steel_core::OutputSelection`. All default to on, so an existing
/// runfile with no `[outputs]` table produces everything `STEEL.py`
/// does.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct OutputsSection {
    pub subhalo_mass_functions: bool,
    pub high_z_smf: bool,
    pub satellite_smhm: bool,
    pub mergers: bool,
    pub ssfr: bool,
    pub total_star_formation: bool,
}

impl Default for OutputsSection {
    fn default() -> Self {
        Self {
            subhalo_mass_functions: true,
            high_z_smf: true,
            satellite_smhm: true,
            mergers: true,
            ssfr: true,
            total_star_formation: true,
        }
    }
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
/// `g18_not_se`, `g19_se`, `g19_c_mod`, `illustris`, `override_0`,
/// `override_z`; for `behroozi_form` one of `b18c`, `b18t`,
/// `behroozi13`, `lorenzo18`. The two `override_*` presets read their
/// coefficients from `[smhm.params]`.
#[derive(Debug, Deserialize)]
pub struct SmhmConfig {
    pub model: String,
    pub preset: String,
    #[serde(default = "default_true")]
    pub z_evo: bool,
    /// Coefficients for the `override_0` / `override_z` presets, which
    /// correspond to `AbnMtch['Override_0']` / `AbnMtch['Override_z']`.
    /// Required for those presets and ignored for the named ones.
    ///
    /// Paper 3's pair-fraction systematics suite (`AbnMtch['PFT']` plus
    /// one of `M_PFT1..3`, `N_PFT1..3`, `b_PFT1..3`, `g_PFT1..4`) is a
    /// set of single-coefficient perturbations of one base relation, so
    /// it is expressed here as explicit coefficients rather than
    /// fourteen more named presets.
    #[serde(default)]
    pub params: Option<SmhmParams>,
    /// Overrides the `AbnMtch` key used in the output directory name.
    ///
    /// Needed by Paper 3's pair-fraction systematics: all fourteen
    /// variants are the same `override_z` preset with different
    /// coefficients, so without this they would derive the same
    /// directory name and overwrite each other. In the Python each is
    /// its own `AbnMtch` key (`M_PFT1`, `g_PFT4`, ...), and that key is
    /// what the published directory names carry.
    #[serde(default)]
    pub legacy_name: Option<String>,
}

/// Moster-form SMHM coefficients, named as in `Functions.py`'s
/// `Override` dictionary.
#[derive(Debug, Deserialize, Clone, Copy)]
pub struct SmhmParams {
    pub m10: f64,
    pub shmnorm10: f64,
    pub beta10: f64,
    pub gamma10: f64,
    #[serde(default)]
    pub m11: f64,
    #[serde(default)]
    pub shmnorm11: f64,
    #[serde(default)]
    pub beta11: f64,
    #[serde(default)]
    pub gamma11: f64,
    #[serde(default = "default_scatter")]
    pub scatter: f64,
}

fn default_scatter() -> f64 {
    0.15
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
    /// `Paramaters['PreProcessing']` (the `_PP` run-tuple suffix).
    pub pre_processing: bool,
    pub stellar_stripping: bool,
    pub n_realizations: usize,
    pub sat_sm_min: f64,
    pub sat_sm_max: f64,
    pub sat_sm_bin: f64,
    /// `sSFR_Range` grid \[log10 yr^-1\].
    pub ssfr_min: f64,
    pub ssfr_max: f64,
    pub ssfr_bin: f64,
    /// `SM_Cuts` — stellar-mass thresholds for the richness integrals.
    pub sm_cuts: Vec<f64>,
    /// Pair-fraction separation limits \[physical kpc\].
    pub pair_radius_outer_kpc: f64,
    pub pair_radius_inner_kpc: f64,
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
            pre_processing: false,
            stellar_stripping: false,
            n_realizations: 5,
            sat_sm_min: 9.0,
            sat_sm_max: 13.0,
            sat_sm_bin: 0.1,
            ssfr_min: -14.0,
            ssfr_max: -8.0,
            ssfr_bin: 0.1,
            sm_cuts: vec![9.0, 9.5, 10.0, 10.5, 11.0, 11.45],
            pair_radius_outer_kpc: 30.0,
            pair_radius_inner_kpc: 5.0,
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
    fn parses_smhm_override_coefficients() {
        // Paper 3's pair-fraction systematics suite perturbs one
        // coefficient of a common base relation at a time, so it needs
        // explicit coefficients rather than a named preset.
        let toml = r#"
            [smhm]
            model = "moster_form"
            preset = "override_z"

            [smhm.params]
            m10 = 12.0
            shmnorm10 = 0.032
            beta10 = 1.5
            gamma10 = 0.56
            m11 = 0.6
            shmnorm11 = -0.014
            beta11 = -0.7
            gamma11 = 0.08

            [sfr]
            model = "double_power_law"
        "#;
        let run = RunFile::parse(toml).unwrap();
        let p = run.smhm.params.expect("params should parse");
        assert_eq!(p.m10, 12.0);
        assert_eq!(p.beta11, -0.7);
        assert_eq!(p.scatter, 0.15, "scatter should default to the Python's 0.15");
    }

    #[test]
    fn an_infinite_dynamical_time_factor_parses() {
        // Paper 1's `f_tdyn = inf` model ("satellites never merge").
        let toml = r#"
            [merger_time]
            dynamical_time_factor = inf

            [smhm]
            model = "moster_form"
            preset = "g18"

            [sfr]
            model = "double_power_law"
        "#;
        let run = RunFile::parse(toml).unwrap();
        assert!(run.merger_time.dynamical_time_factor.is_infinite());
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
