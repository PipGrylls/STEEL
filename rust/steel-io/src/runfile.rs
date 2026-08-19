//! TOML runfile schema, replacing the hardcoded `AbnMtch`/`Paramaters`
//! dicts and the `Tdyn_Factors` tuple list at the bottom of `STEEL.py`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RunFile {
    #[serde(default)]
    pub merger_time: MergerTimeConfig,
    /// Memoryless `SmhmModel`-based stellar mass, the alternative to
    /// `[stellar_growth]`. Exactly one of the two must be present:
    /// `steel_cli::registry::build_simulation` rejects a runfile that
    /// sets neither (there would be no `Capability::StellarMass`
    /// supplier at all) and, when both are set, the composition
    /// validator rejects the pair as a duplicate `Capability::StellarMass`
    /// rather than silently letting one shadow the other.
    #[serde(default)]
    pub smhm: Option<SmhmConfig>,
    /// STEEL's own post-infall satellite SFR law, consumed by
    /// `BaryonicPipeline` (`[run].star_formation`/`.stellar_stripping`).
    /// Optional: absent means no post-infall satellite evolution is
    /// possible, and `steel_cli::registry::build_simulation` requires
    /// both `[run].star_formation` and `[run].stellar_stripping` to be
    /// `false` in that case (checked, not assumed) -- there would
    /// otherwise be no `SfrModel` to drive it.
    ///
    /// Absent in every `[stellar_growth]` runfile today: EMERGE's own
    /// halo-mass axis is `HFree` while every `[sfr]` model's declared
    /// convention is `PerH` (compatible with `[smhm]`, which shares
    /// that convention, but not with EMERGE), and UniverseMachine's rate
    /// declares `Capability::StarFormationRate` itself, so pairing it
    /// with a second `SfrModel` is a genuine duplicate the composition
    /// validator correctly rejects. Fully integrating `[stellar_growth]`
    /// with post-infall satellite evolution (using the object's own
    /// post-infall halo-mass track, not yet wired anywhere -- see
    /// `steel_core::stripping::HaloStrippingModel`, always `None` in
    /// `Simulation` today) is out of scope here; see `docs/VALIDATION.md`.
    #[serde(default)]
    pub sfr: Option<SfrConfig>,
    #[serde(default)]
    pub gas: GasConfig,
    #[serde(default)]
    pub stripping: StrippingConfig,
    #[serde(default)]
    pub run: RunSection,
    #[serde(default)]
    pub outputs: OutputsSection,
    /// Satellite quenching model. Optional: absent means `Wetzel13`
    /// (STEEL's own default), matching the behaviour of every runfile
    /// written before this field existed. `model = "none"` selects
    /// `NoQuenching`, needed for `UniverseMachineGrowth`
    /// (`[stellar_growth] model = "universe_machine"`), whose SFR PDF
    /// already contains quenching -- stacking `Wetzel13` on top would
    /// quench twice, and the composition validator rejects the
    /// combination (`docs/model-assumptions.md`).
    #[serde(default)]
    pub quenching: Option<QuenchingConfig>,
    /// Rate-based stellar growth model (EMERGE, UniverseMachine), an
    /// alternative supplier of `Capability::StellarMass` to `[smhm]`.
    /// See the doc on `smhm` for the exactly-one-of-the-two contract.
    #[serde(default)]
    pub stellar_growth: Option<StellarGrowthConfig>,
}

/// `model`: `"wetzel13"` (the default, selected both when this section
/// is present with that value and when the section is absent entirely)
/// or `"none"` (`steel_plugins::NoQuenching`, a provably inert model --
/// see its doc comment).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuenchingConfig {
    pub model: String,
}

/// `model`: `"emerge"` (needs `preset`: `o_leary23`) or
/// `"universe_machine"` (needs `preset`: `um_saga`, optional
/// `concentration`: `"dutton_maccio14"`, the default). See
/// `steel_plugins::growth_models`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StellarGrowthConfig {
    pub model: String,
    pub preset: String,
    /// Concentration-mass relation `universe_machine` converts its
    /// vMpeak axis through. Ignored by other models. `None` selects the
    /// default (`dutton_maccio14`).
    #[serde(default)]
    pub concentration: Option<String>,
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
    /// Master switch for all stochastic sources; `false` is the
    /// validation harness's deterministic mode.
    pub scatter: bool,
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
            scatter: true,
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
        let smhm = run.smhm.as_ref().expect("[smhm] present");
        assert_eq!(smhm.model, "moster_form");
        assert_eq!(smhm.preset, "g19_se");
        assert!(smhm.z_evo);
        assert_eq!(run.sfr.expect("[sfr] present").model, "double_power_law");
        assert_eq!(run.run.n_realizations, 5); // default
        assert_eq!(run.merger_time.dynamical_time_factor, 1.0); // default
        assert!(run.quenching.is_none());
    }

    #[test]
    fn sfr_is_optional_when_absent_and_present_when_set() {
        let without = RunFile::parse(
            r#"
            [stellar_growth]
            model = "universe_machine"
            preset = "um_saga"
            "#,
        )
        .unwrap();
        assert!(without.sfr.is_none());
        assert!(without.smhm.is_none());

        let with = RunFile::parse(
            r#"
            [smhm]
            model = "moster_form"
            preset = "g19_se"

            [sfr]
            model = "double_power_law"
            "#,
        )
        .unwrap();
        assert!(with.sfr.is_some());
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
        let p = run.smhm.expect("[smhm] present").params.expect("params should parse");
        assert_eq!(p.m10, 12.0);
        assert_eq!(p.beta11, -0.7);
        assert_eq!(p.scatter, 0.15, "scatter should default to the Python's 0.15");
    }

    /// `[sfr]` remains required (governs post-infall satellite
    /// evolution, orthogonal to how infall-time M* is assigned); `[smhm]`
    /// is deliberately absent here to check that `[stellar_growth]` alone
    /// is enough to parse -- the shape every EMERGE/UM runfile uses.
    #[test]
    fn parses_stellar_growth_section() {
        let run: RunFile = toml::from_str(
            r#"
            [sfr]
            model = "double_power_law"

            [stellar_growth]
            model = "emerge"
            preset = "o_leary23"
            "#,
        )
        .expect("should parse");
        assert!(run.smhm.is_none());
        let sg = run.stellar_growth.expect("section present");
        assert_eq!(sg.model, "emerge");
        assert_eq!(sg.preset, "o_leary23");
    }

    #[test]
    fn smhm_is_optional_when_absent_and_present_when_set() {
        let without = RunFile::parse(
            r#"
            [sfr]
            model = "double_power_law"

            [stellar_growth]
            model = "universe_machine"
            preset = "um_saga"
            "#,
        )
        .unwrap();
        assert!(without.smhm.is_none());

        let with = RunFile::parse(
            r#"
            [smhm]
            model = "moster_form"
            preset = "g19_se"

            [sfr]
            model = "double_power_law"
            "#,
        )
        .unwrap();
        assert!(with.smhm.is_some());
    }

    #[test]
    fn quenching_section_is_absent_by_default() {
        let toml = r#"
            [smhm]
            model = "moster_form"
            preset = "g19_se"

            [sfr]
            model = "double_power_law"
        "#;
        let run = RunFile::parse(toml).unwrap();
        assert!(run.quenching.is_none(), "absence must mean the Wetzel13 default, not a parse requirement");
    }

    #[test]
    fn parses_quenching_none() {
        let toml = r#"
            [smhm]
            model = "moster_form"
            preset = "g19_se"

            [sfr]
            model = "double_power_law"

            [quenching]
            model = "none"
        "#;
        let run = RunFile::parse(toml).unwrap();
        assert_eq!(run.quenching.expect("section present").model, "none");
    }

    #[test]
    fn stellar_growth_section_is_absent_by_default() {
        let toml = r#"
            [smhm]
            model = "moster_form"
            preset = "g19_se"

            [sfr]
            model = "double_power_law"
        "#;
        let run = RunFile::parse(toml).unwrap();
        assert!(run.stellar_growth.is_none());
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
        let smhm = run.smhm.as_ref().expect("[smhm] present");
        assert_eq!(smhm.model, "behroozi_form");
        assert!(!smhm.z_evo);
        assert_eq!(run.sfr.expect("[sfr] present").preset.as_deref(), Some("ce"));
        assert!(run.run.star_formation);
        assert_eq!(run.run.n_realizations, 10);
        assert_eq!(run.run.rng_seed, 7);
    }
}
