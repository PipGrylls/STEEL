//! Builds a [`steel_core::Simulation`] from a [`steel_io::RunFile`].
//!
//! A plain string-keyed match per trait rather than a generic
//! `HashMap<&str, fn(...)>` registry — most traits here (`Cosmology`,
//! `HaloGrowthModel`, `HaloMassFunctionModel`, `SubhaloMassFunctionModel`)
//! have exactly one implementation today, so a lookup table would be
//! pure ceremony; `SmhmModel` and `SfrModel` (the ones with real preset
//! variety) get real `match`es. Selection happens once at startup from
//! the TOML runfile, so this is proportionate to the actual need.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use steel_core::baryonic::BaryonicPipeline;
use steel_core::compat::{validate_composition, CosmologyTag, DescribedPlugin, PluginDescriptor};
use steel_core::context::{ModelContext, OutputSelection, RunConfig, Simulation};
use steel_core::stellar_growth::StellarGrowthModel;
use steel_core::{QuenchingModel, SfrModel, SmhmModel, StellarGrowthAsSmhm, StellarStrippingModel};
use steel_io::runfile::RunFile;
use steel_plugins::{
    BehrooziFormSmhm, Cattaneo11, ConcentrationMassRelation, Despali16, DoublePowerLawSfr,
    DuttonMaccio14, EmergeGrowth, Jiang16, McCavanaBK08, MosterFormSmhm, NoQuenching, Planck15,
    RodriguezPuebla17, SchreiberFormSfr, StewartScaling, TomczakFormSfr, UniverseMachineGrowth,
    VandenBosch14, Wetzel13,
};

fn build_smhm(cfg: &steel_io::runfile::SmhmConfig) -> Result<(Box<dyn SmhmModel>, PluginDescriptor)> {
    match cfg.model.as_str() {
        "moster_form" => {
            let m: MosterFormSmhm = match cfg.preset.as_str() {
                "moster13" => MosterFormSmhm::moster13(cfg.z_evo),
                "moster10" => MosterFormSmhm::moster10(cfg.z_evo),
                "g18" => MosterFormSmhm::g18(cfg.z_evo),
                "g18_not_se" => MosterFormSmhm::g18_not_se(cfg.z_evo),
                "g19_se" => MosterFormSmhm::g19_se(cfg.z_evo),
                "g19_c_mod" => MosterFormSmhm::g19_c_mod(cfg.z_evo),
                "illustris" => MosterFormSmhm::illustris(cfg.z_evo),
                "pft" => MosterFormSmhm::pft(cfg.z_evo),
                "hmevo" => {
                    // `AbnMtch['HMevo_param']` is the only free
                    // coefficient; the other seven are fixed, so it
                    // rides in on `gamma11` rather than needing a
                    // one-field table of its own.
                    let p = cfg
                        .params
                        .ok_or_else(|| anyhow!("smhm preset hmevo requires a [smhm.params] table with gamma11"))?;
                    MosterFormSmhm::hmevo(p.gamma11, cfg.z_evo)
                }
                preset @ ("override_0" | "override_z") => {
                    let p = cfg.params.ok_or_else(|| {
                        anyhow!("smhm preset {preset} requires a [smhm.params] table")
                    })?;
                    if preset == "override_0" {
                        MosterFormSmhm::override_z0(
                            p.m10, p.shmnorm10, p.beta10, p.gamma10, p.scatter, cfg.z_evo,
                        )
                    } else {
                        MosterFormSmhm::override_full(
                            p.m10, p.shmnorm10, p.beta10, p.gamma10, p.m11, p.shmnorm11,
                            p.beta11, p.gamma11, p.scatter, cfg.z_evo,
                        )
                    }
                }
                other => return Err(anyhow!("unknown moster_form preset: {other}")),
            };
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        "behroozi_form" => {
            let m: BehrooziFormSmhm = match cfg.preset.as_str() {
                "b18c" => BehrooziFormSmhm::behroozi18c(),
                "b18t" => BehrooziFormSmhm::behroozi18t(),
                // `behrozi13` was the original (misspelled) key; kept as
                // an alias so any runfile already written against it
                // keeps working.
                "behroozi13" | "behrozi13" => BehrooziFormSmhm::behroozi13(),
                "lorenzo18" => BehrooziFormSmhm::lorenzo18(),
                other => return Err(anyhow!("unknown behroozi_form preset: {other}")),
            };
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        // A third functional family with exactly one preset — kept as
        // its own `model` rather than folded into either sibling
        // because it shares neither's equation.
        "rodriguez_puebla_form" => match cfg.preset.as_str() {
            "rp17" => {
                let m = RodriguezPuebla17;
                let descriptor = m.descriptor();
                Ok((Box::new(m), descriptor))
            }
            other => Err(anyhow!("unknown rodriguez_puebla_form preset: {other}")),
        },
        other => Err(anyhow!("unknown smhm model: {other}")),
    }
}

/// Builds a rate-based `StellarGrowthModel`, the `[stellar_growth]`
/// alternative to `[smhm]`. `build_simulation` wraps the result in
/// `steel_core::StellarGrowthAsSmhm` when `[smhm]` is absent, so this
/// actually drives the run; when both sections are present it is still
/// built here (discarded) purely so its descriptor takes part in
/// `validate_composition`'s duplicate-`Capability::StellarMass` check.
fn build_stellar_growth(
    cfg: &steel_io::runfile::StellarGrowthConfig,
) -> Result<(Box<dyn StellarGrowthModel>, PluginDescriptor)> {
    match (cfg.model.as_str(), cfg.preset.as_str()) {
        ("emerge", "o_leary23") => {
            let m = EmergeGrowth::o_leary23();
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        ("universe_machine", "um_saga") => {
            let cm: Arc<dyn ConcentrationMassRelation> = match cfg.concentration.as_deref() {
                None | Some("dutton_maccio14") => Arc::new(DuttonMaccio14),
                Some(other) => return Err(anyhow!("unknown concentration relation: {other}")),
            };
            let m = UniverseMachineGrowth::um_saga(cm);
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        (model, preset) => Err(anyhow!("unknown stellar_growth model/preset: {model}/{preset}")),
    }
}

fn build_sfr(cfg: &steel_io::runfile::SfrConfig) -> Result<(Box<dyn SfrModel>, PluginDescriptor)> {
    match cfg.model.as_str() {
        "tomczak_form" => {
            let preset = cfg.preset.as_deref().ok_or_else(|| anyhow!("tomczak_form requires a preset"))?;
            let m: TomczakFormSfr = match preset {
                "t16" => TomczakFormSfr::t16(),
                "ce" => TomczakFormSfr::ce(),
                "illustris" => TomczakFormSfr::illustris(),
                other => return Err(anyhow!("unknown tomczak_form preset: {other}")),
            };
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        "schreiber_form" => {
            let preset = cfg.preset.as_deref().ok_or_else(|| anyhow!("schreiber_form requires a preset"))?;
            let m: SchreiberFormSfr = match preset {
                "s15" => SchreiberFormSfr::s15(),
                "s16ce" => SchreiberFormSfr::s16ce(),
                other => return Err(anyhow!("unknown schreiber_form preset: {other}")),
            };
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        // The satellite branch of `Starformation_c`. The central branch
        // (`Starformation_Centrals`) has different coefficients and is
        // reached through `steel_postprocess::CentralEvolution`, not
        // through a runfile, so it has no key here.
        "double_power_law" => {
            let m = DoublePowerLawSfr::satellite();
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        other => Err(anyhow!("unknown sfr model: {other}")),
    }
}

/// Stand-in `SfrModel` used only when `[sfr]` is absent from the
/// runfile. `build_simulation` requires `[run].star_formation` and
/// `[run].stellar_stripping` to both be `false` whenever `[sfr]` is
/// absent (checked explicitly, not merely hoped for), so
/// `BaryonicPipeline::evolve` -- the only place any `SfrModel` is ever
/// called -- structurally never runs for such a runfile: `evolved =
/// n_w > 0 && (config.star_formation || config.stellar_stripping)` in
/// `steel_core::context::Simulation::run` is always `false`. This type
/// only exists to satisfy `BaryonicPipeline::new`'s constructor, which
/// takes a concrete `Box<dyn SfrModel>` unconditionally; it panics if
/// ever actually called, so a future change that starts invoking it
/// without also lifting the `[sfr]` requirement fails loudly rather
/// than silently returning a wrong, always-zero SFR.
struct UnreachableSfr;

impl SfrModel for UnreachableSfr {
    fn log_sfr(&self, _log_sm: f64, _z: f64, _ctx: &steel_core::accretion::AccretionContext<'_>) -> f64 {
        unreachable!(
            "UnreachableSfr::log_sfr called -- [sfr] was absent, so build_simulation should have \
             refused a runfile with star_formation or stellar_stripping enabled"
        )
    }
}

fn build_gas(cfg: &steel_io::runfile::GasConfig, cosmology: &Planck15) -> Result<Box<dyn steel_core::GasMassModel>> {
    match cfg.model.as_str() {
        "stewart_scaling" => Ok(Box::new(StewartScaling::from_cosmology(cosmology))),
        other => Err(anyhow!("unknown gas model: {other}")),
    }
}

fn build_stripping(cfg: &steel_io::runfile::StrippingConfig) -> Result<Box<dyn StellarStrippingModel>> {
    match cfg.model.as_str() {
        "cattaneo11" => Ok(Box::new(Cattaneo11)),
        other => Err(anyhow!("unknown stripping model: {other}")),
    }
}

/// `cfg = None` (the `[quenching]` section absent from the runfile) and
/// `cfg = Some(model = "wetzel13")` are the same thing: `Wetzel13`,
/// STEEL's own satellite quenching model and the default every runfile
/// written before `[quenching]` existed already gets. `model = "none"`
/// selects `NoQuenching` -- see its doc comment for why this exists.
fn build_quenching(
    cfg: Option<&steel_io::runfile::QuenchingConfig>,
) -> Result<(Box<dyn QuenchingModel>, PluginDescriptor)> {
    match cfg.map(|c| c.model.as_str()) {
        None | Some("wetzel13") => {
            let m = Wetzel13::new();
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        Some("none") => {
            let m = NoQuenching;
            let descriptor = m.descriptor();
            Ok((Box::new(m), descriptor))
        }
        Some(other) => Err(anyhow!("unknown quenching model: {other}")),
    }
}

/// The Python identifier for an SMHM preset — the `AbnMtch` key that
/// `STEEL.py`'s `Factor_Stripping_SF` tuple carries into the output
/// directory name (`G19_SE`, `Moster`, ...). Needed so Rust-produced
/// runs land in directories the existing Python plotting scripts
/// already know how to find.
pub fn smhm_legacy_name(cfg: &steel_io::runfile::SmhmConfig) -> &str {
    if let Some(name) = cfg.legacy_name.as_deref() {
        return name;
    }
    match (cfg.model.as_str(), cfg.preset.as_str()) {
        ("moster_form", "moster13") => "Moster",
        ("moster_form", "moster10") => "Moster10",
        ("moster_form", "g18") => "G18",
        ("moster_form", "g18_not_se") => "G18_notSE",
        ("moster_form", "g19_se") => "G19_SE",
        ("moster_form", "g19_c_mod") => "G19_cMod",
        ("moster_form", "illustris") => "Illustris",
        ("moster_form", "pft") => "PFT",
        // `STEEL.py` parses the free parameter back out of the last
        // three characters of this name, so an `hmevo` run wanting a
        // Python-compatible directory must set `legacy_name`
        // explicitly (e.g. `HMevo_alt_0.3`); this is only the fallback.
        ("moster_form", "hmevo") => "HMevo",
        ("moster_form", "override_0") => "Override_0",
        ("moster_form", "override_z") => "Override_z",
        ("behroozi_form", "b18c") => "B18c",
        ("behroozi_form", "b18t") => "B18t",
        ("behroozi_form", "behroozi13" | "behrozi13") => "Behroozi13",
        ("behroozi_form", "lorenzo18") => "Lorenzo18",
        ("rodriguez_puebla_form", "rp17") => "RP17",
        _ => "Unknown",
    }
}

/// The output-directory name for a `[stellar_growth]`-driven run, the
/// `StellarGrowthConfig` counterpart to [`smhm_legacy_name`]. There is
/// no `STEEL.py`/`AbnMtch` precedent for EMERGE or UniverseMachine
/// (they never ran through the Python), so this coins a new but
/// analogous identifier (`EMERGE_o_leary23`, `UniverseMachine_um_saga`)
/// rather than forcing one of the `AbnMtch` names to mean something it
/// doesn't.
pub fn stellar_growth_legacy_name(cfg: &steel_io::runfile::StellarGrowthConfig) -> String {
    match cfg.model.as_str() {
        "emerge" => format!("EMERGE_{}", cfg.preset),
        "universe_machine" => format!("UniverseMachine_{}", cfg.preset),
        other => format!("Unknown_{other}"),
    }
}

/// The Python `SFR_Model` string for an SFR preset (`CE`, `G19_DPL`,
/// ...), used for the same output-directory-compatibility reason as
/// [`smhm_legacy_name`].
pub fn sfr_legacy_name(cfg: &steel_io::runfile::SfrConfig) -> &'static str {
    match (cfg.model.as_str(), cfg.preset.as_deref()) {
        ("tomczak_form", Some("t16")) => "T16",
        ("tomczak_form", Some("ce")) => "CE",
        ("tomczak_form", Some("illustris")) => "Illustris",
        ("schreiber_form", Some("s15")) => "S15",
        ("schreiber_form", Some("s16ce")) => "S16CE",
        ("double_power_law", _) => "G19_DPL",
        _ => "Unknown",
    }
}

pub fn build_simulation(runfile: &RunFile) -> Result<(Simulation, RunConfig)> {
    let cosmology = Planck15::new();
    // The only cosmology `build_simulation` wires in today; kept as a
    // local rather than a runfile field until a second cosmology
    // plugin exists to choose between.
    let run_cosmology_tag = CosmologyTag::Planck15;

    let mut descriptors = Vec::new();
    let sfr: Box<dyn SfrModel> = match &runfile.sfr {
        Some(cfg) => {
            let (m, d) = build_sfr(cfg)?;
            descriptors.push(d);
            m
        }
        None => {
            if runfile.run.star_formation || runfile.run.stellar_stripping {
                return Err(anyhow!(
                    "runfile enables [run].star_formation or [run].stellar_stripping but has \
                     no [sfr] section -- post-infall satellite evolution needs an SfrModel. \
                     Either add [sfr], or set both to false (the [stellar_growth] runfiles ship \
                     with today, since [stellar_growth] only drives infall-time stellar mass; \
                     see docs/VALIDATION.md)."
                ));
            }
            Box::new(UnreachableSfr)
        }
    };
    let gas = build_gas(&runfile.gas, &cosmology)?;
    let stripping = build_stripping(&runfile.stripping)?;
    let (quenching, quenching_descriptor) = build_quenching(runfile.quenching.as_ref())?;
    descriptors.push(quenching_descriptor);

    // Exactly one of `[smhm]` / `[stellar_growth]` supplies
    // `Capability::StellarMass`. Neither present is caught here
    // (`validate_composition` only checks pairwise conflicts among
    // descriptors that exist, so it has nothing to say about an
    // *absent* capability); both present is caught by
    // `validate_composition` below, once both descriptors are pushed.
    if runfile.smhm.is_none() && runfile.stellar_growth.is_none() {
        return Err(anyhow!(
            "runfile must set exactly one of [smhm] or [stellar_growth] -- \
             neither section is present, so nothing supplies a stellar mass"
        ));
    }
    // `StellarGrowthAsSmhm` (steel-core) integrates a rate-based model's
    // `stellar_growth_rate` along the same `AccretionContext` the
    // orchestrator already builds for `[smhm]`'s memoryless
    // `stellar_mass`, so either section can drive
    // `Simulation::smhm: Arc<dyn SmhmModel>` through one call site.
    let smhm_from_config: Option<Arc<dyn SmhmModel>> = match &runfile.smhm {
        Some(cfg) => {
            let (m, d) = build_smhm(cfg)?;
            descriptors.push(d);
            Some(Arc::from(m))
        }
        None => None,
    };
    let smhm_from_stellar_growth: Option<Arc<dyn SmhmModel>> = match &runfile.stellar_growth {
        Some(cfg) => {
            let (m, d) = build_stellar_growth(cfg)?;
            descriptors.push(d);
            Some(Arc::new(StellarGrowthAsSmhm::new(m)))
        }
        None => None,
    };

    if let Err(problems) = validate_composition(&descriptors, run_cosmology_tag) {
        let detail = problems.iter().map(|p| format!("  - {p}")).collect::<Vec<_>>().join("\n");
        return Err(anyhow!(
            "incompatible plugin combination in this runfile:\n{detail}\n\n\
             See docs/model-assumptions.md for what each plugin assumes."
        ));
    }

    // Validation just confirmed at most one of the two supplies
    // `Capability::StellarMass` (both present would have failed as a
    // `DuplicateCapability` above), and the emptiness check above ruled
    // out neither being present, so exactly one of these is `Some`.
    let smhm = smhm_from_config
        .or(smhm_from_stellar_growth)
        .expect("exactly one of [smhm]/[stellar_growth] checked present above");

    let halo_growth = Arc::new(VandenBosch14::new(&cosmology));
    let hmf = Arc::new(Despali16::new(&cosmology));
    let shmf = Arc::new(Jiang16::default_calibration());
    let merger_time = Arc::new(McCavanaBK08::new(
        runfile.merger_time.dynamical_time_factor,
        runfile.merger_time.redshift_correction,
    ));

    let baryonic = BaryonicPipeline::new(sfr, quenching, gas, stripping);

    let context = ModelContext { cosmology: Arc::new(cosmology), rng_seed: runfile.run.rng_seed };

    let simulation = Simulation {
        context,
        halo_growth,
        hmf,
        shmf,
        merger_time,
        halo_stripping: None,
        smhm,
        baryonic,
    };

    let config = RunConfig {
        log_m_min: runfile.run.log_m_min,
        log_m_max: runfile.run.log_m_max,
        log_m_bin: runfile.run.log_m_bin,
        sat_min_offset: runfile.run.sat_min_offset,
        z_reference_min: runfile.run.z_reference_min,
        star_formation: runfile.run.star_formation,
        pre_processing: runfile.run.pre_processing,
        stellar_stripping: runfile.run.stellar_stripping,
        n_realizations: runfile.run.n_realizations,
        scatter: runfile.run.scatter,
        sat_sm_min: runfile.run.sat_sm_min,
        sat_sm_max: runfile.run.sat_sm_max,
        sat_sm_bin: runfile.run.sat_sm_bin,
        ssfr_min: runfile.run.ssfr_min,
        ssfr_max: runfile.run.ssfr_max,
        ssfr_bin: runfile.run.ssfr_bin,
        sm_cuts: runfile.run.sm_cuts.clone(),
        outputs: OutputSelection {
            subhalo_mass_functions: runfile.outputs.subhalo_mass_functions,
            high_z_smf: runfile.outputs.high_z_smf,
            satellite_smhm: runfile.outputs.satellite_smhm,
            mergers: runfile.outputs.mergers,
            ssfr: runfile.outputs.ssfr,
            total_star_formation: runfile.outputs.total_star_formation,
        },
        pair_radius_outer_kpc: runfile.run.pair_radius_outer_kpc,
        pair_radius_inner_kpc: runfile.run.pair_radius_inner_kpc,
    };

    Ok((simulation, config))
}
