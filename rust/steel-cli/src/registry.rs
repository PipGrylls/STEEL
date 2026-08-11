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
use steel_core::context::{ModelContext, OutputSelection, RunConfig, Simulation};
use steel_core::{QuenchingModel, SfrModel, SmhmModel, StellarStrippingModel};
use steel_io::runfile::RunFile;
use steel_plugins::{
    BehrooziFormSmhm, Cattaneo11, Despali16, DoublePowerLawSfr, Jiang16, McCavanaBK08, MosterFormSmhm,
    Planck15, SchreiberFormSfr, StewartScaling, TomczakFormSfr, VandenBosch14, Wetzel13,
};

fn build_smhm(cfg: &steel_io::runfile::SmhmConfig) -> Result<Box<dyn SmhmModel>> {
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
            Ok(Box::new(m))
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
            Ok(Box::new(m))
        }
        other => Err(anyhow!("unknown smhm model: {other}")),
    }
}

fn build_sfr(cfg: &steel_io::runfile::SfrConfig) -> Result<Box<dyn SfrModel>> {
    match cfg.model.as_str() {
        "tomczak_form" => {
            let preset = cfg.preset.as_deref().ok_or_else(|| anyhow!("tomczak_form requires a preset"))?;
            let m: TomczakFormSfr = match preset {
                "t16" => TomczakFormSfr::t16(),
                "ce" => TomczakFormSfr::ce(),
                "illustris" => TomczakFormSfr::illustris(),
                other => return Err(anyhow!("unknown tomczak_form preset: {other}")),
            };
            Ok(Box::new(m))
        }
        "schreiber_form" => {
            let preset = cfg.preset.as_deref().ok_or_else(|| anyhow!("schreiber_form requires a preset"))?;
            let m: SchreiberFormSfr = match preset {
                "s15" => SchreiberFormSfr::s15(),
                "s16ce" => SchreiberFormSfr::s16ce(),
                other => return Err(anyhow!("unknown schreiber_form preset: {other}")),
            };
            Ok(Box::new(m))
        }
        "double_power_law" => Ok(Box::new(DoublePowerLawSfr)),
        other => Err(anyhow!("unknown sfr model: {other}")),
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

fn build_quenching() -> Box<dyn QuenchingModel> {
    Box::new(Wetzel13::new())
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
        ("moster_form", "override_0") => "Override_0",
        ("moster_form", "override_z") => "Override_z",
        ("behroozi_form", "b18c") => "B18c",
        ("behroozi_form", "b18t") => "B18t",
        ("behroozi_form", "behroozi13" | "behrozi13") => "Behroozi13",
        ("behroozi_form", "lorenzo18") => "Lorenzo18",
        _ => "Unknown",
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

    let smhm = build_smhm(&runfile.smhm)?;
    let sfr = build_sfr(&runfile.sfr)?;
    let gas = build_gas(&runfile.gas, &cosmology)?;
    let stripping = build_stripping(&runfile.stripping)?;
    let quenching = build_quenching();

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
        smhm: Arc::from(smhm),
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
