//! JSON-in/JSON-out wrapper over `steel_plugins::harmonise`.
//!
//! The research apparatus performs no unit or definition conversion of its
//! own; it shells out here, so the conversion logic stays in one tested
//! place rather than being reimplemented in Python.

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;
use serde_json::json;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_plugins::harmonise::{convert_mass_definition, DuttonMaccio14, HConvention, Imf};
use steel_plugins::Planck15;

#[derive(Deserialize)]
struct Endpoint {
    mass_def: Option<String>,
    imf: Option<String>,
    h_convention: String,
}

#[derive(Deserialize)]
struct Request {
    op: String,
    log_m: f64,
    from: Endpoint,
    to: Endpoint,
    #[serde(default)]
    z: f64,
}

fn parse_mass_def(s: &str) -> Result<MassDefinition> {
    match s {
        "Mvir" => Ok(MassDefinition::Vir),
        _ if s.ends_with('c') => s[1..s.len() - 1]
            .parse()
            .map(MassDefinition::Critical)
            .map_err(|_| anyhow!("bad critical overdensity: {s}")),
        _ if s.ends_with('m') => s[1..s.len() - 1]
            .parse()
            .map(MassDefinition::Mean)
            .map_err(|_| anyhow!("bad mean overdensity: {s}")),
        // "unknown" lands here too: refuse rather than guess.
        _ => bail!("unrecognised mass definition: {s}"),
    }
}

fn parse_h(s: &str) -> Result<HConvention> {
    match s {
        "h_free" => Ok(HConvention::HFree),
        "per_h" => Ok(HConvention::PerH),
        _ => bail!("unrecognised h convention: {s}"),
    }
}

fn parse_imf(s: &str) -> Result<Imf> {
    match s {
        "chabrier" => Ok(Imf::Chabrier),
        "kroupa" => Ok(Imf::Kroupa),
        "salpeter" => Ok(Imf::Salpeter),
        _ => bail!("unrecognised IMF: {s}"),
    }
}

fn main() -> Result<()> {
    let req: Request = serde_json::from_reader(std::io::stdin())?;
    let cosmo = Planck15::new();
    let h = cosmo.h();
    let mut path: Vec<String> = Vec::new();

    let from_h = parse_h(&req.from.h_convention)?;
    let to_h = parse_h(&req.to.h_convention)?;

    let log_m = match req.op.as_str() {
        "convert_mass" => {
            // `convert_mass_definition` works in Msun/h, matching
            // `Cosmology::m_to_r`. `to_h_free` followed by `from_h_free`
            // under the *same* convention is always an identity (they are
            // exact inverses), so the only convention that actually needs
            // adjusting on the way in is `HFree`, which must pick up a
            // factor of h to become per-h.
            let as_per_h = match from_h {
                HConvention::HFree => {
                    path.push("h_free->per_h".into());
                    HConvention::PerH.from_h_free(req.log_m, h)
                }
                _ => req.log_m,
            };
            let from_def = parse_mass_def(
                req.from.mass_def.as_deref().ok_or_else(|| anyhow!("from.mass_def required"))?,
            )?;
            let to_def = parse_mass_def(
                req.to.mass_def.as_deref().ok_or_else(|| anyhow!("to.mass_def required"))?,
            )?;
            let converted =
                convert_mass_definition(as_per_h, from_def, to_def, req.z, &cosmo, &DuttonMaccio14);
            path.push(format!(
                "{}->{} (DuttonMaccio14, NFW)",
                req.from.mass_def.as_deref().unwrap(),
                req.to.mass_def.as_deref().unwrap()
            ));
            match to_h {
                HConvention::HFree => {
                    path.push("per_h->h_free".into());
                    HConvention::PerH.to_h_free(converted, h)
                }
                _ => converted,
            }
        }
        "convert_stellar" => {
            let from_imf = parse_imf(
                req.from.imf.as_deref().ok_or_else(|| anyhow!("from.imf required"))?,
            )?;
            let to_imf =
                parse_imf(req.to.imf.as_deref().ok_or_else(|| anyhow!("to.imf required"))?)?;
            let offset = from_imf.log_offset_to(to_imf);
            path.push(format!("imf {from_imf:?}->{to_imf:?} ({offset:+.3} dex)"));
            let h_free = from_h.to_h_free(req.log_m, h);
            if from_h != to_h {
                path.push(format!("{}->{}", req.from.h_convention, req.to.h_convention));
            }
            to_h.from_h_free(h_free, h) + offset
        }
        other => bail!("unrecognised op: {other}"),
    };

    println!("{}", json!({"log_m": log_m, "path": path}));
    Ok(())
}
