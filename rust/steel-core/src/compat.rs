//! Composition validation for plugin sets.
//!
//! The dangerous failures when mixing models are not type errors: they
//! are *silent double-counting* of a physical effect, which yields
//! plausible-looking output. UniverseMachine's bimodal SFR PDF already
//! contains quenching, so combining it with STEEL's `QuenchingModel`
//! quenches twice and no error is raised anywhere.
//!
//! A literal N-by-N model compatibility matrix is deliberately not used:
//! it needs a new row *and* column per plugin and is no stricter than
//! this rule set over declared metadata. Spec section 8.1.
//!
//! **Limitation.** This detects conflicts only along the dimensions
//! enumerated below. It cannot detect a novel incompatibility nobody
//! declared; that is what the planned derived-contract mechanism and
//! property-based cross-validation address (spec section 8.2).

use crate::cosmology::MassDefinition;

/// Stellar IMF a calibration assumes. Mirrors
/// `steel_plugins::harmonise::Imf`; declared here because `steel-core`
/// must not depend on `steel-plugins` (the dependency runs the other
/// way). Keep the two in sync — the round-trip test in
/// `steel-plugins::harmonise` guards the numeric offsets, this copy
/// carries only the tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imf {
    Chabrier,
    Kroupa,
    Salpeter,
    NotApplicable,
}

/// Whether masses carry a factor of `h`. Mirrors
/// `steel_plugins::harmonise::HConvention`; see [`Imf`] on why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HConvention {
    HFree,
    PerH,
}

/// Cosmology a plugin's parameters were fitted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosmologyTag {
    Planck15,
    Planck18,
    Wmap7,
    Wmap9,
}

/// An exclusive physical effect a plugin supplies. Two plugins in one
/// run must not supply the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// Determines M*. Either an `SmhmModel` or a `StellarGrowthModel`,
    /// never both.
    StellarMass,
    /// Suppresses star formation. Held by explicit quenching models *and*
    /// by any model whose SFR prescription already encodes quenching.
    Quenching,
    /// Applies intrinsic scatter to M*.
    Scatter,
    /// Supplies the star-formation rate.
    StarFormationRate,
}

/// What a plugin declares about its own assumptions.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    pub id: &'static str,
    pub imf: Imf,
    pub mass_definition: MassDefinition,
    pub h_convention: HConvention,
    /// `None` means the plugin is cosmology-agnostic and is not checked.
    pub calibrated_cosmology: Option<CosmologyTag>,
    pub provides: &'static [Capability],
}

pub trait DescribedPlugin {
    fn descriptor(&self) -> PluginDescriptor;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Incompatibility {
    DuplicateCapability { capability: Capability, first: &'static str, second: &'static str },
    ImfMismatch { first: &'static str, first_imf: Imf, second: &'static str, second_imf: Imf },
    MassDefinitionMismatch { first: &'static str, second: &'static str },
    HConventionMismatch { first: &'static str, second: &'static str },
    CosmologyMismatch { plugin: &'static str, fitted: CosmologyTag, run: CosmologyTag },
}

impl std::fmt::Display for Incompatibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Incompatibility::DuplicateCapability { capability, first, second } => write!(
                f,
                "'{first}' and '{second}' both supply {capability:?}; the effect would be \
                 applied twice. Select only one, or a variant that does not supply it."
            ),
            Incompatibility::ImfMismatch { first, first_imf, second, second_imf } => write!(
                f,
                "'{first}' assumes {first_imf:?} but '{second}' assumes {second_imf:?}. The \
                 offset is comparable to the signal under comparison; convert one, or select \
                 matching calibrations."
            ),
            Incompatibility::MassDefinitionMismatch { first, second } => write!(
                f,
                "'{first}' and '{second}' use different halo mass definitions; convert via \
                 MassDefinition before combining."
            ),
            Incompatibility::HConventionMismatch { first, second } => write!(
                f,
                "'{first}' and '{second}' disagree on the h convention (Msun vs Msun/h)."
            ),
            Incompatibility::CosmologyMismatch { plugin, fitted, run } => write!(
                f,
                "'{plugin}' was fitted under {fitted:?} but the run uses {run:?}; its \
                 normalisation does not transfer."
            ),
        }
    }
}

/// Validate a plugin set. Returns **every** violation, not just the
/// first, so one run surfaces all problems.
///
/// Callers must treat an `Err` as fatal at startup. A warning in a batch
/// run is indistinguishable from silence.
pub fn validate_composition(
    descriptors: &[PluginDescriptor],
    run_cosmology: CosmologyTag,
) -> Result<(), Vec<Incompatibility>> {
    let mut errors = Vec::new();

    // Rule 1: no duplicated exclusive capability.
    for (i, a) in descriptors.iter().enumerate() {
        for b in &descriptors[i + 1..] {
            for &cap in a.provides {
                if b.provides.contains(&cap) {
                    errors.push(Incompatibility::DuplicateCapability {
                        capability: cap,
                        first: a.id,
                        second: b.id,
                    });
                }
            }
        }
    }

    // Rules 2-4: pairwise agreement on IMF, mass definition, h.
    // Compared against the first descriptor that expresses an opinion,
    // so N plugins give at most N-1 errors rather than N^2.
    if let Some(first_imf) = descriptors.iter().find(|d| d.imf != Imf::NotApplicable) {
        for d in descriptors {
            if d.imf != Imf::NotApplicable && d.imf != first_imf.imf {
                errors.push(Incompatibility::ImfMismatch {
                    first: first_imf.id,
                    first_imf: first_imf.imf,
                    second: d.id,
                    second_imf: d.imf,
                });
            }
        }
    }

    if let Some(first) = descriptors.first() {
        for d in &descriptors[1..] {
            if d.mass_definition != first.mass_definition {
                errors.push(Incompatibility::MassDefinitionMismatch {
                    first: first.id,
                    second: d.id,
                });
            }
            if d.h_convention != first.h_convention {
                errors.push(Incompatibility::HConventionMismatch { first: first.id, second: d.id });
            }
        }
    }

    // Rule 5: declared calibration cosmology must match the run.
    for d in descriptors {
        if let Some(fitted) = d.calibrated_cosmology {
            if fitted != run_cosmology {
                errors.push(Incompatibility::CosmologyMismatch {
                    plugin: d.id,
                    fitted,
                    run: run_cosmology,
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(id: &'static str, provides: &'static [Capability]) -> PluginDescriptor {
        PluginDescriptor {
            id,
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::PerH,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            provides,
        }
    }

    #[test]
    fn a_coherent_composition_passes() {
        let d = [
            base("g19_se", &[Capability::StellarMass, Capability::Scatter]),
            base("ce_sfr", &[Capability::StarFormationRate]),
            base("wetzel13", &[Capability::Quenching]),
        ];
        assert!(validate_composition(&d, CosmologyTag::Planck15).is_ok());
    }

    #[test]
    fn duplicate_quenching_is_rejected() {
        // The live case: UniverseMachine's bimodal SFR PDF already
        // contains quenching, so pairing it with STEEL's QuenchingModel
        // double-counts.
        let d = [
            base("universe_machine", &[Capability::StellarMass, Capability::Quenching]),
            base("wetzel13", &[Capability::Quenching]),
        ];
        let err = validate_composition(&d, CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(
            e,
            Incompatibility::DuplicateCapability { capability: Capability::Quenching, .. }
        )), "{err:?}");
    }

    #[test]
    fn duplicate_stellar_mass_source_is_rejected() {
        let d = [
            base("g19_se", &[Capability::StellarMass]),
            base("emerge", &[Capability::StellarMass]),
        ];
        let err = validate_composition(&d, CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(
            e,
            Incompatibility::DuplicateCapability { capability: Capability::StellarMass, .. }
        )), "{err:?}");
    }

    #[test]
    fn duplicate_scatter_is_rejected() {
        let d = [
            base("g19_se", &[Capability::Scatter]),
            base("emerge", &[Capability::Scatter]),
        ];
        let err = validate_composition(&d, CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(
            e,
            Incompatibility::DuplicateCapability { capability: Capability::Scatter, .. }
        )), "{err:?}");
    }

    #[test]
    fn imf_mismatch_is_rejected() {
        let mut a = base("emerge", &[Capability::StellarMass]);
        a.imf = Imf::Chabrier;
        let mut b = base("steel_ssfr", &[Capability::StarFormationRate]);
        b.imf = Imf::Kroupa;
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(e, Incompatibility::ImfMismatch { .. })), "{err:?}");
    }

    #[test]
    fn mass_definition_mismatch_is_rejected() {
        let mut a = base("a", &[Capability::StellarMass]);
        a.mass_definition = MassDefinition::Vir;
        let mut b = base("b", &[Capability::StarFormationRate]);
        b.mass_definition = MassDefinition::Critical(200.0);
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        assert!(
            err.iter().any(|e| matches!(e, Incompatibility::MassDefinitionMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn h_convention_mismatch_is_rejected() {
        let mut a = base("a", &[Capability::StellarMass]);
        a.h_convention = HConvention::PerH;
        let mut b = base("b", &[Capability::StarFormationRate]);
        b.h_convention = HConvention::HFree;
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        assert!(
            err.iter().any(|e| matches!(e, Incompatibility::HConventionMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn cosmology_mismatch_against_the_run_is_rejected() {
        let mut a = base("fitted_on_wmap7", &[Capability::StellarMass]);
        a.calibrated_cosmology = Some(CosmologyTag::Wmap7);
        let err = validate_composition(&[a], CosmologyTag::Planck15).expect_err("must reject");
        assert!(
            err.iter().any(|e| matches!(e, Incompatibility::CosmologyMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn unspecified_calibration_cosmology_is_accepted() {
        let mut a = base("cosmology_agnostic", &[Capability::StellarMass]);
        a.calibrated_cosmology = None;
        assert!(validate_composition(&[a], CosmologyTag::Planck15).is_ok());
    }

    #[test]
    fn all_violations_are_reported_not_just_the_first() {
        let mut a = base("a", &[Capability::StellarMass]);
        a.imf = Imf::Chabrier;
        let mut b = base("b", &[Capability::StellarMass]);
        b.imf = Imf::Salpeter;
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        // Duplicate StellarMass *and* an IMF mismatch.
        assert!(err.len() >= 2, "expected multiple violations, got {err:?}");
    }
}
