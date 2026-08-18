//! A runfile requesting an incompatible plugin set must fail at startup
//! with an actionable message, not run and produce wrong numbers.

use steel_core::compat::{
    validate_composition, Capability, CosmologyTag, HConvention, Imf, PluginDescriptor,
};
use steel_core::cosmology::MassDefinition;

#[test]
fn message_names_both_plugins_and_the_duplicated_effect() {
    let um = PluginDescriptor {
        id: "universe_machine",
        imf: Imf::Chabrier,
        mass_definition: MassDefinition::Vir,
        h_convention: HConvention::PerH,
        calibrated_cosmology: Some(CosmologyTag::Planck15),
        provides: &[Capability::StellarMass, Capability::Quenching],
    };
    let wetzel = PluginDescriptor {
        id: "wetzel13",
        provides: &[Capability::Quenching],
        ..um.clone()
    };
    let errs = validate_composition(&[um, wetzel], CosmologyTag::Planck15).expect_err("reject");
    let text = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("universe_machine"), "{text}");
    assert!(text.contains("wetzel13"), "{text}");
    assert!(text.contains("twice"), "message should say the effect is applied twice: {text}");
}
