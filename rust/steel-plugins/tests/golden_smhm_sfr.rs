//! Refactor guard. These values were captured from the pre-
//! `AccretionContext` signatures and must remain bit-identical through
//! the trait widening. A change here means the widening was not inert.

use rand::rngs::StdRng;
use rand::SeedableRng;
use steel_core::{SfrModel, SmhmModel};
use steel_plugins::{MosterFormSmhm, RodriguezPuebla17, TomczakFormSfr};

const LOG_DM: [f64; 5] = [10.0, 11.0, 12.0, 13.0, 14.0];
const Z: [f64; 4] = [0.1, 0.5, 1.0, 3.0];
const LOG_SM: [f64; 4] = [9.0, 10.0, 10.7, 11.5];

/// Prints the current values as Rust literals. Run with
/// `--ignored --nocapture` to regenerate the tables below.
#[test]
#[ignore]
fn print_golden_values() {
    let smhm: Vec<(&str, Box<dyn SmhmModel>)> = vec![
        ("g19_se", Box::new(MosterFormSmhm::g19_se(true))),
        ("moster13", Box::new(MosterFormSmhm::moster13(true))),
        ("rp17", Box::new(RodriguezPuebla17)),
    ];
    for (name, m) in &smhm {
        for &dm in &LOG_DM {
            for &z in &Z {
                println!("{name} {dm} {z} {:.17e}", m.stellar_mass(dm, z, None));
            }
        }
    }
    let sfr: Vec<(&str, Box<dyn SfrModel>)> = vec![
        ("ce", Box::new(TomczakFormSfr::ce())),
        ("t16", Box::new(TomczakFormSfr::t16())),
    ];
    for (name, s) in &sfr {
        for &sm in &LOG_SM {
            for &z in &Z {
                println!("{name} {sm} {z} {:.17e}", s.log_sfr(sm, z));
            }
        }
    }
}

/// `(model, log_dm, z, expected_log_sm)` — captured from Step 2.
/// Replace every `f64::NAN` with the printed value; NAN never compares
/// equal, so an unedited table fails loudly.
const EXPECTED_SMHM: &[(&str, f64, f64, f64)] = &[
    ("g19_se", 10.0, 0.1, 5.65107622682445054e0),
    ("g19_se", 10.0, 0.5, 5.72957338292011897e0),
    ("g19_se", 10.0, 1.0, 5.81216680932306673e0),
    ("g19_se", 10.0, 3.0, 5.97789319830740062e0),
    ("g19_se", 11.0, 0.1, 8.28586379115783878e0),
    ("g19_se", 11.0, 0.5, 8.18074736548787662e0),
    ("g19_se", 11.0, 1.0, 8.13672859681005001e0),
    ("g19_se", 11.0, 3.0, 8.11218673790300748e0),
    ("g19_se", 12.0, 0.1, 1.05390759945507462e1),
    ("g19_se", 12.0, 0.5, 1.04083131422103712e1),
    ("g19_se", 12.0, 1.0, 1.03045605504332620e1),
    ("g19_se", 12.0, 3.0, 1.01494918114138262e1),
    ("g19_se", 13.0, 0.1, 1.12322710917268029e1),
    ("g19_se", 13.0, 0.5, 1.12484797094125550e1),
    ("g19_se", 13.0, 1.0, 1.12532468918288195e1),
    ("g19_se", 13.0, 3.0, 1.12440663862872743e1),
    ("g19_se", 14.0, 0.1, 1.17022663923343266e1),
    ("g19_se", 14.0, 0.5, 1.17146846055103051e1),
    ("g19_se", 14.0, 1.0, 1.17202715000095665e1),
    ("g19_se", 14.0, 3.0, 1.17220108613807295e1),
    ("moster13", 10.0, 0.1, 6.60763300446432389e0),
    ("moster13", 10.0, 0.5, 6.54164629676192533e0),
    ("moster13", 10.0, 1.0, 6.55137949923116647e0),
    ("moster13", 10.0, 3.0, 6.63959257335129571e0),
    ("moster13", 11.0, 0.1, 8.88997730543038500e0),
    ("moster13", 11.0, 0.5, 8.63554401039497499e0),
    ("moster13", 11.0, 1.0, 8.51069054399138558e0),
    ("moster13", 11.0, 3.0, 8.39438881998108499e0),
    ("moster13", 12.0, 0.1, 1.05248600552445790e1),
    ("moster13", 12.0, 0.5, 1.04313184378455190e1),
    ("moster13", 12.0, 1.0, 1.03093319946132009e1),
    ("moster13", 12.0, 3.0, 1.00864445072316329e1),
    ("moster13", 13.0, 0.1, 1.09861735154655076e1),
    ("moster13", 13.0, 0.5, 1.09979653544374365e1),
    ("moster13", 13.0, 1.0, 1.10137830235331986e1),
    ("moster13", 13.0, 3.0, 1.10211608335911979e1),
    ("moster13", 14.0, 0.1, 1.13495496818973365e1),
    ("moster13", 14.0, 0.5, 1.12864422040347208e1),
    ("moster13", 14.0, 1.0, 1.12575446654548390e1),
    ("moster13", 14.0, 3.0, 1.12250282337751308e1),
    ("rp17", 10.0, 0.1, 6.53335032030466678e0),
    ("rp17", 10.0, 0.5, 6.53149607748220618e0),
    ("rp17", 10.0, 1.0, 6.53717490204728069e0),
    ("rp17", 10.0, 3.0, 6.68315239177107134e0),
    ("rp17", 11.0, 0.1, 8.51399216622242605e0),
    ("rp17", 11.0, 0.5, 8.46552007575204968e0),
    ("rp17", 11.0, 1.0, 8.38421828014757153e0),
    ("rp17", 11.0, 3.0, 8.33556380730482793e0),
    ("rp17", 12.0, 0.1, 1.04360169448585527e1),
    ("rp17", 12.0, 0.5, 1.04219820770165175e1),
    ("rp17", 12.0, 1.0, 1.03411030837722695e1),
    ("rp17", 12.0, 3.0, 9.94849023014032774e0),
    ("rp17", 13.0, 0.1, 1.10082956229485163e1),
    ("rp17", 13.0, 0.5, 1.10299987709630560e1),
    ("rp17", 13.0, 1.0, 1.10436249274567828e1),
    ("rp17", 13.0, 3.0, 1.08264094225965888e1),
    ("rp17", 14.0, 0.1, 1.13685222507630126e1),
    ("rp17", 14.0, 0.5, 1.13953428915961990e1),
    ("rp17", 14.0, 1.0, 1.14117765082549276e1),
    ("rp17", 14.0, 3.0, 1.10628691525386653e1),
];

const EXPECTED_SFR: &[(&str, f64, f64, f64)] = &[
    ("ce", 9.0, 0.1, -1.05947877328819495e0),
    ("ce", 9.0, 0.5, -8.92614564819538936e-1),
    ("ce", 9.0, 1.0, -6.65854532796023468e-1),
    ("ce", 9.0, 3.0, 2.48324248677588066e-1),
    ("ce", 10.0, 0.1, 1.14123888190810763e-1),
    ("ce", 10.0, 0.5, 2.97186744664939551e-1),
    ("ce", 10.0, 1.0, 5.01976206291000038e-1),
    ("ce", 10.0, 3.0, 1.22495660820147934e0),
    ("ce", 10.7, 0.1, 5.80280985701329244e-1),
    ("ce", 10.7, 0.5, 8.95423727564605754e-1),
    ("ce", 10.7, 1.0, 1.18030961666724932e0),
    ("ce", 10.7, 3.0, 1.83381426881151999e0),
    ("ce", 11.5, 0.1, 7.04953300748032641e-1),
    ("ce", 11.5, 0.5, 1.12501170953644913e0),
    ("ce", 11.5, 1.0, 1.54367560647083457e0),
    ("ce", 11.5, 3.0, 2.28053960540178347e0),
    ("t16", 9.0, 0.1, -2.05239810935281319e-1),
    ("t16", 9.0, 0.5, -1.52130563265773233e-2),
    ("t16", 9.0, 1.0, 1.54837851433982543e-1),
    ("t16", 9.0, 3.0, 4.80702104727741109e-1),
    ("t16", 10.0, 0.1, 2.39983034166371428e-1),
    ("t16", 10.0, 0.5, 6.05722742685575244e-1),
    ("t16", 10.0, 1.0, 9.56852868330342332e-1),
    ("t16", 10.0, 3.0, 1.53679513511869059e0),
    ("t16", 10.7, 0.1, 2.97051110356002290e-1),
    ("t16", 10.7, 0.5, 7.13014435319052176e-1),
    ("t16", 10.7, 1.0, 1.15598104356095388e0),
    ("t16", 10.7, 3.0, 2.08186537647925807e0),
    ("t16", 11.5, 0.1, 3.07692510316519230e-1),
    ("t16", 11.5, 0.5, 7.34516072019639088e-1),
    ("t16", 11.5, 1.0, 1.20186754234791793e0),
    ("t16", 11.5, 3.0, 2.32775139407422715e0),
];

fn smhm_by_name(name: &str) -> Box<dyn SmhmModel> {
    match name {
        "g19_se" => Box::new(MosterFormSmhm::g19_se(true)),
        "moster13" => Box::new(MosterFormSmhm::moster13(true)),
        "rp17" => Box::new(RodriguezPuebla17),
        other => panic!("unknown smhm preset in golden table: {other}"),
    }
}

fn sfr_by_name(name: &str) -> Box<dyn SfrModel> {
    match name {
        "ce" => Box::new(TomczakFormSfr::ce()),
        "t16" => Box::new(TomczakFormSfr::t16()),
        other => panic!("unknown sfr preset in golden table: {other}"),
    }
}

#[test]
fn existing_smhm_plugins_are_bit_identical_to_golden() {
    for &(name, dm, z, expected) in EXPECTED_SMHM {
        let got = smhm_by_name(name).stellar_mass(dm, z, None);
        assert_eq!(got.to_bits(), expected.to_bits(), "{name} at dm={dm} z={z}: {got} != {expected}");
    }
}

#[test]
fn existing_sfr_plugins_are_bit_identical_to_golden() {
    for &(name, sm, z, expected) in EXPECTED_SFR {
        let got = sfr_by_name(name).log_sfr(sm, z);
        assert_eq!(got.to_bits(), expected.to_bits(), "{name} at sm={sm} z={z}: {got} != {expected}");
    }
}

#[test]
fn seeded_scatter_is_reproducible() {
    let m = MosterFormSmhm::g19_se(true);
    let mut a = StdRng::seed_from_u64(20260817);
    let mut b = StdRng::seed_from_u64(20260817);
    assert_eq!(
        m.stellar_mass(12.0, 0.1, Some(&mut a)).to_bits(),
        m.stellar_mass(12.0, 0.1, Some(&mut b)).to_bits()
    );
}
