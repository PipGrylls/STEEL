use std::io::Write;
use std::process::{Command, Stdio};

fn run(input: &str) -> serde_json::Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_steel-harmonise-cli"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn cli");
    child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "cli failed: {}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("cli emitted valid json")
}

#[test]
fn converts_m500c_to_virial_and_reports_the_path() {
    let v = run(r#"{"op":"convert_mass","log_m":14.0,
        "from":{"mass_def":"M500c","h_convention":"h_free"},
        "to":{"mass_def":"Mvir","h_convention":"per_h"},"z":0.1}"#);
    let log_m = v["log_m"].as_f64().unwrap();
    assert!(log_m > 14.0 && log_m < 15.0, "got {log_m}");
    assert!(v["path"].as_array().unwrap().len() >= 2, "path must record each step");
}

#[test]
fn rejects_an_unknown_mass_definition() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_steel-harmonise-cli"))
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    child.stdin.as_mut().unwrap()
        .write_all(br#"{"op":"convert_mass","log_m":14.0,
            "from":{"mass_def":"unknown","h_convention":"h_free"},
            "to":{"mass_def":"Mvir","h_convention":"per_h"},"z":0.1}"#).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success(), "unknown definition must be an error, not a guess");
}

#[test]
fn converts_stellar_mass_between_imfs() {
    let v = run(r#"{"op":"convert_stellar","log_m":10.0,
        "from":{"imf":"salpeter","h_convention":"h_free"},
        "to":{"imf":"chabrier","h_convention":"h_free"}}"#);
    // Salpeter masses sit ~0.24 dex above Chabrier, so the offset is negative.
    assert!((v["log_m"].as_f64().unwrap() - 9.76).abs() < 1e-6);
}

/// `convert_mass` with `to.h_convention: "h_free"` exercises the egress
/// branch (`HConvention::PerH.to_h_free`) that no other test reaches: the
/// existing mass test only ever lands on `per_h`. Round-tripping
/// h_free->per_h and back per_h->h_free, with the mass definitions
/// reversed, is the strongest check here because it does not require
/// knowing `log10(h)` up front -- a swapped `to_h_free`/`from_h_free` call
/// on either leg would fail to recover the original input.
#[test]
fn convert_mass_round_trips_through_both_h_conventions_and_records_the_return_leg() {
    let forward = run(r#"{"op":"convert_mass","log_m":14.0,
        "from":{"mass_def":"M500c","h_convention":"h_free"},
        "to":{"mass_def":"Mvir","h_convention":"per_h"},"z":0.1}"#);
    let intermediate = forward["log_m"].as_f64().unwrap();

    let backward = run(&format!(
        r#"{{"op":"convert_mass","log_m":{intermediate},
        "from":{{"mass_def":"Mvir","h_convention":"per_h"}},
        "to":{{"mass_def":"M500c","h_convention":"h_free"}},"z":0.1}}"#
    ));
    let recovered = backward["log_m"].as_f64().unwrap();
    assert!(
        (recovered - 14.0).abs() < 1e-9,
        "round trip should recover the original 14.0, got {recovered}"
    );

    let path = backward["path"].as_array().unwrap();
    assert!(
        path.iter().any(|p| p.as_str() == Some("per_h->h_free")),
        "return leg must record the per_h->h_free conversion, got {path:?}"
    );
}

/// `convert_stellar` with `from.h_convention != to.h_convention` exercises
/// the cross-convention branch that no other test reaches: the existing
/// stellar test uses `h_free` on both sides.
///
/// Reasoned expectation, not read off a run: converting a value expressed
/// in Msun (`h_free`) into Msun/h (`per_h`) means the *numeric* log10 value
/// gains a term of `log10(h)` (per `HConvention::from_h_free`, which adds
/// `h.log10()`). Planck15's h = 0.6774 < 1, so `log10(h) < 0` -- expressing
/// the same physical mass in Msun/h therefore reads as a *smaller* number
/// than in Msun. So converting h_free -> per_h must yield a result exactly
/// `log10(h)` (a negative quantity) below the same-convention (h_free ->
/// h_free) result, with the IMF held fixed so only the h effect is
/// isolated.
#[test]
fn convert_stellar_mass_across_h_conventions_shifts_by_log10_h() {
    let same_convention = run(r#"{"op":"convert_stellar","log_m":10.0,
        "from":{"imf":"chabrier","h_convention":"h_free"},
        "to":{"imf":"chabrier","h_convention":"h_free"}}"#);
    let cross_convention = run(r#"{"op":"convert_stellar","log_m":10.0,
        "from":{"imf":"chabrier","h_convention":"h_free"},
        "to":{"imf":"chabrier","h_convention":"per_h"}}"#);

    let same_m = same_convention["log_m"].as_f64().unwrap();
    let cross_m = cross_convention["log_m"].as_f64().unwrap();

    let h: f64 = 0.6774; // Planck15's h (H0 = 67.74 km/s/Mpc).
    let log10_h = h.log10();
    assert!(log10_h < 0.0, "sanity: h < 1 so log10(h) must be negative");

    assert!(
        (cross_m - (same_m + log10_h)).abs() < 1e-9,
        "expected cross-convention result {same_m} + log10(h) = {}, got {cross_m}",
        same_m + log10_h
    );

    let path = cross_convention["path"].as_array().unwrap();
    assert!(
        path.iter().any(|p| p.as_str() == Some("h_free->per_h")),
        "cross-convention path must record the h_convention change, got {path:?}"
    );
}
