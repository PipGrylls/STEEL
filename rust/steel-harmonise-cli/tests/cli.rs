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
