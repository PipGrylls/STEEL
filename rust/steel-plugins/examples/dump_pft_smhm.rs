//! Dumps the SMHM-parameter sensitivity sweep behind Paper 3 Fig. 3 /
//! Table 2: each of the Moster-form coefficients (M, N, beta, gamma)
//! is perturbed one at a time off the G19_SE (PyMorph) baseline, both
//! its z=0.1 value and its z-evolution term, reproducing the panel
//! structure "8 panels: {M,N,beta,gamma} x {z=0.1-altered,
//! z-evolution-altered}". See Scripts/Validation/pft_smhm_sensitivity.py
//! for the Python side and the exact deltas (Table 2 of arXiv:2001.06017).

use steel_core::smhm::SmhmModel;
use steel_plugins::MosterFormSmhm;

fn base() -> MosterFormSmhm {
    MosterFormSmhm::g19_se(true)
}

fn main() {
    let log_dm: Vec<f64> = {
        let mut v = vec![];
        let mut x = 10.5;
        while x <= 15.0 + 1e-9 {
            v.push(x);
            x += 0.05;
        }
        v
    };

    println!("panel,variant,z,log_dm,log_sm");

    // (panel name, z0.1 delta applied to the *_10 field, z-evo delta applied to the *_11 field)
    // Deltas match Functions.py's PFT branch (M_PFT1/N_PFT1/b_PFT1/g_PFT1
    // and the _PFT2/_PFT3 pairs) exactly, not Paper 3 Table 2's printed
    // N alt value (+0.04) -- the code uses +0.004; the code is the
    // validation target here, not the paper text.
    let specs: Vec<(&str, f64, f64)> = vec![
        ("M", -0.25, 0.1),
        ("N", 0.004, 0.007),
        ("beta", -0.3, 0.3),
        ("gamma", 0.06, 0.2),
    ];

    for (panel, d10, d11) in specs {
        let variants: Vec<(&str, MosterFormSmhm)> = vec![
            ("baseline", base()),
            ("alt_z0.1", perturb10(panel, d10)),
            ("zevo_plus", perturb11(panel, d11)),
            ("zevo_minus", perturb11(panel, -d11)),
        ];
        for (variant, smhm) in variants {
            for &z in &[0.1, 2.0] {
                for &dm in &log_dm {
                    println!("{panel},{variant},{z},{dm:.3},{:.6}", smhm.stellar_mass(dm, z, None));
                }
            }
        }
    }
}

fn perturb10(panel: &str, delta: f64) -> MosterFormSmhm {
    let mut s = base();
    match panel {
        "M" => s.m10 += delta,
        "N" => s.shmnorm10 += delta,
        "beta" => s.beta10 += delta,
        "gamma" => s.gamma10 += delta,
        _ => unreachable!(),
    }
    s
}

fn perturb11(panel: &str, delta: f64) -> MosterFormSmhm {
    let mut s = base();
    match panel {
        "M" => s.m11 += delta,
        "N" => s.shmnorm11 += delta,
        "beta" => s.beta11 += delta,
        "gamma" => s.gamma11 += delta,
        _ => unreachable!(),
    }
    s
}
