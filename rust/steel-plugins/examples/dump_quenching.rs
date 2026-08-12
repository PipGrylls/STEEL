//! Dumps `Wetzel13::timescales` over a stellar-mass grid for three host
//! masses, reproducing Paper 2 Figure 6 (the Wetzel+2013 /
//! Fillingham+2016 quenching-delay model). z_infall=0 removes the
//! Cowley+2019 (1+z)^-1.5 scaling so the output is directly comparable
//! to the paper's static plot of tau_q(M*). See
//! `Scripts/Validation/paper2_figures.py`, which drives the equivalent
//! `Functions.py::StarFormation` block for py-as-is/py-corrected, and
//! `docs/PORT_CORRECTIONS.md` A8 for the host-mass-dependence bug this
//! figure surfaced.

use steel_core::quenching::QuenchingModel;
use steel_plugins::Wetzel13;

fn main() {
    let model = Wetzel13::new();
    let host_masses = [10.0, 12.5, 15.0];

    println!("log_sm,host_mass,tau_delay_gyr");
    let mut log_sm = 7.0;
    while log_sm <= 12.0 + 1e-9 {
        for &host in &host_masses {
            let q = model.timescales(log_sm, 0.0, host, 0.0, false);
            println!("{log_sm:.3},{host:.2},{:.10}", q.tau_delay);
        }
        log_sm += 0.02;
    }
}
