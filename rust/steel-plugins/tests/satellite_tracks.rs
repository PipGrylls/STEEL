//! A satellite's `own_track` must be its own pre-infall central
//! history, not the host's. Spec §5.

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::SmhmModel;
use steel_plugins::{Planck15, VandenBosch14};

/// Records the `own_track` head mass it was called with.
struct SpySmhm {
    seen: std::sync::Mutex<Vec<f64>>,
}

impl SmhmModel for SpySmhm {
    fn stellar_mass(
        &self,
        log_dm: f64,
        _z: f64,
        ctx: &AccretionContext<'_>,
        _rng: Option<&mut dyn rand::RngCore>,
    ) -> f64 {
        self.seen.lock().unwrap().push(ctx.own_track.log_mass[0]);
        log_dm - 2.0
    }
}

#[test]
fn satellite_own_track_head_equals_its_infall_mass() {
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);
    let z_infall = 1.5;
    let log_m_sub = 11.4;

    let own = growth.growth_history(log_m_sub, z_infall);
    let host = growth.growth_history(13.8, 0.0);
    let ctx = AccretionContext::satellite(&own, &host, z_infall, &cosmo, MassDefinition::Vir);

    // own_track starts at the subhalo's own infall mass...
    assert!((ctx.own_track.log_mass[0] - log_m_sub).abs() < 1e-3);
    // ...and is distinct from the host's.
    assert!((ctx.host_track.expect("host").log_mass[0] - 13.8).abs() < 1e-3);

    let spy = SpySmhm { seen: std::sync::Mutex::new(Vec::new()) };
    let _ = spy.stellar_mass(log_m_sub, z_infall, &ctx, None);
    assert!((spy.seen.lock().unwrap()[0] - log_m_sub).abs() < 1e-3);
}
