//! Shared run context and the top-level orchestrator: the Rust
//! equivalent of `STEEL.py`'s module-level grid setup plus
//! `OneRealization`'s triple loop over redshift step / host-halo bin /
//! subhalo bin.
//!
//! Scope note: this produces the surviving-satellite stellar mass
//! function (`Functions.py::SaveData_3`'s `Figure3` output —
//! `Surviving_Sat_SMF_Weighting_Totals`), the single output the thesis
//! treats as STEEL's headline deliverable, and the loop machinery that
//! produces it (weight-list construction, abundance matching, optional
//! baryonic evolution) is the same machinery every other Python output
//! (`Sat_SMHM`, `Accretion_History`, `Pair_Frac`, the high-z SMF, the
//! sSFR distribution, ...) is built from. Wiring up the other ~13
//! accumulator arrays from `Functions.py`'s `SaveData_*` family is a
//! mechanical extension of this same loop, not a design gap — left for
//! a follow-up rather than attempted here.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::baryonic::{BaryonicPipeline, SatelliteState, Timeline};
use crate::cosmology::Cosmology;
use crate::halo_growth::HaloGrowthModel;
use crate::hmf::HaloMassFunctionModel;
use crate::merger_time::MergerTimescaleModel;
use crate::numerics::digitize;
use crate::shmf::SubhaloMassFunctionModel;
use crate::smhm::SmhmModel;
use crate::stripping::HaloStrippingModel;

/// Values shared read-only across every plugin call in a run.
pub struct ModelContext {
    pub cosmology: Arc<dyn Cosmology>,
    /// Seed for the run's random number generator. Threaded explicitly
    /// rather than an ambient/reseeded-per-call global (unlike the Python
    /// original's `np.random.seed(...)` inside `DarkMatterToStellarMass`),
    /// so runs are reproducible.
    pub rng_seed: u64,
}

/// The independently-injected plugins for one STEEL run, plus the single
/// composed [`BaryonicPipeline`] for per-timestep satellite evolution.
///
/// This is the dependency-injection container: every field is a trait
/// object chosen at startup (from a TOML runfile, via `steel-cli`'s
/// plugin registry) and never branched on again — the orchestrator only
/// ever calls the trait methods.
pub struct Simulation {
    pub context: ModelContext,
    pub halo_growth: Arc<dyn HaloGrowthModel>,
    pub hmf: Arc<dyn HaloMassFunctionModel>,
    pub shmf: Arc<dyn SubhaloMassFunctionModel>,
    pub merger_time: Arc<dyn MergerTimescaleModel>,
    pub halo_stripping: Option<Arc<dyn HaloStrippingModel>>,
    pub smhm: Arc<dyn SmhmModel>,
    pub baryonic: BaryonicPipeline,
}

/// Run-specific parameters — the Rust equivalent of `STEEL.py`'s
/// module-level constants (`AnalyticHaloMass_min`, etc.) and
/// `OneRealization`'s `Factor_Stripping_SF` tuple, both replaced by a
/// TOML runfile at the `steel-cli` layer.
pub struct RunConfig {
    /// `AnalyticHaloMass_min` — minimum host halo mass, log10 Msun (h-free).
    pub log_m_min: f64,
    /// `AnalyticHaloMass_max`.
    pub log_m_max: f64,
    /// `AnalyticHaloBin` — grid spacing for both host and subhalo mass axes.
    pub log_m_bin: f64,
    /// `Min_Corr` — how far below `log_m_min` the subhalo mass grid extends.
    pub sat_min_offset: f64,
    /// Reference epoch STEEL matches to observational data at
    /// (`Functions.py::Get_HM_History` trims the growth history to
    /// `z >= 0.1` to match SDSS) rather than literal `z=0`.
    pub z_reference_min: f64,
    /// `SF` in `Factor_Stripping_SF`.
    pub star_formation: bool,
    /// `Stripping` in `Factor_Stripping_SF`.
    pub stellar_stripping: bool,
    /// `N` — abundance-matching scatter realizations per subhalo bin.
    pub n_realizations: usize,
    /// `SatM_min`/`SatM_max`/`SatBin` for the output stellar-mass grid.
    pub sat_sm_min: f64,
    pub sat_sm_max: f64,
    pub sat_sm_bin: f64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            log_m_min: 11.0,
            log_m_max: 16.6,
            log_m_bin: 0.1,
            sat_min_offset: -1.0,
            z_reference_min: 0.1,
            star_formation: false,
            stellar_stripping: false,
            n_realizations: 5,
            sat_sm_min: 9.0,
            sat_sm_max: 13.0,
            sat_sm_bin: 0.1,
        }
    }
}

/// The primary run output: the surviving-satellite stellar mass
/// function at the reference epoch (`RunConfig::z_reference_min`),
/// matching `Functions.py::SaveData_3`'s `Figure3` fields.
pub struct RunOutput {
    /// Redshift steps, `z[0] ~= z_reference_min` increasing to `z_max`.
    pub z: Vec<f64>,
    /// Host halo mass \[log10 Msun/h\] at each `(redshift, host bin)`,
    /// `host_halo_mass[i][j]`.
    pub host_halo_mass: Vec<Vec<f64>>,
    /// Left edges of the satellite stellar-mass histogram bins
    /// \[log10 Msun\].
    pub sat_sm_range: Vec<f64>,
    /// Surviving-satellite stellar mass function at the reference
    /// epoch, `dn/dlog10(M*)` \[Mpc^-3 dex^-1\], one entry per
    /// `sat_sm_range` bin.
    pub surviving_sat_smf: Vec<f64>,
}

/// Index of the uniform histogram bin `x` falls into over
/// `[min, min + n_bins*bin_width)`, or `None` if `x` is outside that
/// range — matches `fast_histogram.histogram1d`'s convention (drop
/// out-of-range values) rather than `numpy.digitize`'s (clip to an edge
/// index), which is what `STEEL.py` actually uses for this binning.
fn histogram_bin_index(x: f64, min: f64, bin_width: f64, n_bins: usize) -> Option<usize> {
    if x < min {
        return None;
    }
    let idx = ((x - min) / bin_width) as usize;
    if idx < n_bins {
        Some(idx)
    } else {
        None
    }
}

impl Simulation {
    /// Run the statistical dark-matter-accretion-history pipeline and
    /// return the surviving-satellite SMF (see the module doc comment
    /// for what's in and out of scope for this port).
    pub fn run(&self, config: &RunConfig) -> RunOutput {
        let h = self.context.cosmology.h();
        let log_h = h.log10();

        // Host halo mass grid at z=0 [log10 Msun/h] (`AnalyticHaloMass`).
        let n_host = ((config.log_m_max - config.log_m_min) / config.log_m_bin).round() as usize;
        let host_mass_z0: Vec<f64> =
            (0..n_host).map(|j| config.log_m_min + log_h + j as f64 * config.log_m_bin).collect();

        // Growth history for every host bin (`Get_HM_History`); every
        // bin shares the same redshift grid since z0=0 throughout.
        let raw_z = self.halo_growth.redshift_grid(0.0);
        let n_z_raw = raw_z.len();
        let mut raw_host_mass = vec![vec![0.0_f64; n_host]; n_z_raw]; // [i][j]
        for (j, &log_m0) in host_mass_z0.iter().enumerate() {
            let track = self.halo_growth.growth_history(log_m0, 0.0);
            for (i, row) in raw_host_mass.iter_mut().enumerate().take(n_z_raw) {
                row[j] = track.log_mass[i];
            }
        }

        // Trim to the reference epoch, matching `Get_HM_History`'s
        // z>=0.1 cut (`raw_z` is increasing, so this drops the leading
        // entries below the cut).
        let cut = digitize(config.z_reference_min, &raw_z);
        let z: Vec<f64> = raw_z[cut..].to_vec();
        let host_mass: Vec<Vec<f64>> = raw_host_mass[cut..].to_vec();
        let n_z = z.len();

        // Host mass bin widths, accounting for bins converging at high z
        // (`AvaHaloMassBins`).
        let mut host_mass_bins = vec![vec![0.0_f64; n_host]; n_z];
        for i in 0..n_z {
            for j in 0..n_host.saturating_sub(1) {
                host_mass_bins[i][j] = host_mass[i][j + 1] - host_mass[i][j];
            }
            if n_host >= 2 {
                host_mass_bins[i][n_host - 1] = host_mass_bins[i][n_host - 2];
            }
        }

        // Subhalo mass grid (`SatHaloMass`).
        let sat_min = config.log_m_min + config.sat_min_offset + log_h;
        let sat_max = config.log_m_max - 0.1 + log_h;
        let n_sat = ((sat_max - sat_min) / config.log_m_bin).round() as usize;
        let sat_mass: Vec<f64> = (0..n_sat).map(|k| sat_min + k as f64 * config.log_m_bin).collect();

        // Unevolved SHMF accreted between consecutive redshift steps
        // (`SHMFs_Entering`), shape (n_z-1, n_host, n_sat).
        let n_z_pairs = n_z.saturating_sub(1);
        let mut shmf_entering = vec![vec![vec![0.0_f64; n_sat]; n_host]; n_z_pairs];
        for (i, entering_i) in shmf_entering.iter_mut().enumerate() {
            for (j, entering_ij) in entering_i.iter_mut().enumerate() {
                for (k, entering_ijk) in entering_ij.iter_mut().enumerate() {
                    let x_now = 10f64.powf(sat_mass[k] - host_mass[i][j]);
                    let x_next = 10f64.powf(sat_mass[k] - host_mass[i + 1][j]);
                    *entering_ijk = self.shmf.dn_dlog10x(x_now) - self.shmf.dn_dlog10x(x_next);
                }
            }
        }

        // Output stellar-mass grid.
        let n_sm = ((config.sat_sm_max - config.sat_sm_min) / config.sat_sm_bin).round() as usize;
        let sat_sm_range: Vec<f64> =
            (0..n_sm).map(|b| config.sat_sm_min + b as f64 * config.sat_sm_bin).collect();
        let mut surviving_sat_smf = vec![0.0_f64; n_sm];

        // Cosmic time bookkeeping (`Times`, `Time_To_0`).
        let times: Vec<f64> = z.iter().map(|&zi| self.context.cosmology.age(zi)).collect();
        let time_to_0: Vec<f64> = times.iter().map(|&t| times[0] - t).collect();

        let mut rng = StdRng::seed_from_u64(self.context.rng_seed);
        let h3 = h * h * h;

        for i in (0..n_z_pairs).rev() {
            let ttz0 = time_to_0[i];
            for j in 0..n_host {
                for k in 0..n_sat {
                    if host_mass[i][j] <= sat_mass[k] {
                        continue; // host must outmass its subhalo
                    }

                    let tdyf = self.merger_time.infall_time(
                        host_mass[i][j],
                        sat_mass[k],
                        z[i],
                        self.context.cosmology.as_ref(),
                    );

                    // Only satellites surviving to the reference epoch
                    // feed this output — mergers build a separate
                    // (not-yet-ported) accretion-history output.
                    if tdyf < ttz0 {
                        continue;
                    }
                    let z_bin = 0_usize;

                    // Number density of surviving subhalos of this mass
                    // at the reference epoch (`WeightList[0]` in the
                    // Python — the value at the *first* index of the
                    // z_bin..i survival window, which is the reference
                    // epoch since z_bin=0 for survivors).
                    let weight_at_reference = if i != 0 && z_bin != i {
                        self.hmf.dn_dlog10m(host_mass[z_bin][j], z[z_bin])
                            * shmf_entering[i][j][k]
                            * (host_mass_bins[z_bin][j] * config.log_m_bin)
                    } else {
                        self.hmf.dn_dlog10m(host_mass[i][j], z[i])
                            * shmf_entering[i][j][k]
                            * (host_mass_bins[i][j] * config.log_m_bin)
                    };

                    let sm_infall_dm = sat_mass[k] - log_h;
                    let final_sm: Vec<f64> = if z_bin < i && (config.star_formation || config.stellar_stripping) {
                        // Build the infall-to-reference-epoch timeline
                        // and evolve each realization.
                        let window_z: Vec<f64> = z[z_bin..=i].iter().rev().copied().collect();
                        let window_t: Vec<f64> = times[z_bin..=i].iter().rev().copied().collect();
                        let mut window_dt: Vec<f64> = window_t.windows(2).map(|w| w[1] - w[0]).collect();
                        window_dt.push(*window_dt.last().unwrap());
                        let timeline = Timeline {
                            z: window_z,
                            t: window_t,
                            dt: window_dt,
                            log_host_mass: vec![host_mass[i][j]; i - z_bin + 1],
                            t_dyn_friction: tdyf,
                        };

                        (0..config.n_realizations)
                            .map(|_| {
                                let log_sm_infall = self.smhm.stellar_mass(sm_infall_dm, z[i], Some(&mut rng));
                                let galaxy = SatelliteState {
                                    log_sm_infall,
                                    log_host_mass_infall: host_mass[i][j],
                                    log_sat_mass_infall: sat_mass[k],
                                    z_infall: z[i],
                                    pre_quenched: false,
                                };
                                let history = self.baryonic.evolve(
                                    &galaxy,
                                    &timeline,
                                    config.stellar_stripping,
                                    true,
                                    &mut rng,
                                );
                                *history.log_sm.last().unwrap()
                            })
                            .collect()
                    } else {
                        (0..config.n_realizations)
                            .map(|_| self.smhm.stellar_mass(sm_infall_dm, z[i], Some(&mut rng)))
                            .collect()
                    };

                    for &sm in &final_sm {
                        if let Some(bin) = histogram_bin_index(sm, config.sat_sm_min, config.sat_sm_bin, n_sm) {
                            surviving_sat_smf[bin] +=
                                weight_at_reference * h3 / (config.n_realizations as f64) / config.sat_sm_bin;
                        }
                    }
                }
            }
        }

        RunOutput { z, host_halo_mass: host_mass, sat_sm_range, surviving_sat_smf }
    }
}
