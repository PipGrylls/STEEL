//! Shared run context and the top-level orchestrator: the Rust
//! equivalent of `STEEL.py`'s module-level grid setup plus
//! `OneRealization`'s triple loop over redshift step / host-halo bin /
//! subhalo bin.
//!
//! # Correspondence with `STEEL.py`
//!
//! The loop reproduces `OneRealization` accumulator-for-accumulator.
//! Names in [`RunOutput`] deliberately keep the Python spellings
//! (including `Satilite_sSFR`'s typo, rendered here as
//! `satellite_ssfr`) so each field is a grep-able cross-reference.
//! Python array-shape symbols used throughout: `a` = redshift steps,
//! `b` = host-halo bins, `c` = subhalo bins.
//!
//! # Deliberate deviations from the Python
//!
//! Five off-by-one/dead-path defects were found in `OneRealization`
//! while porting it. Per the port's "clean reimplementation" mandate
//! they are fixed here rather than reproduced; each is marked
//! `PORT-FIX N` at its site and listed here so the set is auditable:
//!
//! 1. **Evolution window one step short.** `Functions.py::StarFormation`
//!    slices `z_all[z_bin_i:z_bin_r]`, giving `i - z_bin` grid points
//!    running `z[i] … z[z_bin+1]`, so `Starformation_c` applies
//!    `i - z_bin - 1` steps and never reaches the merge/reference epoch
//!    `z[z_bin]`. The accumulators then re-label those columns (via
//!    `np.flipud`) as if they spanned `z[i-1] … z[z_bin]`, a second,
//!    compensating shift. Here the timeline covers `z_bin..=i` —
//!    `i - z_bin + 1` points, `i - z_bin` steps — and ends exactly at
//!    `z[z_bin]`.
//! 2. **`Pair_Frac`/`Pair_Frac_Halo` silently dead whenever star
//!    formation or stripping is on.** `STEEL.py:436` guards the whole
//!    pair-fraction block with `if len(np.shape(SM_Sat)) == 1:` and has
//!    no `else`, so exactly the runs Papers 2 and 3 rely on write zeros.
//!    Here the block runs for both cases.
//! 3. **Stale `WeightList_SubOnly`.** The Python only assigns it in the
//!    `i != 0 and z_bin != i` branch, so the `z_bin == i` path reads
//!    whatever the previous `k` iteration left behind. Here it is always
//!    defined from the current bin.
//! 4. **`np.digitize` used as a histogram bin index** for
//!    `Total_StarFormation`'s `bin_` (`STEEL.py:333`) and for the
//!    `AnalyticalModel_Cuts_*` integration limit (`STEEL.py:450`,
//!    `:463`), while every other binning in the same function uses
//!    `fast_histogram`. The two conventions differ by one, so e.g. the
//!    `SM_Cuts = 9.0` integral actually excluded the `9.0`–`9.1` bin.
//!    Both use the histogram convention here.
//! 5. **sSFR histogram bins inconsistent with the axis saved beside
//!    them.** `sSFR_Range = np.arange(-14, -8, 0.1)` has 60 entries;
//!    `sSFR_len = size - 1 = 59` bins are then spread over the full
//!    `(-14, -8)` range, giving 0.1017-wide bins labelled with a
//!    0.1-spaced axis. Here the grid is 60 bins of exactly 0.1.
//!
//! Nothing else departs from the Python's numerics.

use std::sync::Arc;

use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::baryonic::{BaryonicPipeline, SatelliteState, Timeline};
use crate::cosmology::{Cosmology, MassDefinition};
use crate::halo_growth::HaloGrowthModel;
use crate::hmf::HaloMassFunctionModel;
use crate::merger_time::MergerTimescaleModel;
use crate::numerics::{arange_len, digitize};
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

/// Which output families to accumulate.
///
/// The full set costs both memory (`surviving_subhalos_z_z` alone is
/// `a * a * c` doubles — ~19 MB at the default grid) and time, and the
/// SMHM-fitting path in `steel-fit` needs only the high-z satellite SMF.
/// This is the Rust equivalent of `OneRealization`'s `ParamOverRide`
/// early-`continue` (`STEEL.py:383`), generalized from one hardcoded cut
/// point into named flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSelection {
    /// `SurvivingSubhalos`, `..._ByParent`, `..._z_z`.
    pub subhalo_mass_functions: bool,
    /// `Surviving_Sat_SMF_Weighting_Totals_highz`,
    /// `Surviving_Sat_SMF_Weighting_highz`, and the `Sat_Env_Highz` /
    /// `Raw_Richness` integrals derived from them.
    pub high_z_smf: bool,
    /// `Sat_SMHM`, `Sat_SMHM_Host`.
    pub satellite_smhm: bool,
    /// `Accretion_History{,_Halo}`, `Pair_Frac{,_Halo}`.
    pub mergers: bool,
    /// `Satilite_sSFR`.
    pub ssfr: bool,
    /// `Total_StarFormation` means and standard deviations.
    pub total_star_formation: bool,
}

impl OutputSelection {
    /// Everything — what a normal `steel` run produces.
    pub fn all() -> Self {
        Self {
            subhalo_mass_functions: true,
            high_z_smf: true,
            satellite_smhm: true,
            mergers: true,
            ssfr: true,
            total_star_formation: true,
        }
    }

    /// Only the satellite stellar mass functions (z=0 totals plus
    /// high-z), which is all the SMHM grid search consumes.
    pub fn smf_only() -> Self {
        Self {
            subhalo_mass_functions: false,
            high_z_smf: true,
            satellite_smhm: false,
            mergers: false,
            ssfr: false,
            total_star_formation: false,
        }
    }
}

impl Default for OutputSelection {
    fn default() -> Self {
        Self::all()
    }
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
    /// `Paramaters['PreProcessing']` — carried in `STEEL.py`'s run
    /// tuple as a `_PP` suffix on the `SFR_Model` field. Pre-quenches a
    /// mass-dependent fraction of each satellite's realization ensemble
    /// at infall, standing in for environmental processing the
    /// satellite underwent before entering this host.
    pub pre_processing: bool,
    /// `Stripping` in `Factor_Stripping_SF`.
    pub stellar_stripping: bool,
    /// `N` — abundance-matching scatter realizations per subhalo bin.
    pub n_realizations: usize,
    /// `SatM_min`/`SatM_max`/`SatBin` for the output stellar-mass grid.
    pub sat_sm_min: f64,
    pub sat_sm_max: f64,
    pub sat_sm_bin: f64,
    /// `sSFR_Range` grid \[log10 yr^-1\].
    pub ssfr_min: f64,
    pub ssfr_max: f64,
    pub ssfr_bin: f64,
    /// `SM_Cuts` — stellar-mass thresholds the richness integrals use.
    pub sm_cuts: Vec<f64>,
    /// Which output families to build.
    pub outputs: OutputSelection,
    /// Pair-fraction separation limits \[physical kpc\]
    /// (`STEEL.py:434-435`'s hardcoded 30 and 5).
    pub pair_radius_outer_kpc: f64,
    pub pair_radius_inner_kpc: f64,
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
            pre_processing: false,
            stellar_stripping: false,
            n_realizations: 5,
            sat_sm_min: 9.0,
            sat_sm_max: 13.0,
            sat_sm_bin: 0.1,
            ssfr_min: -14.0,
            ssfr_max: -8.0,
            ssfr_bin: 0.1,
            sm_cuts: vec![9.0, 9.5, 10.0, 10.5, 11.0, 11.45],
            outputs: OutputSelection::all(),
            pair_radius_outer_kpc: 30.0,
            pair_radius_inner_kpc: 5.0,
        }
    }
}

/// Everything `OneRealization` accumulates, in the shapes
/// `Functions.py`'s `SaveData_*` family writes.
///
/// Number densities carry the Python's units: `N Mpc^-3 h^3 dex^-1` for
/// the halo-mass-function-weighted arrays, `N dex^-1 per central halo`
/// for the `_SubOnly`-weighted merger and pair-fraction arrays.
pub struct RunOutput {
    // ---- grids ----
    /// Redshift steps, `z[0] ~= z_reference_min` increasing to `z_max`. `(a,)`
    pub z: Vec<f64>,
    /// `AvaHaloMass`: host halo mass \[log10 Msun/h\]. `(a, b)`
    pub host_halo_mass: Array2<f64>,
    /// `SatHaloMass`: subhalo mass grid \[log10 Msun/h\]. `(c,)`
    pub sat_halo_mass: Vec<f64>,
    /// `Surviving_Sat_SMF_MassRange[:-1]`: left edges of the satellite
    /// stellar-mass bins \[log10 Msun\]. `(n_sm,)`
    pub sat_sm_range: Vec<f64>,
    /// `sSFR_Range`: left edges of the sSFR bins \[log10 yr^-1\]. `(n_ssfr,)`
    pub ssfr_range: Vec<f64>,
    /// `SM_Cuts`. `(n_cuts,)`
    pub sm_cuts: Vec<f64>,

    // ---- unevolved surviving subhalo mass functions ----
    /// `SurvivingSubhalos`. `(a, c)`
    pub surviving_subhalos: Array2<f64>,
    /// `SurvivingSubhalos_ByParent`. `(a, b, c)`
    pub surviving_subhalos_by_parent: Array3<f64>,
    /// `SurvivingSubhalos_z_z`, indexed `[z_observed, z_infall, subhalo]`. `(a, a, c)`
    pub surviving_subhalos_z_z: Array3<f64>,

    // ---- satellite stellar mass functions ----
    /// `Surviving_Sat_SMF_Weighting_Totals` — the headline z=0 satellite
    /// SMF (`Figure3`). `(n_sm,)`
    pub surviving_sat_smf: Vec<f64>,
    /// `Surviving_Sat_SMF_Weighting` — the same, split by host bin
    /// (`Figure10`). `(b, n_sm)`
    pub surviving_sat_smf_by_host: Array2<f64>,
    /// `Surviving_Sat_SMF_Weighting_Totals_highz`. `(a, n_sm)`
    pub surviving_sat_smf_highz: Array2<f64>,
    /// `Surviving_Sat_SMF_Weighting_highz` (`Raw_Richness`). `(a, b, n_sm)`
    pub surviving_sat_smf_by_host_highz: Array3<f64>,

    // ---- satellite stellar-mass–halo-mass relation ----
    /// `Sat_SMHM`, by subhalo mass. `(a, c, n_sm)`
    pub sat_smhm: Array3<f64>,
    /// `Sat_SMHM_Host`, by host halo mass. `(a, b, n_sm)`
    pub sat_smhm_host: Array3<f64>,

    // ---- specific star formation rate ----
    /// `Satilite_sSFR` at the reference epoch. `(n_sm, n_ssfr)`
    pub satellite_ssfr: Array2<f64>,

    // ---- infall redshifts ----
    /// `z_infall`: the z=0 satellite SMF attributed to the redshift the
    /// satellite fell in at. `(a, n_sm)`
    pub z_infall: Array2<f64>,

    // ---- mergers and pairs ----
    /// `Accretion_History`, by merger redshift and host bin. `(a, b, n_sm)`
    pub accretion_history: Array3<f64>,
    /// `Accretion_History_Halo`, by subhalo mass instead of stellar mass. `(a, b, c)`
    pub accretion_history_halo: Array3<f64>,
    /// `Pair_Frac`. `(a, b, n_sm)`
    pub pair_frac: Array3<f64>,
    /// `Pair_Frac_Halo`. `(a, b, c)`
    pub pair_frac_halo: Array3<f64>,

    // ---- richness integrals above each SM cut ----
    /// `AnalyticalModel_Cuts_Frac`. `(n_cuts, b)`
    pub cuts_frac: Array2<f64>,
    /// `AnalyticalModel_Cuts_NoFrac`. `(n_cuts, b)`
    pub cuts_nofrac: Array2<f64>,
    /// `AnalyticalModel_Cuts_Frac_highz`. `(n_cuts, a, b)`
    pub cuts_frac_highz: Array3<f64>,
    /// `AnalyticalModel_Cuts_NoFrac_highz`. `(n_cuts, a, b)`
    pub cuts_nofrac_highz: Array3<f64>,

    // ---- bulk star formation in merged satellites ----
    /// `Total_StarFormation_Means` \[Msun\]. `NaN` where no satellite
    /// contributed, matching `np.mean([])`. `(a, b, n_sm)`
    pub total_star_formation_mean: Array3<f64>,
    /// `Total_StarFormation_Std` \[Msun\]. `(a, b, n_sm)`
    pub total_star_formation_std: Array3<f64>,
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

/// Index of the first histogram bin whose *left edge* is at or above
/// `cut` — the integration lower limit for the richness cuts.
///
/// PORT-FIX 4: `STEEL.py` uses `np.digitize(Cut, Surviving_Sat_SMF_MassRange)`
/// here, which for a `Cut` sitting exactly on a bin edge (all six of
/// STEEL's `SM_Cuts` except 11.45 do) returns one *past* that bin,
/// dropping it from the integral. The `1e-9` tolerance below absorbs the
/// floating-point noise in `min + n*bin` so an exact edge lands on its
/// own bin rather than the next one.
fn cut_bin_index(cut: f64, min: f64, bin_width: f64, n_bins: usize) -> usize {
    if cut <= min {
        return 0;
    }
    let idx = ((cut - min) / bin_width - 1e-9).ceil();
    (idx.max(0.0) as usize).min(n_bins)
}

/// Running mean and (population) standard deviation, accumulated with
/// Welford's algorithm.
///
/// `STEEL.py` grows a `list` per `(a, b, n_sm)` cell and calls
/// `np.mean`/`np.std` at the end — an `a*b*n_sm` grid of Python lists
/// (433 200 of them at the default resolution) holding every sample.
/// This keeps three doubles per cell instead, and reports `NaN` for
/// empty cells to match `np.mean([])`.
#[derive(Clone, Copy, Default)]
struct Welford {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    /// Population standard deviation (`np.std`'s default `ddof=0`).
    fn std(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            (self.m2 / self.count as f64).sqrt()
        }
    }
}

impl Simulation {
    /// Run the statistical dark-matter-accretion-history pipeline,
    /// producing every accumulator `STEEL.py::OneRealization` builds
    /// (subject to `config.outputs`).
    pub fn run(&self, config: &RunConfig) -> RunOutput {
        let h = self.context.cosmology.h();
        let log_h = h.log10();
        let sel = config.outputs;

        // Host halo mass grid at z=0 [log10 Msun/h] (`AnalyticHaloMass`).
        // Sized with `arange_len` (numpy `ceil` semantics) against the
        // same h-offset start/stop numpy is given, so the grid matches
        // `STEEL.py`'s `np.arange` element-for-element.
        let host_min = config.log_m_min + log_h;
        let host_max = config.log_m_max + log_h;
        let n_host = arange_len(host_min, host_max, config.log_m_bin);
        let host_mass_z0: Vec<f64> =
            (0..n_host).map(|j| host_min + j as f64 * config.log_m_bin).collect();

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
        let n_sat = arange_len(sat_min, sat_max, config.log_m_bin);
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

        // Output stellar-mass and sSFR grids.
        let n_sm = arange_len(config.sat_sm_min, config.sat_sm_max, config.sat_sm_bin);
        let sat_sm_range: Vec<f64> =
            (0..n_sm).map(|b| config.sat_sm_min + b as f64 * config.sat_sm_bin).collect();
        let n_ssfr = arange_len(config.ssfr_min, config.ssfr_max, config.ssfr_bin);
        let ssfr_range: Vec<f64> =
            (0..n_ssfr).map(|b| config.ssfr_min + b as f64 * config.ssfr_bin).collect();

        // Cosmic time bookkeeping (`Times`, `Time_To_0`).
        let times: Vec<f64> = z.iter().map(|&zi| self.context.cosmology.age(zi)).collect();
        let time_to_0: Vec<f64> = times.iter().map(|&t| times[0] - t).collect();

        // Halo mass function and host virial radius on the (redshift,
        // host bin) grid. Both depend only on `(i, j)`, but the loops
        // below need them once per `(i, j, k)` *window step* — up to
        // `n_z` times per subhalo bin — so evaluating them in place
        // would repeat the same `sigma(M)` quadrature ~1e8 times and
        // dominate the entire run. This is the same move
        // `Functions.py::Make_HMF_Interp` makes by building a 2-D
        // interpolation table, except tabulated exactly on the grid it
        // is queried at, so there is no interpolation error to carry.
        let mut hmf_grid = vec![vec![0.0_f64; n_host]; n_z];
        let mut virial_radius_kpc = vec![vec![0.0_f64; n_host]; n_z];
        for i in 0..n_z {
            for j in 0..n_host {
                hmf_grid[i][j] = self.hmf.dn_dlog10m(host_mass[i][j], z[i]);
                virial_radius_kpc[i][j] = self.context.cosmology.m_to_r(
                    10f64.powf(host_mass[i][j]),
                    z[i],
                    MassDefinition::Vir,
                ) / h;
            }
        }

        // ---- accumulators ----
        // The unevolved surviving subhalo mass function is only defined
        // when nothing evolves the subhalo, matching `STEEL.py:298`'s
        // `(Stripping_DM == False) and (Stripping or SF) == False`
        // guard. The Python still allocates and saves the zero-filled
        // arrays in that case; here they are left empty so the writer
        // omits the files rather than emitting ~19 MB of zeros that a
        // plotting script would happily draw as a flat line.
        let want_unevolved_shmf = sel.subhalo_mass_functions
            && self.halo_stripping.is_none()
            && !(config.stellar_stripping || config.star_formation);
        let shmf_2d = if want_unevolved_shmf { (n_z, n_sat) } else { (0, 0) };
        let mut surviving_subhalos = Array2::<f64>::zeros(shmf_2d);
        let mut surviving_subhalos_by_parent =
            Array3::<f64>::zeros(if want_unevolved_shmf { (n_z, n_host, n_sat) } else { (0, 0, 0) });
        let mut surviving_subhalos_z_z =
            Array3::<f64>::zeros(if want_unevolved_shmf { (n_z, n_z, n_sat) } else { (0, 0, 0) });
        let mut surviving_sat_smf = vec![0.0_f64; n_sm];
        let mut surviving_sat_smf_by_host = Array2::<f64>::zeros((n_host, n_sm));
        let mut surviving_sat_smf_highz =
            Array2::<f64>::zeros(if sel.high_z_smf { (n_z, n_sm) } else { (0, 0) });
        let mut surviving_sat_smf_by_host_highz = Array3::<f64>::zeros(if sel.high_z_smf {
            (n_z, n_host, n_sm)
        } else {
            (0, 0, 0)
        });
        let mut sat_smhm =
            Array3::<f64>::zeros(if sel.satellite_smhm { (n_z, n_sat, n_sm) } else { (0, 0, 0) });
        let mut sat_smhm_host =
            Array3::<f64>::zeros(if sel.satellite_smhm { (n_z, n_host, n_sm) } else { (0, 0, 0) });
        let mut satellite_ssfr =
            Array2::<f64>::zeros(if sel.ssfr { (n_sm, n_ssfr) } else { (0, 0) });
        let mut z_infall = Array2::<f64>::zeros((n_z, n_sm));
        let merger_shape = if sel.mergers { (n_z, n_host, n_sm) } else { (0, 0, 0) };
        let merger_halo_shape = if sel.mergers { (n_z, n_host, n_sat) } else { (0, 0, 0) };
        let mut accretion_history = Array3::<f64>::zeros(merger_shape);
        let mut accretion_history_halo = Array3::<f64>::zeros(merger_halo_shape);
        let mut pair_frac = Array3::<f64>::zeros(merger_shape);
        let mut pair_frac_halo = Array3::<f64>::zeros(merger_halo_shape);
        let mut total_sf = if sel.total_star_formation && (config.star_formation || config.stellar_stripping)
        {
            Array3::<Welford>::default((n_z, n_host, n_sm))
        } else {
            Array3::<Welford>::default((0, 0, 0))
        };

        let mut rng = StdRng::seed_from_u64(self.context.rng_seed);
        let h3 = h * h * h;
        let n_real = config.n_realizations;
        let per_realization = 1.0 / n_real as f64;

        // Scratch buffers reused across (i, j, k) so the hot loop does no
        // per-bin allocation.
        let mut weight_list: Vec<f64> = Vec::with_capacity(n_z);
        let mut sm_infall: Vec<f64> = vec![0.0; n_real];
        // `trajectory[r]` is realization `r`'s stellar-mass track,
        // indexed by *history* step (0 = infall at z[i], last = the
        // merge/reference epoch at z[z_bin]).
        let mut trajectory: Vec<Vec<f64>> = vec![Vec::new(); n_real];
        let mut ssfr_final: Vec<f64> = vec![0.0; n_real];

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

                    // `z_bin` indexes the redshift step the satellite
                    // returns at: its merger epoch, or the reference
                    // epoch (0) if it survives.
                    let merges = tdyf < ttz0;
                    let z_bin = if merges { digitize(tdyf + times[i], &times) } else { 0 };
                    // A merging satellite always lands at or before `i`;
                    // `digitize` on the decreasing `times` array cannot
                    // exceed it, but clamp rather than trust that.
                    let z_bin = z_bin.min(i);
                    let n_w = i - z_bin; // number of evolution steps

                    // ---- weight lists ----
                    // `WeightList[m]` is the comoving number density of
                    // subhalos of mass `sat_mass[k]` accreted at step `i`
                    // into hosts of bin `j`, evaluated at step `z_bin+m`.
                    //
                    // The Python builds this with `interp2d` over the
                    // whole window and then extracts the anti-diagonal
                    // (`np.diag(np.fliplr(Arr2D))`) to recover the
                    // element-wise pairing; evaluating the HMF pointwise
                    // gives the same numbers without the detour.
                    weight_list.clear();
                    let shmf_e = shmf_entering[i][j][k];
                    if i != 0 && z_bin != i {
                        for m in 0..n_w {
                            let idx = z_bin + m;
                            weight_list.push(
                                hmf_grid[idx][j] * shmf_e * (host_mass_bins[idx][j] * config.log_m_bin),
                            );
                        }
                    } else {
                        // Makes sure accretion in the final redshift step
                        // is still counted.
                        weight_list.push(
                            hmf_grid[i][j] * shmf_e * (host_mass_bins[i][j] * config.log_m_bin),
                        );
                    }
                    // PORT-FIX 3: always defined, never inherited from
                    // the previous `k`. Constant over the window, so one
                    // scalar suffices where the Python carries an array.
                    let weight_sub_only = shmf_e * config.log_m_bin; // N per central

                    // ---- unevolved surviving subhalo mass function ----
                    if want_unevolved_shmf {
                        for (m, &weight) in weight_list.iter().enumerate().take(n_w) {
                            let idx = z_bin + m;
                            let w = weight / config.log_m_bin;
                            surviving_subhalos[[idx, k]] += w;
                            surviving_subhalos_by_parent[[idx, j, k]] += w;
                            surviving_subhalos_z_z[[idx, i, k]] += w;
                        }
                    }

                    // ---- abundance matching at infall ----
                    let sm_infall_dm = sat_mass[k] - log_h;
                    for slot in sm_infall.iter_mut() {
                        *slot = self.smhm.stellar_mass(sm_infall_dm, z[i], Some(&mut rng));
                    }

                    // ---- baryonic evolution over the infall window ----
                    let evolved = n_w > 0 && (config.star_formation || config.stellar_stripping);

                    // `Paramaters['PreProcessing']`: pre-quench part of
                    // the ensemble at infall.
                    // `Functions.py::StarFormation` derives `PP_Frac`
                    // from the ensemble-mean infall stellar mass and
                    // then does `T_quench[:int(PP_Frac*len)] = t[0]` —
                    // a *prefix* of the realization axis, not a random
                    // subset. Since the realizations are i.i.d. draws
                    // that is equivalent to choosing at random, and
                    // taking the prefix reproduces the Python exactly.
                    let n_pre_quenched = if config.pre_processing {
                        let mean_sm = ((0..n_real).map(|r| 10f64.powf(sm_infall[r])).sum::<f64>()
                            * per_realization)
                            .log10();
                        let pp_frac = if mean_sm < 6.0 {
                            0.6
                        } else if mean_sm > 8.0 {
                            0.3
                        } else {
                            0.6 - 0.3 * ((mean_sm - 6.0) / 2.0)
                        };
                        (pp_frac * n_real as f64) as usize
                    } else {
                        0
                    };

                    if evolved {
                        // PORT-FIX 1: the window spans `z_bin..=i`
                        // inclusive, so the track ends exactly at the
                        // merge/reference epoch.
                        let window_z: Vec<f64> = z[z_bin..=i].iter().rev().copied().collect();
                        let window_t: Vec<f64> = times[z_bin..=i].iter().rev().copied().collect();
                        let mut window_dt: Vec<f64> =
                            window_t.windows(2).map(|w| w[1] - w[0]).collect();
                        window_dt.push(*window_dt.last().unwrap());
                        let timeline = Timeline {
                            z: window_z,
                            t: window_t,
                            dt: window_dt,
                            log_host_mass: vec![host_mass[i][j]; n_w + 1],
                            t_dyn_friction: tdyf,
                        };

                        for r in 0..n_real {
                            let galaxy = SatelliteState {
                                log_sm_infall: sm_infall[r],
                                log_host_mass_infall: host_mass[i][j],
                                log_sat_mass_infall: sat_mass[k],
                                z_infall: z[i],
                                pre_quenched: r < n_pre_quenched,
                            };
                            let history = self.baryonic.evolve(
                                &galaxy,
                                &timeline,
                                config.stellar_stripping,
                                true,
                                &mut rng,
                            );
                            ssfr_final[r] = *history.log_ssfr.last().unwrap();
                            trajectory[r] = history.log_sm;
                        }
                    } else {
                        // Unevolved: the satellite keeps its infall mass
                        // at every step, which is exactly what the
                        // Python's tiled `np.full((n_w, ...), Wt_Corr)`
                        // does with a 1-D `SM_Sat`.
                        for r in 0..n_real {
                            trajectory[r].clear();
                            trajectory[r].push(sm_infall[r]);
                            ssfr_final[r] = f64::NEG_INFINITY;
                        }
                    }

                    // Stellar mass at the return epoch (`SM_Sat[:,-1]`).
                    // Reading through `trajectory` keeps the evolved and
                    // unevolved cases on one code path.
                    let final_sm = |r: usize| -> f64 { *trajectory[r].last().unwrap() };
                    // Stellar mass at redshift step `z_bin + m`. History
                    // step `p` sits at redshift index `i - p`, so step
                    // `z_bin + m` is history step `n_w - m`. When the
                    // satellite was not evolved the track is a single
                    // entry, held constant across the window.
                    let sm_at = |r: usize, m: usize| -> f64 {
                        if evolved {
                            trajectory[r][n_w - m]
                        } else {
                            trajectory[r][0]
                        }
                    };

                    // ---- bulk star formation in merged satellites ----
                    if sel.total_star_formation && evolved && merges && !total_sf.is_empty() {
                        let mass_before: f64 = (0..n_real)
                            .map(|r| 10f64.powf(sm_infall[r]))
                            .sum::<f64>()
                            * per_realization;
                        let mass_after: f64 =
                            (0..n_real).map(|r| 10f64.powf(final_sm(r))).sum::<f64>() * per_realization;
                        // PORT-FIX 4: histogram convention, not `np.digitize`.
                        if let Some(bin) = histogram_bin_index(
                            mass_before.log10(),
                            config.sat_sm_min,
                            config.sat_sm_bin,
                            n_sm,
                        ) {
                            total_sf[[z_bin, j, bin]].push(mass_after - mass_before);
                        }
                    }

                    // ---- sSFR distribution at the reference epoch ----
                    if sel.ssfr && z_bin == 0 && config.star_formation {
                        let scale = weight_list[0] * h3 * per_realization;
                        for (r, &ssfr) in ssfr_final.iter().enumerate().take(n_real) {
                            let sm_bin = histogram_bin_index(
                                final_sm(r),
                                config.sat_sm_min,
                                config.sat_sm_bin,
                                n_sm,
                            );
                            let s_bin = histogram_bin_index(
                                ssfr,
                                config.ssfr_min,
                                config.ssfr_bin,
                                n_ssfr,
                            );
                            if let (Some(sb), Some(fb)) = (sm_bin, s_bin) {
                                satellite_ssfr[[sb, fb]] += scale;
                            }
                        }
                    }

                    // ---- z=0 satellite SMF (survivors only) ----
                    if !merges {
                        let scale = weight_list[0] * h3 * per_realization / config.sat_sm_bin;
                        for r in 0..n_real {
                            if let Some(bin) = histogram_bin_index(
                                final_sm(r),
                                config.sat_sm_min,
                                config.sat_sm_bin,
                                n_sm,
                            ) {
                                surviving_sat_smf[bin] += scale;
                                surviving_sat_smf_by_host[[j, bin]] += scale;
                                z_infall[[i, bin]] += scale;
                            }
                        }
                    }

                    // ---- high-z satellite SMF and satellite SMHM ----
                    let want_window = sel.high_z_smf || sel.satellite_smhm;
                    if want_window && z_bin != i && i != 0 {
                        for (m, &weight) in weight_list.iter().enumerate().take(n_w) {
                            let idx = z_bin + m;
                            let scale = weight * h3 * per_realization / config.sat_sm_bin;
                            for r in 0..n_real {
                                let Some(bin) = histogram_bin_index(
                                    sm_at(r, m),
                                    config.sat_sm_min,
                                    config.sat_sm_bin,
                                    n_sm,
                                ) else {
                                    continue;
                                };
                                if sel.high_z_smf {
                                    surviving_sat_smf_highz[[idx, bin]] += scale;
                                    surviving_sat_smf_by_host_highz[[idx, j, bin]] += scale;
                                }
                                if sel.satellite_smhm {
                                    sat_smhm[[idx, k, bin]] += scale;
                                    sat_smhm_host[[idx, j, bin]] += scale;
                                }
                            }
                        }
                    } else if sel.satellite_smhm {
                        let scale = weight_list[0] * h3 * per_realization / config.sat_sm_bin;
                        for r in 0..n_real {
                            if let Some(bin) = histogram_bin_index(
                                final_sm(r),
                                config.sat_sm_min,
                                config.sat_sm_bin,
                                n_sm,
                            ) {
                                sat_smhm[[i, k, bin]] += scale;
                                sat_smhm_host[[i, j, bin]] += scale;
                            }
                        }
                    }

                    if !sel.mergers {
                        continue;
                    }

                    // ---- merger rate per mass track ----
                    if merges {
                        let scale = weight_sub_only * per_realization / config.sat_sm_bin;
                        for r in 0..n_real {
                            if let Some(bin) = histogram_bin_index(
                                final_sm(r),
                                config.sat_sm_min,
                                config.sat_sm_bin,
                                n_sm,
                            ) {
                                accretion_history[[z_bin, j, bin]] += scale;
                            }
                        }
                        accretion_history_halo[[z_bin, j, k]] += weight_sub_only / config.log_m_bin;
                    }

                    // ---- pair fraction ----
                    // PORT-FIX 2: runs for evolved satellites too.
                    if z_bin != i {
                        // Host virial radius in physical kpc.
                        let vr_kpc = virial_radius_kpc[i][j];
                        // Guo+2011 linear infall: the separation shrinks
                        // linearly from the virial radius over one
                        // dynamical friction time. Monotonically
                        // increasing in `m` (elapsed time since infall
                        // decreases as `m` rises), so counting how many
                        // entries fall below a threshold gives the index
                        // at which it is crossed.
                        let radius = |m: usize| -> f64 {
                            let elapsed = (time_to_0[z_bin + m] - time_to_0[i]).abs();
                            vr_kpc * (1.0 - elapsed / tdyf)
                        };
                        let pf_upper =
                            (0..n_w).filter(|&m| radius(m) < config.pair_radius_outer_kpc).count();
                        let pf_lower =
                            (0..n_w).filter(|&m| radius(m) < config.pair_radius_inner_kpc).count();

                        for m in pf_lower..pf_upper {
                            let idx = z_bin + m;
                            let scale = weight_sub_only * per_realization / config.sat_sm_bin;
                            for r in 0..n_real {
                                if let Some(bin) = histogram_bin_index(
                                    sm_at(r, m),
                                    config.sat_sm_min,
                                    config.sat_sm_bin,
                                    n_sm,
                                ) {
                                    pair_frac[[idx, j, bin]] += scale;
                                }
                            }
                            pair_frac_halo[[idx, j, k]] += weight_sub_only / config.log_m_bin;
                        }
                    }
                }
            }
        }

        // ---- richness integrals above each stellar-mass cut ----
        let n_cuts = config.sm_cuts.len();
        let mut cuts_frac = Array2::<f64>::zeros((n_cuts, n_host));
        let mut cuts_nofrac = Array2::<f64>::zeros((n_cuts, n_host));
        for (c_idx, &cut) in config.sm_cuts.iter().enumerate() {
            let sm_bin = cut_bin_index(cut, config.sat_sm_min, config.sat_sm_bin, n_sm);
            let mut total = 0.0;
            for j in 0..n_host {
                let integral: f64 =
                    (sm_bin..n_sm).map(|b| surviving_sat_smf_by_host[[j, b]]).sum::<f64>()
                        * config.sat_sm_bin;
                cuts_nofrac[[c_idx, j]] = integral;
                total += integral;
            }
            for j in 0..n_host {
                cuts_frac[[c_idx, j]] = cuts_nofrac[[c_idx, j]] / total;
            }
        }

        let highz_cut_shape = if sel.high_z_smf { (n_cuts, n_z, n_host) } else { (0, 0, 0) };
        let mut cuts_frac_highz = Array3::<f64>::zeros(highz_cut_shape);
        let mut cuts_nofrac_highz = Array3::<f64>::zeros(highz_cut_shape);
        if sel.high_z_smf {
            for (c_idx, &cut) in config.sm_cuts.iter().enumerate() {
                let sm_bin = cut_bin_index(cut, config.sat_sm_min, config.sat_sm_bin, n_sm);
                for i in 0..n_z {
                    let mut total = 0.0;
                    for j in 0..n_host {
                        let integral: f64 = (sm_bin..n_sm)
                            .map(|b| surviving_sat_smf_by_host_highz[[i, j, b]])
                            .sum::<f64>()
                            * config.sat_sm_bin;
                        cuts_nofrac_highz[[c_idx, i, j]] = integral;
                        total += integral;
                    }
                    for j in 0..n_host {
                        cuts_frac_highz[[c_idx, i, j]] = cuts_nofrac_highz[[c_idx, i, j]] / total;
                    }
                }
            }
        }

        let total_star_formation_mean = total_sf.map(|w| w.mean());
        let total_star_formation_std = total_sf.map(|w| w.std());

        let mut host_halo_mass = Array2::<f64>::zeros((n_z, n_host));
        for (i, row) in host_mass.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                host_halo_mass[[i, j]] = v;
            }
        }

        RunOutput {
            z,
            host_halo_mass,
            sat_halo_mass: sat_mass,
            sat_sm_range,
            ssfr_range,
            sm_cuts: config.sm_cuts.clone(),
            surviving_subhalos,
            surviving_subhalos_by_parent,
            surviving_subhalos_z_z,
            surviving_sat_smf,
            surviving_sat_smf_by_host,
            surviving_sat_smf_highz,
            surviving_sat_smf_by_host_highz,
            sat_smhm,
            sat_smhm_host,
            satellite_ssfr,
            z_infall,
            accretion_history,
            accretion_history_halo,
            pair_frac,
            pair_frac_halo,
            cuts_frac,
            cuts_nofrac,
            cuts_frac_highz,
            cuts_nofrac_highz,
            total_star_formation_mean,
            total_star_formation_std,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_bin_index_drops_out_of_range_values() {
        assert_eq!(histogram_bin_index(8.9, 9.0, 0.1, 40), None);
        assert_eq!(histogram_bin_index(9.0, 9.0, 0.1, 40), Some(0));
        assert_eq!(histogram_bin_index(9.05, 9.0, 0.1, 40), Some(0));
        assert_eq!(histogram_bin_index(12.95, 9.0, 0.1, 40), Some(39));
        assert_eq!(histogram_bin_index(13.0, 9.0, 0.1, 40), None);
    }

    #[test]
    fn cut_bin_index_includes_the_bin_starting_at_the_cut() {
        // PORT-FIX 4. `np.digitize(9.0, np.arange(9, 13.1, 0.1))` is 1,
        // which drops the 9.0-9.1 bin from an integral that is supposed
        // to be "everything above 9.0"; the same happens at every cut
        // that lands on an edge.
        assert_eq!(cut_bin_index(9.0, 9.0, 0.1, 40), 0);
        assert_eq!(cut_bin_index(9.5, 9.0, 0.1, 40), 5);
        assert_eq!(cut_bin_index(10.0, 9.0, 0.1, 40), 10);
        assert_eq!(cut_bin_index(11.0, 9.0, 0.1, 40), 20);
        // 11.45 is mid-bin: the 11.4-11.5 bin is only partly above the
        // cut, so the integral starts at the next whole bin, as it did
        // in the Python.
        assert_eq!(cut_bin_index(11.45, 9.0, 0.1, 40), 25);
        // Out of range on either side.
        assert_eq!(cut_bin_index(8.0, 9.0, 0.1, 40), 0);
        assert_eq!(cut_bin_index(99.0, 9.0, 0.1, 40), 40);
    }

    #[test]
    fn welford_matches_numpy_mean_and_population_std() {
        let mut w = Welford::default();
        assert!(w.mean().is_nan(), "empty should be NaN, like np.mean([])");
        assert!(w.std().is_nan());
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.push(x);
        }
        // np.mean -> 5.0, np.std (ddof=0) -> 2.0
        assert!((w.mean() - 5.0).abs() < 1e-12, "mean = {}", w.mean());
        assert!((w.std() - 2.0).abs() < 1e-12, "std = {}", w.std());
    }

    #[test]
    fn welford_single_sample_has_zero_spread() {
        let mut w = Welford::default();
        w.push(3.5);
        assert_eq!(w.mean(), 3.5);
        assert_eq!(w.std(), 0.0);
    }

    #[test]
    fn output_selection_presets_differ_where_it_matters() {
        let all = OutputSelection::all();
        let smf = OutputSelection::smf_only();
        assert!(all.mergers && all.subhalo_mass_functions && all.ssfr);
        assert!(smf.high_z_smf, "the grid search consumes the high-z SMF");
        assert!(!smf.mergers && !smf.subhalo_mass_functions && !smf.ssfr);
    }
}
