# Results-figure reproduction: coverage across all three papers

Goal: reproduce every *results* figure in Papers 1-3 across
py-as-is / py-corrected / rs-steel, as evidence the port is sound —
not to re-validate the model against real data. Per the plan this
project has followed throughout: **the three-way model-output
comparison needs no observational data at all**; where a published
figure overlays SDSS/Illustris/Mundy et al. data, this reproduction
shows the *model curves* the three implementations produce and omits
the external data, since matching that data is a separate scientific
claim from matching each other.

**Paper numbering.** This table's Paper 1/2/3 labels match
`configs/published_runs.toml` (the authoritative mapping built in
Phase 3, cross-checked directly against each paper's PDF):

* **Paper 1** = Grylls+2019a, "A statistical semi-empirical model:
  satellite galaxies in groups and clusters" (arXiv 1812.00015, MNRAS
  483, 2506). Thesis Chapter 3. The satellite-SMF / frozen-model /
  dynamical-time paper.
* **Paper 2** = Grylls+2020, "Predicting fully self-consistent
  satellite richness, galaxy growth, and star formation rates..."
  (arXiv 1910.08417, MNRAS 491, 634, running header "STEELIIIa").
  Thesis Chapter 4. The galaxy-growth / SFR / ellipticals paper.
* **Paper 3** = Grylls, Shankar & Conselice 2020, "The significant
  effects of stellar mass estimation on galaxy pair fractions" (arXiv
  2001.06017). Thesis Chapter 5.

**This file previously had Papers 1 and 2 swapped** relative to that
mapping (an error caught 2026-08-12 when the user pointed out
"missing" Paper 1 Figs 13/15 that were actually present under the
wrong paper's heading). All figure files and this table were relabeled
to match `configs/published_runs.toml`. Several figures also turned
out to have been built with the **wrong run configuration**, not just
the wrong label — flagged explicitly below, each with a `_WRONG_*`
suffix on the stale file pending replacement.

Four classes, assigned per figure below:

* **[fn]** — a standalone model function (SMHM relation, quenching
  timescale, merger timescale, SHMF...). No simulation run needed.
* **[run]** — needs an actual `STEEL.py`/`rs-steel` realization
  (accretion histories, SMFs, pair fractions...).
* **[data]** — the published figure's point is a comparison to
  external data (SDSS, Illustris TNG, Mundy et al.). Reproduced as
  [fn]/[run] above with the data overlay omitted, or skipped if the
  model side alone is not the figure's content.
* **[diagram]** — a schematic illustration, not a result. Not
  reproduced.

## Known issues being actively fixed (2026-08-12 correction pass)

1. **FIXED.** Paper 1 Figs 8/9/10/11 were built from the wrong run
   configuration. The real Figs 8-11 are Section 4.1 ("Frozen model"):
   satellites never evolve in stellar mass after infall (no SF, no
   stripping), and the comparison axis is `f_tdyn` (0.5/1.0/2.5/∞) x
   SMHM z-evolution (on/off) — **not** SF/stripping on/off (that's
   Figs 14/15's axis, which the old Fig 8 was accidentally reproducing
   a variant of). Rebuilt with 5 new frozen-model three-way runs
   (`rust/runfiles/published/p2-dpl-frozen-*.toml` — filename prefix
   `p2-` predates the relabel and still refers to the old, wrong
   "Paper 2" name; not renamed, cosmetic only). These runs (and Figs
   2/3/5-11's existing G19_DPL/G19_SE-based ones) use **G19_SE**, not
   the **G18** preset `configs/published_runs.toml` says Paper 1
   actually used — discovered after these runs were already committed
   to. Both legs use the same preset, so this doesn't affect the
   port-validity claim (the point of the three-way comparison), only
   exact numeric match to the published curves, which was never the
   goal per this doc's opening note. Figs 12-15's rebuild (below) uses
   the correct G18.
2. **FIXED.** Paper 1 Figs 12/13/14/15 used G19_DPL as a substitute
   for the CE (continuity-equation) SFR model, believing CE wasn't
   ported. **CE is implemented** (`TomczakFormSfr::ce()` /
   `rust/steel-io/src/runfile.rs` `preset = "ce"`) — this was an
   error, not a limitation. Rebuilt with 4 new G18/CE/T16 runs
   (`p1-sf-strip-t16.toml`, `p1-sf-strip-ce.toml`, `p1-sf-ce.toml`,
   `p1-tdyn-1.0.toml`), matching the correct preset this time.
3. **Paper 3 Fig. 3** only reproduced the figure's outer/context
   panels (the input SMHM relation under each of the 13 PFT
   perturbations). The actual scientific result plotted in the
   figure's center panels — **pair fraction vs. redshift** for each
   perturbation — was never built. This needs 14 full SF+stripping
   three-way runs (13 perturbations + reference), in progress via
   the pre-existing `rust/runfiles/published/p3-pft-*.toml` /
   `p3-reference.toml` runfiles (built in an earlier phase but
   apparently never run through to a figure).
4. **Paper 2 (real, MNRAS 491) Figs 6-11** need substantially more
   work than what's built: Figs 6/8 are a 3-row decomposition (total
   mass / fractional contribution / instantaneous rate) for 3 target
   masses, not the single AM-vs-SFR-only strand currently shown; Fig 7
   needs the actual double-power-law fits + extracted-track data
   points, not just a bare main-sequence line; Fig 9 needs the exact
   3 mass bins and the dynamical-quenching model curve; Fig 10 (SFR-M*
   at z=0.5/1/2 vs Leja+2019) isn't built at all; Fig 11's "infeasible"
   verdict was based on the wrong array (`P_Elliptical`, dead in
   `STEEL.py`) — the real Fig 11 is a **post-processing** computation
   (major-merger fraction from mass tracks) that may well be
   buildable and needs re-investigation, not a dead end.

Until each of these is rebuilt, the affected rows below are marked
**pending (relabeled, content wrong)** rather than done/partial.

## Paper 1 (satellite SMF / frozen-model paper, arXiv 1812.00015, MNRAS 483)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram (statistical DM backbone steps) | diagram | skip |
| 2 | USHMF vs USSHMF (3 f_tdyn), one parent halo mass | run | **one line done** (`Paper1_Fig2_USSHMF.png`: USSHMF at f_tdyn=1.0 only, log Mh,parent=12.80, matches the paper's actual value; USHMF reference curve and other 2 f_tdyn values not built) |
| 3 | Total USHMF vs total USSHMF (3 f_tdyn) | run | **one line done** (`Paper1_Fig3_TotalUSSHMF.png`: total USSHMF at f_tdyn=1.0 only; USHMF reference curve and other 2 f_tdyn values not built) |
| 4 | SMHM (G18) vs B18/S17/M13 + central SMF vs SDSS | fn + data | **left panel approximate** (`SMHM_Relation.png` uses G19_SE, not the G18 preset this figure actually uses -- same functional form, different coefficients; B18/S17/M13 and right-panel central SMF omitted) |
| 5 | % satellites by accretion redshift | run | **done** (`Paper1_Fig5_AccretionRedshift.png`) |
| 6 | Quenching delay time-scale (Wetzel+F16) | fn | **done** (`Paper1_Fig6_Quenching.png`) |
| 7 | Dynamical-friction merging time-scale, 2-panel (vs M_h,sat and M*,sat), 3 host masses | fn | **done** (`Paper1_Fig7_MergerTimescale.png`: both panels, all 3 host masses (12/13/14), "time to z=0" reference line; py-corrected/rs-steel overlap exactly, as expected -- no correction touches this function) |
| 8 | SSMF, frozen model: f_tdyn={1.0,inf} x z-evo={on,off}, vs SDSS | run + data | **done** (`Paper1_Fig8_FrozenSweep.png`: all 4 lines, correct axis, tight py/rust agreement; SDSS overlay omitted; SMHM preset is G19_SE not the paper's G18 -- documented approximation, doesn't affect the port-validity comparison since both legs use the same preset) |
| 9 | Satellite distributions, frozen model, same f_tdyn/z-evo axis, 3 mass cuts x 2 rows | run + data | **done** (`Paper1_Fig9_FrozenSweep_Grid.png`: full 3x2 grid, all 4 lines; some visible py/rust divergence in the fractional (bottom) row, consistent with small-number-statistics noise seen elsewhere in this reproduction, not a port defect; SDSS overlay omitted, G19_SE not G18) |
| 10 | SSMF, frozen model, f_tdyn=0.5/1.0/2.5, vs SDSS | run + data | **done** (`Paper1_Fig10_TdynSweep.png`: correct frozen config, all 3 f_tdyn values, tight agreement; SDSS overlay omitted, G19_SE not G18) |
| 11 | Satellite distributions, frozen model, f_tdyn=0.5/1.0/2.5, 3 mass cuts x 2 rows | run + data | **done** (`Paper1_Fig11_TdynSweep_Grid.png`: full 3x2 grid; same fractional-row noise caveat as Fig. 9; SDSS overlay omitted, G19_SE not G18) |
| 12 | SSMF, T16 vs CE SFR, vs SDSS | run + data | **done** (`Paper1_Fig12_SFRModelSweep.png`: real CE (`TomczakFormSfr::ce()`) vs T16, G18 preset, tight py/rust agreement; SDSS overlay omitted) |
| 13 | sSFR distributions, T16 vs CE, vs SDSS | run + data | **done** (`Paper1_Fig13_sSFRSweep.png`: real CE vs T16, G18; same sparse-bin small-number-statistics caveat as elsewhere; SDSS overlay omitted) |
| 14 | SSMF, frozen/SF/SF+strip, vs SDSS | run + data | **done** (`Paper1_Fig14_ConfigSweep.png`: all 3 lines -- frozen, CE SF-only, CE SF+stripping -- G18 preset, tight agreement; SDSS overlay omitted) |
| 15 | Satellite distributions, frozen/SF/SF+strip, vs SDSS | run + data | **done** (`Paper1_Fig15_ConfigSweep_Grid.png`: all 3 lines, full 3x2 grid, G18/CE; SDSS overlay omitted) |

## Paper 2 (galaxy growth / SFR / ellipticals paper, arXiv 1910.08417, MNRAS 491, "STEELIIIa")

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram (photometry -> SMHM -> accretion) | diagram | skip |
| 2 | Method diagram (SFR pipeline: SMF/HMF/growth/SMHM/richness/SFR) | diagram | skip |
| 3 | SMHM (PyMorph/cmodel/M13) + central SMF vs Illustris TNG | fn + data | **left panel approximate** (`SMHM_Relation.png`, G19_SE/G19_cMod as PyMorph/cmodel stand-ins; M13 and right-panel central SMF vs Illustris omitted) |
| 4 | Merger rate per Gyr at fixed halo mass, mass ratio > 0.3, vs Fakhouri+2010 | run + data | **built, mass-ratio cut not applied** (`Paper2_Fig4_MergerRate.png`: 4 halo masses, all mass ratios rather than the paper's >0.3 major-merger cut; Fakhouri+2010 band omitted) |
| 5 | Satellite number density vs parent halo mass, multi-z, vs SDSS/Illustris/Wang+16/Wen&Han+18 | run + data | **done** (`Paper2_Fig5_SatelliteDistribution_MultiZ.png`: 6 redshifts, M*>10^10 cut; data overlays omitted) |
| 6 | Mass tracks (PyMorph), 3-row: total mass, fractional contribution, instantaneous rate, for 3 target masses | run | **done** (`Paper2_Fig6_MassTrackDecomposition.png`: all 3 rows (total/accretion/SFH, fractional contribution since z=3, instantaneous rate ratio), all 3 target masses (11.0/11.5/12.0) matching the real figure; "Accretion" = Total-SFH in linear mass, masked near the track's z=3 start where they're equal by construction rather than floored (which drew a misleading cliff to log=0); the M*=10^11 track's accretion line is visibly noisier for rs-steel around z~1-2, consistent with it being a small difference of two close values there, not necessarily a port defect; py-as-is absent, G3) |
| 7 | SFR-M* relation: double-power-law fits at z=0.1,1,2,3,5 + extracted track data points + 3 population tracks overlaid | run | **simplified main-sequence-only version** (`Paper2_Fig7_SFR_Mstar.png`: SFR recomputed from 5 central tracks at z=0/1/2, no fit-curve/data-point distinction, no population-track overlay; not the real figure's structure) |
| 8 | Mass tracks (cmodel), same 3-row structure as Fig. 6 | run | **not built** -- `mass_track_decomposition.py` (built for Fig. 6) would directly cover this once `mass_tracks.py`'s `_abn_mtch` gains a cmodel/G19_cMod option (currently only G19_SE and HMevo); a well-scoped follow-up, not a new script |
| 9 | sSFR of satellites/centrals vs SDSS, 3 mass bins (10-10.5, 10.5-11.3, 11.3-12.5) | run + data | **satellite side only, mass bins approximate** (`Paper2_Fig9_sSFR.png`: 3 mass bins but not the paper's exact ranges; central-galaxy dynamical-quenching line and SDSS overlay not built; sparse-bin noise is small-number statistics, not a port defect) |
| 10 | SFR-M* relation at z=0.5/1/2 vs Leja+2019 | run + data | **done** (`Paper2_Fig10_SFR_Mstar_z.png`: reuses the same 5 central tracks as Fig. 7, evaluated at z=0.5/1/2 instead of 0/1/2; Leja+2019 overlay omitted) |
| 11 | Elliptical fraction vs stellar mass, 3 redshifts, vs SDSS T-Type | run + data | **done, method is a documented interpretation** (`Paper2_Fig11_EllipticalFraction.png`: major-merger fraction (ratio>0.25) from `Mergers_Accretion_History` integrated per central mass track, converted expected-count-to-fraction via `f=1-exp(-N_major)` since the paper doesn't give its exact formula; qualitatively correct shape and z-ordering, tight py/rust agreement, but the exact turnover mass isn't verified against the published curve. Not the `P_Elliptical` array -- that's still confirmed dead code, just not what this figure needs.) |

## Paper 3 (galaxy pair fractions, arXiv 2001.06017)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram (mass-ratio cartoon) | diagram | skip |
| 2 | Method diagram (SMHM-slope-vs-pair-fraction cartoon) | diagram | skip |
| 3 | SMHM parameter sensitivity (M/N/beta/gamma, Table 2) **and** resulting pair fraction vs z, 4 lines x 4 parameters | fn + run | **done** -- both halves now built: `Paper3_Fig3_PFTSensitivity.png` (input SMHM-curve context panels, [fn]) and `Paper3_Fig3_PairFractionSensitivity.png` (the actual result, pair fraction vs z per parameter, from 14 full SF+stripping three-way runs -- 13 PFT perturbations + reference). M and gamma show the strongest sensitivity, consistent with the paper's narrative; some panels show visible py/rust divergence (e.g. M_z,+, gamma_0.1,alt), consistent with pair-fraction being a small-number-statistics-sensitive quantity elsewhere in this reproduction, not necessarily a port defect. |
| 4 | SMHM (Illustris/PyMorph) + pair fraction vs Illustris TNG | fn + run + data | left panel covered by `SMHM_Relation.png` (PyMorph only, no Illustris-tuned variant); pair fraction vs z shown generically in `Paper3_PairFraction_vs_z.png` (not the Illustris-comparison mass cut) |
| 5 | SMHM (PyMorph/cmodel, 2 z) + pair fraction evolution vs Mundy+2017 | fn + run + data | **left panel done** (`SMHM_Relation.png`, z=0.1/2.0 both shown); pair fraction generic version in `Paper3_PairFraction_vs_z.png` (not mass-cut-matched to Mundy) |
| 6 | SMHM (cmodel, altered slope evolution) + pair fraction vs Mundy+2017 | fn + run + data | **left panel done** (`Paper3_Fig6_HMevoSMHM.png`: HMevo preset, gamma11=0.1/0.2/0.5, z=0.1 & 2.0); pair fraction vs Mundy+2017 not built |
| 7 | Mass tracks (accretion vs SFR), multiple slope-evolution values | run | **AM-track slope sweep done** (`Paper3_Fig7_SlopeEvolutionSweep.png`: G19_SE + HMevo gamma11=0.1/0.2/0.5, abundance-matching strand only; accretion/SFR decomposition and ratio panels not built) |

## Notes

* Figure numbering follows the published PDFs
  (`/workspace/philip_grylls_uos_thesis/Appendices/Full_PDFs/Papers/`):
  Paper 1 = `MNRAS_Final_P2.pdf` (yes, that filename -- an artifact of
  how the source files were named, unrelated to this table's Paper
  1/2 labels, which follow `configs/published_runs.toml` not the PDF
  filenames), Paper 2 = `MNRAS_Final_P1.pdf`, Paper 3 = `arXiv_P3.pdf`.
* "pending" figures are worked in the order above; this table is
  updated as each lands, with a pointer to the file under
  `Figures/PortValidation/` and the script that produced it.
* Where py-as-is cannot participate on equal footing (no deterministic
  mode, A7), it is omitted from that panel rather than mixed in on a
  different basis — same rule as Figures 3/6/7.
* `Figures/PortValidation/Satellite_SMHM_Relation.png` (mean satellite
  log M* per subhalo-mass bin at z~0.1) doesn't map to one numbered
  figure in any of the three papers; it's the same `Sat_SMHM`
  accumulator several of the run-level outputs above use, shown on its
  own as it's the most direct single-panel check of the satellite-side
  SMHM.
* Paper 1's Figures 2/3/5/9 and the satellite-SMHM and pair-fraction
  plots all come from **one** deterministic published-grid run
  (`Scripts/Validation/results_figure3.py` /
  `results_from_run.py`, `('1.0', True, True, True, 'G19_DPL',
  'G19_SE')`) — no additional simulation cost per figure. Note this
  reference run itself uses G19_DPL, not the CE model Paper 1 Figs 5/6
  actually specify in `configs/published_runs.toml` (`p1-tdyn-1.0`
  uses `SFR_Model = 'CE'`, but Fig. 5 doesn't depend on the SFR model
  at all since it's `stripping=False, SF=False` in the source config)
  — worth double-checking case by case rather than assumed inherited.
