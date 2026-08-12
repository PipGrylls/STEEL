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

**Running count** (results figures only, diagrams excluded): Paper 1
5/8 touched (1 fully, 4 partial); Paper 2 14/14 touched (6 fully, 8
partial); Paper 3 3/5 touched (0 fully, 3 partial). 24 figures under
`Figures/PortValidation/` so far -- Paper 2 fully touched. Remaining
pending: Paper 3 Fig. 3 (a discussion figure, not a new run -- may
not need a separate reproduction). Fresh simulation runs beyond the
published-grid deterministic run (§Figure 3) and the frozen-model run
(§Fig. 14/2): f_tdyn=0.5/2.5 (§Fig. 10/11) and T16 SFR (§Fig. 12/13).

## Paper 1 (thesis Ch. 3 / MNRAS central-galaxy SFR paper)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram | diagram | skip |
| 2 | Method diagram | diagram | skip |
| 3 | SMHM relation (PyMorph/cmodel/M13) + central SMF vs Illustris | fn + data | **left panel done** (`SMHM_Relation.png`, M13/Illustris omitted) |
| 4 | Merger rate per Gyr at fixed halo mass vs Fakhouri+2010 | run + data | **done** (`Paper1_Fig4_MergerRate.png`: 4 halo masses; all mass ratios, not major-mergers-only; Fakhouri+2010 band omitted) |
| 5 | Satellite number density vs parent halo mass, multi-z, vs SDSS/Illustris | run + data | **done** (`Paper1_Fig5_SatelliteDistribution_MultiZ.png`: 6 redshifts, M*>10^10 cut; SDSS/Wang+16/Wen&Han+18/Illustris overlays omitted) |
| 6 | Mass tracks (PyMorph): accretion vs SFR mass budget | run | **top-panel strand done** (`MassTrack.png`: abundance-matching vs. in-situ-SFR-only, one target mass; accretion-decomposition and middle/bottom ratio panels not built; py-as-is absent, G3) |
| 7 | SFR-M* relation from central tracks vs SDSS | run + data | **done** (`Paper1_Fig7_SFR_Mstar.png`: main-sequence relation at z=0/1/2 from 5 central mass tracks per leg, G19_DPL SFR recomputed from each track's in-situ mass at that z -- path-dependent track output, not the bare closed-form function; py-as-is omitted, G3; SDSS overlay omitted) |
| 8 | Mass tracks (cmodel) | run | same coverage as Fig. 6 above, but `MassTrack.png` uses G19_SE (PyMorph) not cmodel -- not yet re-run with G19_cMod |
| 9 | sSFR of satellites/centrals vs SDSS, 3 mass bins | run + data | **satellite side done** (`Paper1_Fig9_sSFR.png`, 3 mass bins; central-galaxy post-processed line and SDSS overlay not built; sparse-bin noise is small-number statistics, not a port defect -- same caveat as Fig. 3's p90 column) |
| 11 | Elliptical fraction vs stellar mass, 3 redshifts, vs SDSS | run + data | **infeasible from STEEL.py itself** -- `P_Elliptical = np.full((a,b), 0.)` (`STEEL.py:288`) is the *only* occurrence of `P_Elliptical` in the file: allocated, never written, never saved. Confirmed by grep, not inferred. `steel-io`'s own doc comment already flags this as dead code it deliberately doesn't reproduce. Whatever generated the published figure isn't in `OneRealization`. |

## Paper 2 (main satellite SMF paper)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram | diagram | skip |
| 2 | USHMF vs USSHMF (3 f_tdyn), one parent halo mass | fn | **one line done** (`Paper2_Fig2_USSHMF.png`: USSHMF at f_tdyn=1.0 only, same log Mh,parent=12.80 as the paper; USHMF reference curve and the other 2 f_tdyn values not built; reclassified from [fn] -- it's actually a `run` output, `Surviving_Subhalos_ByParent`) |
| 3 | Total USHMF vs total USSHMF (3 f_tdyn) | fn | **one line done** (`Paper2_Fig3_TotalUSSHMF.png`: total USSHMF at f_tdyn=1.0 only; USHMF reference curve and other 2 f_tdyn values not built; reclassified from [fn] like Fig. 2 -- a `run` output, `Surviving_Subhalos`) |
| 4 | SMHM (G18) vs B18/S17/M13 + central SMF vs SDSS | fn + data | **left panel done** (`SMHM_Relation.png`, using G19 not G18; B18/S17/M13 omitted) |
| 5 | % satellites by accretion redshift | run | **done** (`Paper2_Fig5_AccretionRedshift.png`) |
| 6 | Quenching delay time-scale (Wetzel+F16) | fn | **done** (`Paper2_Fig6_Quenching.png`) |
| 7 | Dynamical-friction merging time-scale | fn | **done** (`Paper2_Fig7_MergerTimescale.png`) |
| 8 | SSMF, f_tdyn=1.0 frozen/evolving, vs SDSS | run + data | **done** (`Paper2_Fig8_FrozenVsEvolving.png`: frozen [no SF/stripping] vs evolving [SF+stripping] at f_tdyn=1.0, reusing the same runs as Fig. 14's first two lines; SFR model is G19_DPL not CE since SF=False makes the frozen line's SFR-model choice inert and CE has no run built; SDSS overlay omitted) |
| 9 | Satellite distributions by parent halo mass, vs SDSS | run + data | **one line done** (`Paper2_Fig9_SatelliteDistribution.png`: f_tdyn=1.0 evolving, M*>10^10 cut only; SDSS band and other 3 lines/columns omitted) |
| 10 | SSMF, f_tdyn=0.5/1.0/2.5, vs SDSS | run + data | **done** (`Paper2_Fig10_TdynSweep.png`: all 3 f_tdyn values, each py-corrected vs rs-steel; SDSS overlay omitted) |
| 11 | Satellite distributions, f_tdyn=0.5/1.0/2.5, vs SDSS | run + data | **done** (`Paper2_Fig11_TdynSweep.png`: all 3 f_tdyn values, M*>10^10 cut, each py-corrected vs rs-steel; SDSS overlay omitted) |
| 12 | SSMF, Tomczak vs continuity SFR, vs SDSS | run + data | **T16 vs G19_DPL done** (`Paper2_Fig12_SFRModelSweep.png`; the paper's second line is CE, not G19_DPL -- no CE run built -- and SDSS overlay omitted) |
| 13 | sSFR distributions, T16 vs CE, vs SDSS | run + data | **T16 vs G19_DPL done** (`Paper2_Fig13_sSFRSweep.png`, 3 mass bins; paper's second model is CE not G19_DPL, no CE run built; SDSS overlay omitted; same sparse-bin small-number-statistics caveat as Fig. 9) |
| 14 | SSMF, frozen/SF/SF+strip, vs SDSS | run + data | **2 of 3 lines done** (`Paper2_Fig14_ConfigSweep.png`: frozen and SF+stripping, each py-corrected vs rs-steel; SF-only line and SDSS overlay not built) |
| 15 | Satellite distributions, frozen/SF/SF+strip, vs SDSS | run + data | **2 of 3 lines done** (`Paper2_Fig15_ConfigSweep.png`: frozen and SF+stripping, M*>10^10 cut, top row only; SF-only line, bottom fractional row, and SDSS overlay not built) |

## Paper 3 (pair-fraction / SMHM-systematics paper)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram | diagram | skip |
| 2 | Method diagram | diagram | skip |
| 3 | Discussion of high-mass-slope evolution (text figure, no new run) | fn | pending |
| 4 | SMHM (Illustris/PyMorph) + pair fraction vs Illustris TNG | fn + run + data | left panel covered by `SMHM_Relation.png` (PyMorph only, no Illustris-tuned variant); pair fraction vs z shown generically in `Paper3_PairFraction_vs_z.png` (not the Illustris-comparison mass cut) |
| 5 | SMHM (PyMorph/cmodel, 2 z) + pair fraction evolution vs Mundy+2017 | fn + run + data | **left panel done** (`SMHM_Relation.png`, z=0.1/2.0 both shown); pair fraction generic version in `Paper3_PairFraction_vs_z.png` (not mass-cut-matched to Mundy) |
| 6 | SMHM (cmodel, altered slope evolution) + pair fraction vs Mundy+2017 | fn + run + data | **left panel done** (`Paper3_Fig6_HMevoSMHM.png`: HMevo preset, gamma11=0.1/0.2/0.5, z=0.1 &amp; 2.0); pair fraction vs Mundy+2017 not built |
| 7 | Mass tracks (accretion vs SFR), multiple slope-evolution values | run | **AM-track slope sweep done** (`Paper3_Fig7_SlopeEvolutionSweep.png`: G19_SE + HMevo gamma11=0.1/0.2/0.5, abundance-matching strand only; accretion/SFR decomposition and ratio panels not built) |

## Notes

* Figure numbering follows the published PDFs
  (`/workspace/philip_grylls_uos_thesis/Appendices/Full_PDFs/Papers/`).
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
* Figures 3, 5, 9, and the satellite-SMHM and pair-fraction plots all
  come from **one** deterministic published-grid run
  (`Scripts/Validation/results_figure3.py` /
  `results_from_run.py`, `('1.0', True, True, True, 'G19_DPL',
  'G19_SE')`) — no additional simulation cost per figure.
