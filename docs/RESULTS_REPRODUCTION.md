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

## Paper 1 (thesis Ch. 3 / MNRAS central-galaxy SFR paper)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram | diagram | skip |
| 2 | Method diagram | diagram | skip |
| 3 | SMHM relation (PyMorph/cmodel/M13) + central SMF vs Illustris | fn + data | pending |
| 4 | Merger rate per Gyr at fixed halo mass vs Fakhouri+2010 | run + data | pending |
| 5 | Satellite number density vs parent halo mass, multi-z, vs SDSS/Illustris | run + data | pending |
| 6 | Mass tracks (PyMorph): accretion vs SFR mass budget | run | pending |
| 7 | SFR-M* relation from central tracks vs SDSS | run + data | pending |
| 8 | Mass tracks (cmodel) | run | pending |
| 9 | sSFR of satellites/centrals vs SDSS, 3 mass bins | run + data | pending |
| 11 | Elliptical fraction vs stellar mass, 3 redshifts, vs SDSS | run + data | pending |

## Paper 2 (main satellite SMF paper)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram | diagram | skip |
| 2 | USHMF vs USSHMF (3 f_tdyn), one parent halo mass | fn | pending |
| 3 | Total USHMF vs total USSHMF (3 f_tdyn) | fn | pending |
| 4 | SMHM (G18) vs B18/S17/M13 + central SMF vs SDSS | fn + data | pending |
| 5 | % satellites by accretion redshift | run | pending |
| 6 | Quenching delay time-scale (Wetzel+F16) | fn | **done** (`Paper2_Fig6_Quenching.png`) |
| 7 | Dynamical-friction merging time-scale | fn | **done** (`Paper2_Fig7_MergerTimescale.png`) |
| 8 | SSMF, f_tdyn=1.0 frozen/evolving, vs SDSS | run + data | pending |
| 9 | Satellite distributions by parent halo mass, vs SDSS | run + data | pending |
| 10 | SSMF, f_tdyn=0.5/1.0/2.5, vs SDSS | run + data | pending |
| 11 | Satellite distributions, f_tdyn=0.5/1.0/2.5, vs SDSS | run + data | pending |
| 12 | SSMF, Tomczak vs continuity SFR, vs SDSS | run + data | pending |
| 13 | sSFR distributions, T16 vs CE, vs SDSS | run + data | pending |
| 14 | SSMF, frozen/SF/SF+strip, vs SDSS | run + data | (Fig. 3 above is the SF+strip case of this family) |
| 15 | Satellite distributions, frozen/SF/SF+strip, vs SDSS | run + data | pending |

## Paper 3 (pair-fraction / SMHM-systematics paper)

| Fig | Content | Class | Status |
|---|---|---|---|
| 1 | Method diagram | diagram | skip |
| 2 | Method diagram | diagram | skip |
| 3 | Discussion of high-mass-slope evolution (text figure, no new run) | fn | pending |
| 4 | SMHM (Illustris/PyMorph) + pair fraction vs Illustris TNG | fn + run + data | pending |
| 5 | SMHM (PyMorph/cmodel, 2 z) + pair fraction evolution vs Mundy+2017 | fn + run + data | pending |
| 6 | SMHM (cmodel, altered slope evolution) + pair fraction vs Mundy+2017 | fn + run + data | pending |
| 7 | Mass tracks (accretion vs SFR), multiple slope-evolution values | run | pending |

## Notes

* Figure numbering follows the published PDFs
  (`/workspace/philip_grylls_uos_thesis/Appendices/Full_PDFs/Papers/`).
* "pending" figures are worked in the order above; this table is
  updated as each lands, with a pointer to the file under
  `Figures/PortValidation/` and the script that produced it.
* Where py-as-is cannot participate on equal footing (no deterministic
  mode, A7), it is omitted from that panel rather than mixed in on a
  different basis — same rule as Figures 3/6/7.
