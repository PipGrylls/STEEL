# The `py-steel-corrected` branch

`py-steel-corrected` is the Python side of the three-way validation: the
same STEEL, with the defects found by porting it to Rust
(`docs/PORT_CORRECTIONS.md`) fixed **one concern per commit**, so each
one's effect on the published outputs can be measured in isolation.

It branches from the Rust port's tip, so the Rust workspace and the
validation harness are present here too. `master` and
`claude/phd-code-rust-plan-zqyvff` carry `STEEL.py` unmodified — that is
the *py-as-is* leg, and it must stay that way.

## The commit series

| # | Commit | Category | Measured effect on the reduced validation grid |
|---|---|---|---|
| 1 | index with slices, not a non-tuple sequence | E | none (lets it run on NumPy ≥ 1.23) |
| 2 | seed the generators once per run, not per call | A5 | run becomes bit-reproducible |
| 3 | convert per-ln to per-dex with `ln(10)` | A4 | +0.11% on every number density |
| 4 | pair fractions for evolved satellites too | B1 | `Pair_Frac` 0 → 1.17e+02; `Pair_Frac_Halo` 0 → 1.86e+03 |
| 5 | histogram convention, not `np.digitize` | C1 | cut integrals +0.6% to +75%, cut-dependent |
| 6 | sSFR bins match the axis saved beside them | C2 | `Satellite_sSFR` (40,59) → (40,60) |
| 7 | always define `WeightList_SubOnly` | B3 | none on this grid (defensive) |
| 8 | fix two never-firing shell/cache checks | B2, D1 | none numerically; ~1 s/process |
| 9 | Schreiber main sequence clamped as published | A1 | S16CE only: total SF −16.7%, SMF per-bin 0.35×–2.2× |
| 10 | no log-ratio added into a linear-Msun total | A3 | < 1e-9 relative — real defect, negligible effect |
| 11 | evolve satellites to their return epoch | A2 | every evolved output; pair fractions −0.95% |
| 12 | drop the unwritten `Sat_SMHM` padding column | D2 | shape only |

Commits 2–12 quote their own measurements; the numbers above are
summaries. All are from the reduced grid (`--halo-min 11.0 --halo-max
12.6 --halo-bin 0.5`) with the
`('1.0', True, True, True, 'G19_DPL', 'G19_SE')` tuple, except #9 which
needs `'S16CE'` to exercise the Schreiber branch at all. Magnitudes are
grid-dependent; signs and mechanisms are not.

## How each one was measured

Commit 2 is what makes the rest measurable. Before it, `STEEL.py`
reseeded NumPy's global generator from the wall clock inside
`DarkMatterToStellarMass`, so no two runs agreed and no correction's
effect could be separated from the noise. After it, a run is
bit-reproducible and a correction's footprint is exactly the set of
files that change.

`Scripts/Validation/measure_correction.sh <label>` runs the reduced
grid, snapshots the output tree, and diffs it against the previous
snapshot:

```bash
export STEEL_SNAPSHOT_DIR=/tmp/steel-corrections
Scripts/Validation/measure_correction.sh before-my-change
# ... make the change ...
Scripts/Validation/measure_correction.sh after-my-change
```

## What is deliberately *not* fixed here

* `Stripping_DM` remains dead (`Stripping_DM = False #Future use`), and
  with it `HaloMassLoss_c`'s two bugs — the hardcoded `Ol = 0` and the
  sign-flipped overdensity term. Fixing unreachable code would add risk
  without changing any result. Recorded in `docs/PORT_CORRECTIONS.md`.
* `Functions.py`'s dead `HM_SM` and `HMF_fit` assignments, and the
  unused `halotools`/`hmf` imports behind them.
* `scipy.interpolate.interp2d` and the other removed APIs listed in
  `env/README.md`. Those are the post-processing cleanup (Phase 5), not
  model corrections; `STEEL.py` itself runs on `env/py-legacy` once
  commit 1 lands.

## Reproducing

```bash
git checkout py-steel-corrected
cd Functions && ../env/py-asis/bin/python Setup.py build_ext --inplace && cd ..
env/py-asis/bin/python Scripts/Validation/run_py_steel.py \
    --run "1.0,True,True,True,G19_DPL,G19_SE"
```

`STEEL_SEED` (default 42) sets the run's seed. Two runs with the same
seed produce byte-identical output; verified across all 55 output files.
