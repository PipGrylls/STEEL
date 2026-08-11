# The `py-steel-corrected-pipgrylls` branch

The Python side of the three-way validation: STEEL as the papers were
run, with the defects found by porting it to Rust
(`docs/PORT_CORRECTIONS.md`) fixed **one concern per commit**, so each
one's effect on the published outputs can be measured in isolation.

## Which baseline, and why the branch was rebuilt

`master`'s tip is 2019-03-04, before all three papers. `PipGrylls` is 26
commits ahead of it on the model core and carries `bfdb4d8`, *"This is
the version of the code used for the 1st submission of Paper2 on
02/05/19"*. The first corrections branch was cut against `master`, which
means every number it measured was measured against code the papers were
not run from.

Both series are kept:

| Branch | Baseline | Status |
|---|---|---|
| `py-steel-corrected` | `master` | superseded; kept for the record |
| `py-steel-corrected-master-baseline` | `master` | identical snapshot of the above |
| **`py-steel-corrected-pipgrylls`** | **`PipGrylls`** | **current** |

The current branch cuts from `claude/phd-code-rust-plan-zqyvff` *after*
that branch merged `origin/PipGrylls`, so the Rust workspace and the
validation harness are present here too. The py-as-is leg is a detached
worktree at `origin/PipGrylls` itself, which must stay byte-for-byte
unmodified.

## The commit series

| # | Commit | Category | Measured effect on the reduced validation grid |
|---|---|---|---|
| 1 | index with slices, not a non-tuple sequence | E | none (lets it run on NumPy ≥ 1.23) |
| 2 | seed the generators once per run, not per call | A5 | run becomes bit-reproducible |
| 3 | convert per-ln to per-dex with `ln(10)` | A4 | +0.11% on every number density |
| 5 | histogram convention, not `np.digitize` | C1 | cut integrals +0.6% to +75%, cut-dependent |
| 6 | sSFR bins match the axis saved beside them | C2 | `Satellite_sSFR` (40,59) → (40,60) |
| 7 | always define `WeightList_SubOnly` | B3 | none on this grid (defensive) |
| 8 | fix two never-firing shell/cache checks | D1 | `rm -r` already fixed on `PipGrylls`; `mkdir` without `-p` was not |
| 9 | Schreiber main sequence clamped as published | A1 | S16CE only: total SF −16.7%, SMF per-bin 0.35×–2.2× |
| 10 | no log-ratio added into a linear-Msun total | A3 | < 1e-9 relative — real defect, negligible effect |
| 11 | evolve satellites to their return epoch | A2 | every evolved output; also rewrites the pair-fraction weights to read `SM_Window` |
| 12 | drop the unwritten `Sat_SMHM` padding column | D2 | shape only |
| 13 | `STEEL_SCATTER` master switch | A7 | makes deterministic mode possible at all |
| 14 | make the gas-supply cap actually engage | A6 | see `docs/VALIDATION.md` |
| 15 | make `Halogrowth` able to run off one machine | G3 | none directly; unblocks 16 |
| 16 | pass the cosmology's actual spectral index | G1 | `AvaHaloMass` max 0.069 dex; `Figure4_6` integrals median 1.0%, max 11.9%; total SF −8.3% |

**Number 4 is deliberately absent.** It was *"compute pair fractions for
evolved satellites too"* (B1), and `PipGrylls` already does — as do
`Paper2`, `Refactor` and `saiduc`. It is a `master`-only defect, and
applying it here would have been correcting something that was never
broken in the published code.

Two commits are new to this series and have no counterpart in the
`master`-baselined one: 15 and 16, both `PipGrylls`-only (see
`docs/PORT_CORRECTIONS.md` §G). A third block of work — the post-processing
cleanup — had to be largely redone, because `PipGrylls`'s
`Scripts/CentralPostprocessing.py` is 2844 lines longer than `master`'s
and reintroduces the same removed APIs in the new code, plus a
`KeyError` that stops any of it running at all (§F7).

## How each one was measured

Commit 2 is what makes the rest measurable. Before it, `STEEL.py`
reseeded NumPy's global generator inside `DarkMatterToStellarMass` on
every call, so no two runs agreed and no correction's effect could be
separated from the noise.

`Scripts/Validation/measure_correction.sh <label>` runs the reduced
grid, snapshots the output tree, and diffs it against the previous
snapshot:

```bash
export STEEL_SNAPSHOT_DIR=/tmp/steel-corrections
Scripts/Validation/measure_correction.sh before-my-change
# ... make the change ...
Scripts/Validation/measure_correction.sh after-my-change
```

Commit 16 needs one extra step, because its effect is carried by a
cached file rather than by the source: swap the mass-accretion-history
table between snapshots with

```bash
env/py-legacy/bin/python Scripts/Validation/make_mah_table.py \
    --cosmology {pipgrylls|corrected} --halo-min 11.0 --halo-max 12.6 \
    --halo-bin 0.5 --out Data/Model/Input/11.012.60.50.6774.dat
```

`Get_HM_History` keys its cache on `<min><max><bin><h>` and not on the
cosmology, so without that swap the correction measures nothing.

## What is deliberately *not* fixed here

* `Stripping_DM` remains dead (`Stripping_DM = False #Future use`), and
  with it `HaloMassLoss_c`'s two bugs — the hardcoded `Ol = 0` and the
  sign-flipped overdensity term. Fixing unreachable code would add risk
  without changing any result. Recorded in `docs/PORT_CORRECTIONS.md`.
* `Functions.py`'s dead `HMF_fit` assignment.
* The `Strip_f[Strip_f>1] = 1` clamp beside `PipGrylls`'s doubled tidal
  stripping. `Strip_f` is a log10 of a quantity ≤ 1 and so is always
  ≤ 0; the clamp cannot fire, and reproducing it would be reproducing
  nothing. The doubling itself is *not* touched — it is the published
  model, not a defect.
* The commented-out second doubling in `Functions.py::StarFormation`
  (`#For reviwer`). Applying both would quadruple the stripping.
* The MAH cache key. Adding the cosmology to it would be correct, and
  would invalidate every table users already have on disk; the
  validation harness stamps the cosmology beside each table instead.

## Reproducing

```bash
git checkout py-steel-corrected-pipgrylls
cd Functions && ../env/py-asis/bin/python Setup.py build_ext --inplace && cd ..
env/py-legacy/bin/python Scripts/Validation/make_mah_table.py \
    --cosmology corrected --halo-min 11.0 --halo-max 16.6 --halo-bin 0.1 \
    --out Data/Model/Input/11.016.60.10.6774.dat
env/py-asis/bin/python Scripts/Validation/run_py_steel.py \
    --run "1.0,True,True,True,G19_DPL,G19_SE"
```

`STEEL_SEED` (default 42) sets the run's seed; `STEEL_SCATTER=0`
disables every stochastic draw. Two runs with the same seed produce
byte-identical output, verified across all 55 output files.
