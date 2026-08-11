# Three-way validation harness

Runs and compares the three implementations the MNRAS letter reports on:

| Name | What it is | How to run |
|---|---|---|
| **py-as-is** | `STEEL.py`, byte-for-byte unmodified, on the period-correct `env/py-asis` stack | `run_py_steel.py` |
| **py-corrected** | the same, with the defects in `docs/PORT_CORRECTIONS.md` fixed one commit at a time (branch `py-steel-corrected`) | `run_py_steel.py` |
| **rs-steel** | the Rust port | `rust/target/release/steel <runfile.toml> <outdir>` |

See `env/README.md` for how the two Python environments are built and
why there are two.

## Running the Python side

`STEEL.py` is a script with its grid, run list and a blocking `input()`
prompt all baked in, so `run_py_steel.py` generates a patched copy with
three anchored substitutions and runs that. The original is never
touched — that is the whole point of the `py-asis` environment.

```bash
# reduced grid: ~30 s, the size used for day-to-day validation
env/py-asis/bin/python Scripts/Validation/run_py_steel.py \
    --halo-min 11.0 --halo-max 12.6 --halo-bin 0.5 \
    --run "1.0,False,False,True,G19_DPL,Moster"

# full published resolution: ~45 min
env/py-asis/bin/python Scripts/Validation/run_py_steel.py \
    --run "1.0,True,True,True,G19_DPL,G19_SE"
```

Output lands in `Data/Model/Output/RunFiles/RunParam_<params>_/`, which
is gitignored.

## Running the Rust side

Runfiles matching the above live in `rust/runfiles/`:

```bash
cargo build --release --manifest-path rust/Cargo.toml
./rust/target/release/steel rust/runfiles/reduced-grid.toml      /tmp/rs-reduced
./rust/target/release/steel rust/runfiles/steel-sf-stripping.toml /tmp/rs-full-sf
```

The Rust writes the same `RunParam_<params>_/` directory names and the
same `.npy` file names, so the Python `LoadData_*` family and the
plotting scripts read either tree unmodified.

## Running all three at once

`three_way.py` drives every leg and compares them:

```bash
# numerical fidelity: scatter off, py-corrected vs rs-steel
env/py-asis/bin/python Scripts/Validation/three_way.py --mode deterministic

# what the corrections change: scatter on, all three legs, 5 seeds
git worktree add ../STEEL-asis --detach claude/phd-code-rust-plan-zqyvff
cd ../STEEL-asis/Functions &&     ../../STEEL/env/py-asis/bin/python Setup.py build_ext --inplace && cd -
env/py-asis/bin/python Scripts/Validation/three_way.py --mode stochastic --seeds 1 2 3 4 5
```

The py-as-is worktree must be **detached**, not on a branch: the branch
carrying the unmodified `STEEL.py` is also the development branch, and
git will not check the same branch out twice.

Results are recorded in `docs/VALIDATION.md`.

## Comparing individual trees

```bash
env/py-legacy/bin/python Scripts/Validation/compare_runs.py \
    Data/Model/Output/RunFiles/RunParam_1.0_False_False_True_G19_DPL_Moster_ \
    /tmp/rs-reduced/RunParam_1.0_False_False_True_G19_DPL_Moster_
```

**Two comparison modes, and they mean different things.**

* *Deterministic* — scatter disabled on both sides. Same arithmetic on
  the same grid; agreement should be at floating-point level, modulo the
  deliberate corrections in `docs/PORT_CORRECTIONS.md`. This is the
  strong numerical-fidelity claim.
* *Stochastic* — scatter on. py-steel draws from NumPy's Mersenne
  Twister and GSL's `taus` (inside `Functions_c`), rs-steel from
  `rand`'s ChaCha. These can never agree element-wise, only in ensemble
  statistics; per-bin fractional deviations of order the Monte-Carlo
  noise are expected and are not defects.

`compare_runs.py` reports numbers and does not know which mode produced
its inputs. Quoting a single blended "agreement" figure across both
would conflate numerical fidelity with Monte-Carlo noise, which is
exactly the distinction the validation exists to draw.

## Known intended differences

These are corrections, not regressions — see `docs/PORT_CORRECTIONS.md`:

* `Satellite_sSFR` / `sSFR_Range` have 60 bins in rs-steel, 59 in
  py-as-is (C2).
* `Sat_SMHM_Sat_SMHM` and `..._Host` drop py-as-is's unwritten trailing
  padding slot (D2).
* `Figure4_6_*` and `Sat_Env_Highz_*` integrals differ by ~8-10% (C1).
* rs-steel additionally writes `Figure3_z.npy` and the
  `Surviving_Subhalos*.npy` pair, which py-steel writes as `.dat`/`.png`
  under `Data/Model/Output/Other/SubHaloes/`.
* rs-steel omits the `Surviving_Subhalos*` files entirely when stripping
  or star formation is on, where py-steel saves arrays of zeros.
