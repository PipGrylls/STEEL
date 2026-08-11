# Validation environments

Two Python environments are needed, because **the original code does not run on a
current scientific Python stack**. Both are reproduced from the committed
lockfiles; the venvs themselves are gitignored.

| Env | Python | Purpose |
|---|---|---|
| `py-asis` | 3.10 | Runs `STEEL.py` **completely unmodified**. Period-correct pins. |
| `py-legacy` | 3.11 | Newer stack, for the `py-steel-corrected` branch and post-processing. |

## Why two

`STEEL.py` fails on any `numpy >= 1.23`. It indexes with a *list* of arrays:

```python
ix = [np.arange(z_bin, i), np.full_like(np.arange(z_bin, i), Bin)]
SurvivingSubhalos[ix] = ...        # STEEL.py:300-301
```

Multidimensional indexing with a non-tuple sequence was deprecated in NumPy 1.15
and **removed in 1.23**. After removal the list is treated as a single fancy
index on axis 0, so the shapes silently change and the run dies with:

```
ValueError: operands could not be broadcast together with shapes (2,25,5) (25,)
```

`numpy 1.22` is therefore the newest usable version, and since numpy 1.22 has no
CPython 3.11 wheels, the as-is environment is pinned to Python 3.10.

Other incompatibilities found while building these (all in post-processing /
fitting rather than `STEEL.py` itself, so they don't affect the as-is run):

| API | Used at | Removed in |
|---|---|---|
| `scipy.interpolate.interp2d` | `Functions.py:229`, `CentralPostprocessing.py:236` | scipy 1.14 |
| `scipy.integrate.cumtrapz` | `CentralPostprocessing.py:20` | scipy 1.14 |
| `pd.Series.get_values()` | `CentralPostprocessing.py:233,235` | pandas 1.0 |
| `delim_whitespace=` | `SDSS_Plots.py:45,52`, `SMHM_Fit.py:156+` | pandas 2.2 |

## Rebuilding

```bash
sudo apt-get install -y libgsl-dev gfortran

python3.10 -m venv env/py-asis
env/py-asis/bin/pip install "pip<24.1"          # hmf's metadata is invalid for pip >= 24.1
env/py-asis/bin/pip install -r env/py-asis-requirements.txt

python3 -m venv env/py-legacy
env/py-legacy/bin/pip install -r env/py-legacy-requirements.txt

# Cython extension, once per env (from inside Functions/):
cd Functions && ../env/py-asis/bin/python Setup.py build_ext --inplace
```

`Functions.py` imports `Functions_c` at module level, so nothing runs until that
extension builds. The `.so` is Python-version-specific — rebuild when switching
envs.

## The Fortran binary

The committed `Functions/OtherModels/VDB13/getPWGH` links against
`libgfortran.so.3` (GCC 6 era) and cannot run on a current system:

```
./getPWGH: error while loading shared libraries: libgfortran.so.3
```

It has been rebuilt in place from the repo's own `.f` sources:

```bash
cd Functions/OtherModels/VDB13
gfortran -c getPWGH.f cosmo_sub.f quadpack.f
gfortran -o getPWGH getPWGH.o cosmo_sub.o quadpack.o -lm && rm -f *.o
```

`Functions.py::Get_HM_History` shells out to this per halo mass, so halo growth
histories cannot be generated at all without it. Note the repo's stale
`getPWGH.in` is missing its output-filename line and will fail if used directly;
`Halogrowth` writes its own input file, which is correct.

## Dead dependencies

`halotools` and `hmf` are imported but never used — `HM_SM = EM.Moster13SmHm(...)`
(`Functions.py:22`) and `HMF_fit = hmf.fitting_functions.Tinker10`
(`Functions.py:214`) are both assigned and never read; the halo mass function
actually used comes from COLOSSUS (`despali16`). They are installed anyway so the
as-is run is genuinely unmodified, but they are removable.
