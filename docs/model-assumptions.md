# Model assumptions: STEEL, EMERGE, UniverseMachine

What each model assumes, per `rs-steel` trait, so cross-model comparisons
are like-for-like. Referenced by the composition validator's error output.

Classification: **identical** (no action) / **convertible** (handled by
`steel_plugins::harmonise`) / **needs plugin** (candidate follow-up spec)
/ **structurally absent** (no analogue; a documented asymmetry).

| Trait | STEEL | EMERGE | UniverseMachine | Status |
|---|---|---|---|---|
| `Cosmology` | Planck15 | Planck15 | Planck15 | convertible |
| `HaloGrowthModel` | van den Bosch 2014 average MAH | N-body trees | N-body trees | needs plugin |
| `HaloMassFunctionModel` | Despali 2016 | implicit in the simulation | implicit in the simulation | needs plugin |
| `SubhaloMassFunctionModel` | Jiang & van den Bosch 2016 | resolved subhaloes | resolved subhaloes | structurally absent upstream |
| `MergerTimescaleModel` | Boylan-Kolchin 2008 / McCavana 2012 | tracked directly | tracked directly | structurally absent upstream |
| `HaloStrippingModel` | van den Bosch 2005 / Cattaneo 2011 | from the tree | from the tree | structurally absent upstream |
| `SmhmModel` | Moster form, G19_SE default | not used (rate-based) | not used (rate-based) | n/a |
| `StellarGrowthModel` | not used | eps x baryonic accretion | SFR from halo properties | identical interface |
| `SfrModel` | Tomczak/Schreiber forms, M*-keyed | implied by the rate | bimodal PDF, v-keyed | needs plugin |
| `QuenchingModel` | Wetzel 2013 timescales | reionization gate (early growth, not satellite) | internal to the SFR PDF | conflict — see below |
| `GasMassModel` | Stewart scaling | not modelled (predicts M* directly via SFE integrated along the accretion track; no gas reservoir stage appears anywhere in the ported code or provenance) | not modelled (SFR / quenched-fraction are evaluated directly from vMpeak and z; no gas reservoir stage) | needs plugin |
| `StellarStrippingModel` | STEEL prescription | not modelled (subhalo M* uses the same average-MAH growth-track integral as centrals; no post-infall stripping term) | not modelled (bimodal SFR PDF has no stripping term; any orphan/stripping handling lives in the upstream merger tree, outside this port) | needs plugin |

## Unit and definition conventions

| | STEEL | EMERGE | UniverseMachine |
|---|---|---|---|
| Halo mass definition | `Vir` (Bryan & Norman 98) | `Vir` — src/galaxies.c's own doxygen on `sfe()`: "hmass Virial mass of the dark matter halo in code units" | `Vir` — `mvir` throughout `split_halo_trees_phase2.c` (standard Rockstar/consistent-trees virial mass) |
| `h` convention | `Msun/h` internally | `h_free` — grid values are absolute log10(Msun), no additional `h` factor applied anywhere the coefficients are used | `h_free` — `vmax`/`vmp` are stored and consumed in km/s with no `1/h` factor anywhere in `split_halo_trees_phase2.c` or `sf_model.c`; the `log_vmpeak` grid axis is likewise absolute |
| IMF | Chabrier | Chabrier — inferred, not directly asserted: `data/smf.dat`'s Li & White (2009) calibration data carries a Chabrier IMF correction, and README.md:868-877 documents that convention; O'Leary et al. 2023 never states an IMF anywhere in its text | Chabrier — explicit: README.pdf section 7.1 states "a Chabrier IMF, a Calzetti dust law, and the BC03 SPS model" |
| Primary halo key | mass | mass | **peak velocity** |

The velocity key is the one genuinely unavoidable conversion: see
`steel_plugins::harmonise::mpeak_to_vmax`. The concentration-mass
relation it needs (default Dutton & Maccio 2014) is a modelling choice
that affects UM's results and is recorded per run.

UM's `vMpeak` is a specific, verified quantity, not the historical
maximum of `Vmax`: upstream's `split_halo_trees_phase2.c` tracks two
distinct running-max pointers, one for peak `Mvir` (`mpeak_halo`) and one
for peak `Vmax` (`vpeak_halo`); the catalog field actually consumed as
`vmp` is assigned from the *former* (`ch.vmp = h->mpeak_halo->vmax;`,
line 736), i.e. **`Vmax` evaluated at the epoch of the halo's peak `Mvir`**,
not the peak-over-history of `Vmax` itself. `steel-plugins`' port
(`growth_models/universe_machine.rs::vmpeak_at`) reproduces this exact
definition, evaluated per snapshot: because `GrowthTrack` masses are
monotonically non-decreasing forward in time, a progenitor's peak mass
*so far* is exactly its own contemporary mass at that epoch, so
`stellar_growth_rate` derives vMpeak from each integration step's own
`(log_mh, z)` rather than holding one value fixed across an object's
whole assembly history (an earlier version of this port did the latter;
see the module doc and `docs/VALIDATION.md` §6.5.3 for why that was a
bug, not upstream's actual definition). `UniverseMachineGrowth::log_vmpeak`
remains available as a convenience query for an object's vMpeak *right
now*, at `ctx.own_track`'s root/observed epoch.

All three models are calibrated against the same Planck15 flat LCDM
cosmology (`h = 0.6774`, `Omega_m0 = 0.3089`, `Omega_b0 = 0.0486`,
`Omega_de0 = 0.6911`) per their respective `provenance.toml`
`[cosmology]` blocks and `steel-plugins::cosmology::Planck15`. One
caveat specific to UM: cosmology is not an input to any formula this port
evaluates (`calc_sf_model`/`sfr_at_vmp`/`f_Q` are pure functions of
`log10(vMpeak)` and `z`), and upstream's *own* SAGA MCMC fit was
calibrated against a different N-body box cosmology
(`Om=0.286, Ol=0.714, h0=0.7`, per `scripts/um_fit_var15_fin.cfg`) —
neither Planck15 nor Bolshoi-Planck. This does not affect the values in
this port's fixtures, since cosmology never enters the evaluated
formulas, but it means the Planck15 tag above describes the *comparison*
convention, not the box UM's best-fit parameters were originally derived
against.

No disagreement was found between either provenance.toml's
`[conventions]` block and the corresponding plugin's `DescribedPlugin`
descriptor: EMERGE's `mass_definition`/`h_convention`/`imf`/
`calibrated_cosmology` in `growth_models/emerge.rs` match
`tests/fixtures/emerge/provenance.toml` exactly, and likewise for
UniverseMachine against `tests/fixtures/um_saga/provenance.toml`.

## Known limitations

1. **Subhalo histories are average MAHs, not individual trees.** STEEL is
   statistical by design: a satellite's pre-infall history comes from
   `growth_history(m_infall, z_infall)`, the same average-MAH
   approximation already used for hosts. Both upstreams resolve
   individual subhaloes. This is the principal asymmetry in any
   comparison, and it is STEEL's pre-existing limitation rather than one
   introduced here.
2. **Concentration-mass relation on the UM path.** Any UM result inherits
   a dependence on the chosen c(M,z).
3. **Quenching is not separable in UM.** UM's bimodal SFR PDF contains
   quenching, so UM cannot be combined with STEEL's `QuenchingModel`; the
   validator rejects it. A comparison of STEEL-plus-Wetzel13 against UM is
   therefore comparing two quenching treatments as well as two SFR
   treatments. EMERGE differs: its reionization gate suppresses early
   low-mass growth and is *not* a satellite quenching prescription, so
   EMERGE and Wetzel13 remain compatible. Correspondingly, EMERGE's
   `descriptor()` declares only `Capability::StellarMass` (no
   `Capability::Quenching`, no `Capability::Scatter` — the port is a
   deterministic mean relation with no scatter coefficient anywhere in
   its verified parameter set, so claiming `Capability::Scatter` would be
   false metadata), while UM's `descriptor()` declares
   `Capability::StellarMass`, `Capability::Quenching`,
   `Capability::Scatter`, and `Capability::StarFormationRate`.
4. **Parameter provenance.** Every coefficient must be traceable to a
   paper table or upstream parameter file. Values sourced only from
   automated summaries are not acceptable.

## Recommended follow-up plugins, by comparison impact

1. **`HaloMassFunctionModel` parity** — sets the abundance normalisation;
   the largest single lever on an SMHM overlay.
2. **`SfrModel` parity (v-keyed)** — lets STEEL's own SFR be evaluated on
   UM's velocity key, separating the key from the prescription.
3. **`HaloGrowthModel` parity** — an average-MAH versus tree-resolved
   comparison bounds limitation 1 quantitatively.
4. **`GasMassModel` / `StellarStrippingModel` parity** — smaller effects;
   worth doing only after the above.
