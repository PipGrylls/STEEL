---
name: making-plots
description: Use when creating, editing or regenerating any figure - enforces clarity, honest representation of limits and missing data, and a look-at-it-before-shipping check
---

# Making Plots

A figure is an argument. Every choice in it either supports the argument
honestly or misleads. These rules exist because each was violated in this
repository, and most of the violations survived until someone looked hard.

**Announce at start:** "I'm using the making-plots skill."

## The Iron Rule

```
RENDER, THEN LOOK AT IT, THEN SHIP IT
```

Save the figure, then **Read the image file** and inspect it. Not the code
— the picture. Clipped titles, colliding annotations, invisible series and
illegible ticks are all invisible in source and obvious on sight. If you
have not looked at the rendered image, the plot is not finished.

## Non-negotiables

### 1. Never draw a limit as if it were a measurement

The worst defect this repo has shipped. A stripping-strength figure drew
16 of 30 points at the maximum tested value because the constraint was
never reached. Rendered identically to the 14 real bounds, it read as
"strength up to 4 is allowed here" when it meant "unconstrained; we
stopped looking at 4". Those are opposite claims.

- Censored points, upper limits, lower limits and saturated values get a
  **different marker** (arrows are conventional) and their **own legend
  entry naming what they are**.
- Never let a caller receive a bare number that silently means two
  different things. Return `(value, kind)` or equivalent.
- If a whole region is unconstrained, say so on the figure.

### 2. Distinguish zero from missing from not-measured

`NaN` in one of this repo's sweeps means "the mass was zero", not "data
absent". Plotting it as a gap implies the opposite of the truth. Decide
explicitly for each series which of the three you have, and encode them
differently.

### 3. Label axes with units *and* convention

In this repository a halo mass is meaningless without knowing whether it
is `Msun` or `Msun/h`, and a stellar mass without its IMF. Both have
caused real, shipped errors. Write the convention into the axis label:

```
log10 M_h [Msun/h]        not     log M
log10 M_500c [Msun] (h-free)      not     halo mass
```

If a quantity was converted to make a comparison legitimate, say so in the
label or the caption. A reader must not have to guess which definition an
axis is in.

### 4. Do not clip data

Axis limits must contain every plotted point, or the omission must be
stated on the figure. Silently cropping outliers changes the argument.
If you deliberately zoom, annotate that N points fall outside the view.

Check after rendering: does any line touch or leave an edge?

### 5. Every visible series is in the legend; nothing invisible is

A curve that is structurally zero everywhere is not "plotted at zero" —
it is absent, and a legend entry for it is a lie. Either give it a visible
representation or drop it and explain the omission.

Conversely, an unlabelled series is unreadable. If the legend has grown
past ~6 entries, the figure is probably doing too much — split it.

### 6. Do not encode meaning by colour alone

Vary marker, linestyle or both. Use a perceptually uniform, colourblind-safe
palette (`viridis`, `cividis`) for ordered quantities. Never `jet`.
Reserve red for the thing that is wrong or forbidden, and use it once.

### 7. Show provenance

A figure with no link to the data that made it cannot be checked or
regenerated. Record — on the figure, in an adjacent caption file, or in a
run record — the input dataset, the code version, and any conversion
applied. In this repo, derivations write a `derivation_run` artifact; a
figure produced outside that path is an orphan and should not be trusted.

### 8. Do not imply precision the data lacks

Interpolating a bound across a gap where the underlying curve is concave
biases the answer, and quoting it to four decimals hides that. State the
interpolation, and round to what the sampling supports.

## Before you ship

Run this list against the **rendered image**:

- [ ] Title, labels and legend fully visible, nothing cut off
- [ ] No annotation overlaps data, and no annotation overlaps another
- [ ] Every plotted point inside the axes
- [ ] Limits/censored values visually distinct and named in the legend
- [ ] Axis labels carry units and convention
- [ ] Legible at the size it will be viewed — tick labels not overlapping
- [ ] Colours survive greyscale, or shape carries the distinction too
- [ ] The caveats that apply to the numbers appear somewhere a reader sees

## Matplotlib specifics

```python
import matplotlib
matplotlib.use("Agg")          # before pyplot, for headless work
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.5, 4.8))
...
fig.tight_layout()             # then LOOK - tight_layout still clips
                               # long suptitles; check the render
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)                 # in loops, or you leak figures
```

- `bbox_inches="tight"` fixes most clipping but changes the figure size —
  do not rely on it to rescue a bad layout.
- Prefer `ax.<method>` over `plt.<method>`; the stateful interface silently
  targets the wrong axes in multi-panel figures.
- For a log axis, check for non-positive values first — matplotlib drops
  them silently, which is defect #2 in disguise.

## Red flags

| Thought | Reality |
|---------|---------|
| "The code looks right, no need to open the PNG" | Clipping and collisions exist only in the render. Look at it. |
| "The plateau is obviously the cap" | Obvious to you, who wrote it. A reader sees an allowed value. |
| "I'll label the axis 'mass'" | Which mass, in what units, on whose convention? Ambiguity here has already caused real errors. |
| "The outlier is off-scale, it's fine" | Then say so on the figure, or widen the axis. |
| "I'll drop the caveat, the caption is getting long" | The caveat is the finding. Drop the decoration instead. |
| "Colour distinguishes them fine" | Not in greyscale, not for ~8% of male readers. |
