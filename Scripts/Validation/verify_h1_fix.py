"""One-off verification for PORT-FIX H1 (Return_PF_Plot's satellite
upper-mass-cut bug). Not part of the test suite -- runs against a real
RunParam tree and reports how much the fixed pair fraction differs from
the buggy one, plus the raw bin indices so the difference is legible.

Usage: run from repo root with env/py-legacy active.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from Scripts import CentralPostprocessing as CP

RunParam = ('1.0', True, True, True, 'G19_DPL', 'G19_SE')
d = CP.PairFractionData(RunParam)

Parent_Cut = 11
Mass_Ratio = np.log10(1 / 4)
Upper_Cut = Parent_Cut + 0.6

buggy_bins = []
fixed_bins = []
n_differ = 0
n_total = 0

for i, SM_Arr in enumerate(d.AvaStellarMass):
    CND_Mass = d.Get_CND_Masses(d.SMF_interp, M=Parent_Cut, z=d.z[i])
    try:
        M_Cut_bin = np.digitize(CND_Mass, SM_Arr)
    except Exception:
        continue
    CND_Mass_Upper = d.Get_CND_Masses(d.SMF_interp, M=Upper_Cut, z=d.z[i])
    M_Cut_bin_upper = np.digitize(CND_Mass_Upper, SM_Arr)

    for j, _ in enumerate(d.AvaHaloMass[i, M_Cut_bin:M_Cut_bin_upper]):
        buggy = np.digitize(CND_Mass_Upper, SM_Arr)
        fixed = np.digitize(SM_Arr[M_Cut_bin + j], d.Surviving_Sat_SMF_MassRange)
        lower = np.digitize(SM_Arr[M_Cut_bin + j] + Mass_Ratio, d.Surviving_Sat_SMF_MassRange)
        buggy_bins.append(buggy)
        fixed_bins.append(fixed)
        n_total += 1
        if buggy != fixed:
            n_differ += 1
        if i == 0 and j < 5:
            print(f"i={i} j={j}: lower_bin={lower} buggy_upper_bin={buggy} "
                  f"fixed_upper_bin={fixed} (SurvivingSatSMF has "
                  f"{len(d.Surviving_Sat_SMF_MassRange)} bins, SM_Arr has {len(SM_Arr)})")

print(f"\n{n_differ}/{n_total} (i,j) cells have a different satellite-upper-bin "
      f"index under the fix.")
print(f"buggy bin range: [{min(buggy_bins)}, {max(buggy_bins)}]")
print(f"fixed bin range: [{min(fixed_bins)}, {max(fixed_bins)}]")

# Now the actual output: PairFracTot with both formulas.
z_old, pf_old, _, _ = d.Return_PF_Plot(d.SMF_interp, Parent_Cut=Parent_Cut, UpperLimit=True)
print("\nz (last 10):", np.array(z_old[-10:]))
print("PairFracTot with the fix (last 10):", np.array(pf_old[-10:]))

z_noupper, pf_noupper, _, _ = d.Return_PF_Plot(d.SMF_interp, Parent_Cut=Parent_Cut, UpperLimit=False)
print("PairFracTot with UpperLimit=False (last 10):", np.array(pf_noupper[-10:]))
