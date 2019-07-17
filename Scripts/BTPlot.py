import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the catalog
Header=['galcount',
        'finalflag',
        'z',
        'Vmaxwt',
        'MsMendSerExp',
        'AbsMag',
        'logReSerExp',
        'BT',
        'n_bulge',
        'NewLCentSat',
        'NewMCentSat' ,
        'MhaloL',
        'probaE',
        'probaEll',
        'probaS0',
        'probaSab',
        'probaScd',
        'TType',
        'P_S0',
        'veldisp',
        'veldisperr',
        'raSDSS7',
        'decSDSS7']

df = pd.read_csv("/home/ssp1e17/Documents/STEEL/Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat", header = None, names = Header, skiprows = 1, delim_whitespace = True)

# Making necessary cuts to the data to ensure all data is physical
goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)
df = df[goodness_cut]

df = df[df.Vmaxwt>0]
df.loc[df.finalflag==5,'BT']=0
df.loc[df.finalflag==1,'BT']=1

df = df[df.Vmaxwt>0]

df = df.dropna()
df = df[df.MsMendSerExp > 0.]
df = df[df.NewLCentSat == 1.]

# Splitting data into relevant morphologies
EarlyGalaxy = df[df.TType <= 0.]
LateGalaxy = df[df.TType > 0.]
EllipticalGalaxy = EarlyGalaxy[EarlyGalaxy.P_S0 < 0.5]
LenticularGalaxy = EarlyGalaxy[EarlyGalaxy.P_S0 >= 0.5]


sm_binwidth = 0.2
sm_bins = np.arange(9, 12.5, sm_binwidth)


# This section puts the data into relevant mass bins
avg_BT_Ell = np.zeros(len(sm_bins))
avg_BT_Lent = np.zeros(len(sm_bins))
avg_BT_Late = np.zeros(len(sm_bins))
for i in range(len(sm_bins)):
    tmpEll = EllipticalGalaxy.BT[(sm_bins[i] <= EllipticalGalaxy.MsMendSerExp) & (EllipticalGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth)]
    wtmpEll = EllipticalGalaxy.Vmaxwt[(sm_bins[i] <= EllipticalGalaxy.MsMendSerExp) & (EllipticalGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth)]    # weights

    tmpLent = LenticularGalaxy.BT[(sm_bins[i] <= LenticularGalaxy.MsMendSerExp) & (LenticularGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth)]
    wtmpLent = LenticularGalaxy.Vmaxwt[(sm_bins[i] <= LenticularGalaxy.MsMendSerExp) & (LenticularGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth)]   # weights

    tmpLate = LateGalaxy.BT[(sm_bins[i] <= LateGalaxy.MsMendSerExp) & (LateGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth)]
    wtmpLate = LateGalaxy.Vmaxwt[(sm_bins[i] <= LateGalaxy.MsMendSerExp) & (LateGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth)]    # weights

    if tmpEll.empty:
        avg_BT_Ell[i] = float('NaN')    # adds NaN to prevent issues with index lengths
    else:
        avg_BT_Ell[i] = np.ma.average(tmpEll, weights = wtmpEll)    # calculates weighted average

    if tmpLent.empty:
        avg_BT_Lent[i] = float('NaN')
    else:
        avg_BT_Lent[i] = np.ma.average(tmpLent, weights = wtmpLent)
        

    if tmpLate.empty:
        avg_BT_Late[i] = float('NaN')
    else:
        avg_BT_Late[i] = np.ma.average(tmpLate, weights = wtmpLate)


EllipticalBT = avg_BT_Ell
LenticularBT = avg_BT_Lent
SpiralBT = avg_BT_Late


# Plots all the data on a graph
plt.plot(sm_bins, EllipticalBT, label = 'Elliptical', marker = '^', linestyle = '')
plt.plot(sm_bins, LenticularBT, label = 'Lenticular', marker = 'o', linestyle = '')
plt.plot(sm_bins, SpiralBT, label = 'Spiral', marker = 'x', linestyle = '')
plt.legend()
plt.ylabel('Bulge Mass to Total Mass Ratio')
plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
plt.savefig('../Figures/Paper2/BTPlot.png')
