import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import wquantiles
import matplotlib as mpl

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
df = pd.read_csv('/home/ssp1e17/Documents/STEEL/Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)

# Making necessary cuts to the data to ensure all data is physical

df = df[df.Vmaxwt>0]

df = df.dropna()
df = df[df.MsMendSerExp > 0.]
df = df[df.NewLCentSat == 1.]

fracper=len(df)/670722
skycov=8000.
fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

# Splitting data into relevant morphologies
EarlyGalaxy = df[df.TType <= 0.]
LateGalaxy = df[df.TType > 0.]
EllipticalGalaxy = EarlyGalaxy[EarlyGalaxy.P_S0 < 0.5]
LenticularGalaxy = EarlyGalaxy[EarlyGalaxy.P_S0 >= 0.5]


sm_binwidth = 0.2
sm_bins = np.arange(9, 12.5, sm_binwidth)

EllipticalIndex = np.zeros(len(sm_bins))
SpiralIndex = np.zeros(len(sm_bins))
LenticularIndex = np.zeros(len(sm_bins))


for i in range(len(sm_bins)):
    tmpEll = EllipticalGalaxy.n_bulge[(EllipticalGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth) & (EllipticalGalaxy.MsMendSerExp >= sm_bins[i] )]
    wtmpEll = EllipticalGalaxy.Vmaxwt[(EllipticalGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth) & (EllipticalGalaxy.MsMendSerExp >= sm_bins[i] )]

    tmpLen = LenticularGalaxy.n_bulge[(LenticularGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth) & (LenticularGalaxy.MsMendSerExp >= sm_bins[i] )]
    wtmpLen = LenticularGalaxy.Vmaxwt[(LenticularGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth) & (LenticularGalaxy.MsMendSerExp >= sm_bins[i] )]

    tmpLate = LateGalaxy.n_bulge[(LateGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth) & (LateGalaxy.MsMendSerExp >= sm_bins[i] )]
    wtmpLate = LateGalaxy.Vmaxwt[(LateGalaxy.MsMendSerExp < sm_bins[i] + sm_binwidth) & (LateGalaxy.MsMendSerExp >= sm_bins[i] )]

    if tmpEll.empty:
        EllipticalIndex[i] = float('NaN')    # adds NaN to prevent issues with index lengths
    else:
#        EllipticalIndex[i] = np.ma.average(tmpEll, weights = wtmpEll)    # calculates weighted average
        EllipticalIndex[i] = wquantiles.median(tmpEll, weights = wtmpEll)
#        
    if tmpLen.empty:
        LenticularIndex[i] = float('NaN')
    else:
#        LenticularIndex[i] = np.ma.average(tmpLen, weights = wtmpLen)
        LenticularIndex[i] = wquantiles.median(tmpLen, weights = wtmpLen)
    
    if tmpLate.empty:
        SpiralIndex[i] = float('NaN')
    else:
#        SpiralIndex[i] = np.ma.average(tmpLate, weights = wtmpLate)
        SpiralIndex[i] = wquantiles.median(tmpLate, weights = wtmpLate)


# Plots all the data on a graph

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['axes.linewidth']=2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['font.size']=22
plt.rcParams['lines.linewidth']=3

plt.ylabel('Sersic Index')
plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")


plt.plot(sm_bins, EllipticalIndex, label = 'Elliptical', marker = '^', linestyle = '')
plt.plot(sm_bins, LenticularIndex, label = 'Lenticular', marker = 'o', linestyle = '')
plt.plot(sm_bins, SpiralIndex, label = 'Spiral', marker = 'x', linestyle = '')

plt.legend(frameon = False, fontsize = 16)
plt.tight_layout()
plt.savefig('../Figures/Paper2/SersicPlot.png')
plt.savefig('../Figures/Paper2/SersicPlot.pdf')