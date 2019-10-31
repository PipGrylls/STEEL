import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wquantiles
import matplotlib as mpl

# Loading the catalog
#Header = ["galcount",
#          "z",
#          "Vmaxwt",
#          "MsMendSerExp",
#          "AbsMag",
#          "logReSerExp",
#          "BT",
#          "n_bulge",
#          "NewLCentSat",
#          "NewMCentSat",
#          "MhaloL",
#          "probaE",
#          "probaEll",
#          "probaS0",
#          "probaSab",
#          "probaScd",
#          "TType",
#          "AbsMagCent",
#          "MsCent",
#          "veldisp",
#          "veldisperr",
#          "AbsModel_newKcorr",
#          "LCentSat",
#          "raSDSS7",
#          "decSDSS7",
#          "Z",
#          "sSFR",
#          "FLAGsSFR",
#          "MEDIANsSFR",
#          "P16sSFR",
#          "P84sSRF", 
#          "SFR",
#          "FLAGSFR",
#          "MEDIANSFR",
#          "P16SFR",
#          "P84SRF",
#          "RA_SDSS",
#          "DEC_SDSS",
#          "Z_2",
#          "Seperation"]
#df = pd.read_csv("/home/ssp1e17/Documents/STEEL/Data/Observational/Bernardi_SDSS/new_catalog_SFRs.dat", header = None, names = Header, skiprows = 1, delim_whitespace = True)

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
df = pd.read_csv('./Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)

#Header = ["galcount",
#          "zMeert",
#          "Vmaxwt",
#          "MsMendSerExp",
#          "AbsMag",
#          "logReSerExp",
#          "BT",
#          "n_bulge",
#          "m_bulge",
#          "r_bulge",
#          "NewLCentSat",
#          "NewMCentSat",
#          "newMhaloL",
#          "probaE",
#          "probaEll",
#          "probaS0",
#          "probaSab",
#          "probaScd",
#          "TType",
#          "P_S0",
#          "AbsMagCent",
#          "MsCent",
#          "veldisp",
#          "veldisperr",
#          "AbsModel_newKcorr",
#          "LCentSat",
#          "raSDSS7",
#          "decSDSS7"]
#df = pd.read_csv("/home/ssp1e17/Documents/STEELinternship/Jay's Work/Notebooks - DO NOT DELETE OR EDIT/Catalogs/new_catalog_morph_Jay.dat", header = None, names = Header, skiprows = 1, delim_whitespace = True)

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


sm_binwidth = 0.1
sm_bins = np.arange(9, 12.0, sm_binwidth)


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
#        avg_BT_Ell[i] = wquantiles.median(tmpEll,np.log10(wtmpEll))

    if tmpLent.empty:
        avg_BT_Lent[i] = float('NaN')
    else:
        avg_BT_Lent[i] = np.ma.average(tmpLent, weights = wtmpLent)
#        avg_BT_Lent[i] = wquantiles.median(tmpLent,np.log10(wtmpLent))
        

    if tmpLate.empty:
        avg_BT_Late[i] = float('NaN')
    else:
        avg_BT_Late[i] = np.ma.average(tmpLate, weights = wtmpLate)
#        avg_BT_Late[i] = wquantiles.median(tmpLate,np.log10(wtmpLate))


EllipticalBT = avg_BT_Ell
LenticularBT = avg_BT_Lent
SpiralBT = avg_BT_Late


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

plt.plot(sm_bins, EllipticalBT, label = 'Elliptical', marker = '^', linestyle = '')
plt.plot(sm_bins, LenticularBT, label = 'Lenticular', marker = 'o', linestyle = '')
plt.plot(sm_bins, SpiralBT, label = 'Spiral', marker = 'x', linestyle = '')
plt.legend(frameon = False, fontsize = 16)
plt.ylabel('B/T')
plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
plt.tight_layout()
plt.savefig('../Figures/Paper2/BTPlot.png')
plt.savefig('../Figures/Paper2/BTPlot.pdf')
