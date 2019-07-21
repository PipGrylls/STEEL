import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import wquantiles

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
df = pd.read_csv('/home/ssp1e17/Documents/STEEL/Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
#
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


sm_binwidth = 0.1
sm_bins = np.arange(9, 12.5, sm_binwidth)


EllipticalWeights = (EllipticalGalaxy.Vmaxwt)
Elliptical_Data, throwaway = np.histogram(EllipticalGalaxy.MsMendSerExp, bins = sm_bins, weights = EllipticalWeights)
Elliptical_Data = np.divide(Elliptical_Data, sm_binwidth*fracsky) #Corrections for skycoverage and binning


SpiralWeights = (LateGalaxy.Vmaxwt)
Spiral_Data, throwaway = np.histogram(LateGalaxy.MsMendSerExp, bins = sm_bins, weights = SpiralWeights)
Spiral_Data = np.divide(Spiral_Data, sm_binwidth*fracsky) #Corrections for skycoverage and binning


LenticularWeights = (LenticularGalaxy.Vmaxwt)
Lenticular_Data, throwaway = np.histogram(LenticularGalaxy.MsMendSerExp, bins = sm_bins, weights = LenticularWeights)
Lenticular_Data = np.divide(Lenticular_Data, sm_binwidth*fracsky) #Corrections for skycoverage and binning


Vmax_Only_weights = (df.Vmaxwt)
Vmax_Only_Data, throwaway = np.histogram(df.MsMendSerExp, bins = sm_bins, weights = Vmax_Only_weights)
Vmax_Only_Data = np.divide(Vmax_Only_Data, sm_binwidth*fracsky)


Avg_HLR_Data = np.zeros(len(sm_bins))
Std_Dev_HLR = np.zeros(len(sm_bins))
for i in range(len(sm_bins)):
    tmp = df.n_bulge[(df.MsMendSerExp < sm_bins[i] + sm_binwidth) & (df.MsMendSerExp >= sm_bins[i] )]
    wtmp = df.Vmaxwt[(df.MsMendSerExp < sm_bins[i] + sm_binwidth) & (df.MsMendSerExp >= sm_bins[i] )]
    Avg_HLR_Data[i] = np.ma.average(tmp, weights = wtmp)
#    Avg_HLR_Data[i] = wquantiles.median(tmp,np.log10(wtmp))
    Std_Dev_HLR[i] =  m.sqrt(np.ma.average((tmp)**2, weights=wtmp))

Avg_HLR_Data = np.divide(Avg_HLR_Data, sm_binwidth*fracsky)

SpiralIndex = np.log10((Spiral_Data/Vmax_Only_Data)*Avg_HLR_Data[:-1])
LenticularIndex = np.log10((Lenticular_Data/Vmax_Only_Data)*Avg_HLR_Data[:-1])
EllipticalIndex = np.log10((Elliptical_Data/Vmax_Only_Data)*Avg_HLR_Data[:-1])


# Plots all the data on a graph
plt.ylabel('Sersic Index')
plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")
plt.plot(sm_bins[5:-1], EllipticalIndex[5:], label = "Elliptical", marker = '^', linestyle = '')
plt.plot(sm_bins[5:-1], LenticularIndex[5:], label = "Lenticular", marker = 'o', linestyle = '')
plt.plot(sm_bins[5:-1], SpiralIndex[5:], label = "Spiral", marker = 'x', linestyle = '')
plt.legend()
plt.savefig('../Figures/Paper2/SersicPlot.png')
