#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:52:57 2019

@author: ssp1e17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import weighted
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

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
goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)
df = df[goodness_cut]

df = df[df.Vmaxwt>0]
df.loc[df.finalflag==5,'BT']=0
df.loc[df.finalflag==1,'BT']=1

df = df[df.Vmaxwt>0]

df = df.dropna()
df = df[df.MsMendSerExp > 0.]
df = df[df.NewLCentSat == 1.]

# Splitting data into relevant morphologies using TTypes and P_S0 values
EarlyGalaxy = df[df.TType <= 0.]
LateGalaxy = df[df.TType > 0.]
EllipticalGalaxy = EarlyGalaxy[EarlyGalaxy.P_S0 < 0.5]
LenticularGalaxy = EarlyGalaxy[EarlyGalaxy.P_S0 >= 0.5]

# Making Halo Mass bins
hm_binwidth = 0.2
hm_bins = np.arange(8,15,hm_binwidth)

# Arrays of zeroes to fill with data
SM_Ell = np.zeros(len(hm_bins))
SM_Len = np.zeros(len(hm_bins))
SM_Late = np.zeros(len(hm_bins))

Number_Ell = np.zeros(len(hm_bins))
Number_Len = np.zeros(len(hm_bins))
Number_Late = np.zeros(len(hm_bins))
Total = np.zeros(len(hm_bins))
Frac_Ell = np.zeros(len(hm_bins))
Frac_Len = np.zeros(len(hm_bins))
Frac_Late = np.zeros(len(hm_bins))

for i in range(len(hm_bins)):
    tmpEll = EllipticalGalaxy.MsMendSerExp[(hm_bins[i] <= EllipticalGalaxy.MhaloL) & (EllipticalGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    wtmpEll = EllipticalGalaxy.Vmaxwt[(hm_bins[i] <= EllipticalGalaxy.MhaloL) & (EllipticalGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    SM_Ell[i] = np.ma.average(tmpEll, weights = wtmpEll)
    if tmpEll.empty:
        SM_Ell[i] = float('NaN')
    else:
        SM_Ell[i] = weighted.median(tmpEll, np.log10(wtmpEll))
    
    tmpLen = LenticularGalaxy.MsMendSerExp[(hm_bins[i] <= LenticularGalaxy.MhaloL) & (LenticularGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    wtmpLen = LenticularGalaxy.Vmaxwt[(hm_bins[i] <= LenticularGalaxy.MhaloL) & (LenticularGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    SM_Len[i] = np.ma.average(tmpLen, weights = wtmpLen)
    if tmpLen.empty:
        SM_Len[i] = float('NaN')
    else:
        SM_Len[i] = weighted.median(tmpLen, np.log10(wtmpLen))
    
    tmpLate = LateGalaxy.MsMendSerExp[(hm_bins[i] <= LateGalaxy.MhaloL) & (LateGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    wtmpLate = LateGalaxy.Vmaxwt[(hm_bins[i] <= LateGalaxy.MhaloL) & (LateGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    SM_Late[i] = np.ma.average(tmpLate, weights = wtmpLate)
    if tmpLate.empty:
        SM_Late[i] = float('NaN')
    else:
        SM_Late[i] = weighted.median(tmpLate, np.log10(wtmpLate))
    
    # Unweighted Number of Galaxies
    numtempEll = EllipticalGalaxy.galcount[(hm_bins[i] <= EllipticalGalaxy.MhaloL) & (EllipticalGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    numtempLen = LenticularGalaxy.galcount[(hm_bins[i] <= LenticularGalaxy.MhaloL) & (LenticularGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    numtempLate = LateGalaxy.galcount[(hm_bins[i] <= LateGalaxy.MhaloL) & (LateGalaxy.MhaloL <= hm_bins[i] + hm_binwidth)]
    
    # Weighted Number of Galaxies
    if numtempEll.empty:
        Num_Ell = float('NaN')
    else:
        Num_Ell = np.sum(wtmpEll)

    if numtempLen.empty:
        Num_Len = float('NaN')
    else:
        Num_Len = np.sum(wtmpLen)

    if numtempLate.empty:
        Num_Late = float('NaN')
    else:
        Num_Late = np.sum(wtmpLate)
    Total = Num_Ell + Num_Len + Num_Late
    
    # Fraction of Each Morphpology
    Frac_Ell[i] = Num_Ell/Total
    Frac_Len[i] = Num_Len/Total
    Frac_Late[i] = Num_Late/Total
        

# Generating Plot with varying line color
# From Matplotlib documentation

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


# First line: Elliptical
x1 = hm_bins[18:-1]
y1 = SM_Ell[18:-1]
fractionEll = Frac_Ell[18:-1] # using these indices to avoid NaN values

points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)

# Second line: Lenticular
x2 = hm_bins[18:-1]
y2 = SM_Len[18:-1]
fractionLen = Frac_Len[18:-1]

points2 = np.array([x2, y2]).T.reshape(-1, 1, 2)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)

# Third line: Spiral
x3 = hm_bins[18:-1]
y3 = SM_Late[18:-1]
fractionLate = Frac_Late[18:-1]

points3 = np.array([x3, y3]).T.reshape(-1, 1, 2)
segments3 = np.concatenate([points3[:-1], points3[1:]], axis=1)


# making an array to get limits of colorbars
lims = np.array([fractionEll.min(), fractionLen.min(), fractionLate.min(), fractionEll.max(), fractionLen.max(), fractionLate.max()])

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
MAP = 'viridis'

# Create a continuous norm to map from data points to colors
norm1 = plt.Normalize(lims.min(), lims.max())
lc1 = LineCollection(segments1, cmap=MAP, norm = norm1, linestyle = '-', label = 'Elliptical')
# Set the values used for colormapping
lc1.set_array(fractionEll)
lc1.set_linewidth(3)
line1 = axs.add_collection(lc1)

norm2 = plt.Normalize(lims.min(), lims.max())
lc2 = LineCollection(segments2, cmap=MAP, norm = norm2, linestyle = '--', label = 'Lenticular')
# Set the values used for colormapping
lc2.set_array(fractionLen)
lc2.set_linewidth(3)
line2 = axs.add_collection(lc2)

norm3 = plt.Normalize(lims.min(), lims.max())
lc3 = LineCollection(segments3, cmap=MAP, norm = norm3, linestyle = ':', label = 'Spiral')
# Set the values used for colormapping
lc3.set_array(fractionLate)
lc3.set_linewidth(3)
line3 = axs.add_collection(lc3)


fig.colorbar(line1, ax=axs)

axs.set_xlim(x1.min()-0.4, x1.max()+0.4)
axs.set_ylim(y1.min()-0.4, y1.max()+0.4)
plt.ylabel("$log_{10}$ $M_*$ [$M_\odot$]")
plt.xlabel("$log_{10}$ $M_h$ [$M_\odot$]")
plt.legend(loc = 'upper left', fontsize = 16, frameon=False)
plt.tight_layout()
plt.savefig('./../Figures/Paper2/PipSideProject.png')
plt.savefig('./../Figures/Paper2/PipSideProject.pdf')
