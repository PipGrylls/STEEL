import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
import Functions as F
from itertools import cycle
from colossus.cosmology import cosmology
cosmology.setCosmology("planck15")
Cosmo =cosmology.getCurrent()
HMF_fun = F.Make_HMF_Interp() #N Mpc^-3 h^3 dex^-1, Args are (Mass, Redshift)
h = Cosmo.h
h_3 = h*h*h
#set plot paramaters here
mpl.rcParams.update(mpl.rcParamsDefault)


#Plots the number of subahlos with sahding representing when they were accreted
def SubhalosByAccretionTime(RunParam = (1.0, False, False, True, True, 'G18')):

    z, SubHaloMass, NumberDensities = F.LoadData_MultiEpoch_SubHalos(RunParam)
    #print("z shape:",np.shape(z))
    #print("SubHaloMass shape:",np.shape(SubHaloMass))
    #print("NumberDensities shape:",np.shape(NumberDensities))

    NumbersAtZero = NumberDensities[0]

    plot = np.zeros_like(NumbersAtZero[0])
    previous = plot
    for i, NumberDen in enumerate(np.flip(NumbersAtZero, 0)):
        plot = plot + NumberDen
        if (i+1)%19 == 0:
            plt.bar(SubHaloMass, plot, 0.1, bottom = previous, label = "z = {} to {}".format(np.flip(z,0)[i], np.flip(z,0)[i-18]))
            previous = plot + previous
            plot = np.zeros_like(NumbersAtZero[0])
    plt.plot(SubHaloMass, HMF_fun(SubHaloMass,0), "k", label = "Centrals")
    plt.yscale("log")
    plt.ylim(10**-6, 10**0)
    plt.xlim(10, 15)
    plt.legend(loc = 1, bbox_to_anchor=(1.3, 1), frameon = False)
    plt.xlabel("$log{10}$ $M_{h}$ $[M_{\odot}]$")
    plt.ylabel("$N$ [$Mpc^{-3}$ $h^{3}$ $dex^{-1}$]")
    plt.savefig("./ExtraPlots/SubhalosByAccretionTime.png")
#==============================================================================
    
    
    
if __name__ == "__main__":
    SubhalosByAccretionTime()