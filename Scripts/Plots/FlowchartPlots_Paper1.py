import matplotlib as mpl
mpl.use('agg')
import numpy as np
import Functions as F
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
import hmf
cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()
h = Cosmo.h

mpl.rcParams.update({'font.size': 25})
mpl.rcParams.update({'lines.linewidth': 5})
mpl.rcParams.update({'lines.markersize': 5})


z, AvaHaloMass_wz = F.Get_HM_History([10, 11, 12, 13, 14, 15], 10, 15, 1.0)
plt.figure(figsize = (7.5, 5))
for i in range(1, len(AvaHaloMass_wz[0])):
    ColorParam = len(AvaHaloMass_wz[0]) - 1
    if (i == 1) or (i == len(AvaHaloMass_wz[0]) -1):
        plt.plot(np.log10(AvaHaloMass_wz[:,0] +1), AvaHaloMass_wz[:,i] - AvaHaloMass_wz[:,i][0], color = ( (ColorParam-i)/ColorParam, 0, (i-1)/ColorParam ), label = "{}".format(AvaHaloMass_wz[:,i][0]) )
    else:
        plt.plot(np.log10(AvaHaloMass_wz[:,0] +1), AvaHaloMass_wz[:,i] - AvaHaloMass_wz[:,i][0], color = ( (ColorParam-i)/ColorParam, 0, (i-1)/ColorParam ) )
plt.xlabel("$\log_{10}[1+z]$")
plt.ylabel("$\log_{10}[M(z)/M_0]$")
plt.tight_layout()
plt.savefig("./Figures/Fig1B.png")
#plt.savefig("./Figures/HaloGrowth.png")
#plt.show()
plt.clf()

#gets the HMF interpolation function
HMF_fun = F.Make_HMF_Interp() #N Mpc^-3 h^3 dex^-1
plt.figure(figsize = (7.5, 5))
X = np.arange(12, 16.2, 0.2)
Y = np.log10(HMF_fun(X, 0))
plt.ylim(-10, -1)
plt.plot(X, Y)
plt.xlabel("$\log_{10}$ HaloMass [$M_\odot$]")
plt.ylabel("$\log_{10} \phi$ $[Mpc^{-3} dex^{-1}]$")
plt.tight_layout()
plt.savefig("./Figures/Fig1A.png")
#plt.show()
plt.clf()

#Subhalomass function parameters macc/M0
Unevolved = {\
'gamma' : 0.22,\
'alpha' : -0.91,\
'beta' : 6,\
'omega' : 3,\
'a' : 1,\
}

CentMass_A = 13
CentMass_B = 13.5
SatMass = np.arange(10, 13.7, 0.2)
Out_A = (F.dn_dlnX(Unevolved, np.power(10, SatMass - CentMass_A)))
Out_B = (F.dn_dlnX(Unevolved, np.power(10, SatMass - CentMass_B)))
plt.plot(SatMass, np.log10(Out_A))
plt.ylabel("$\log_{10} \phi$ $[Mpc^{-3} dex^{-1}]$")
plt.xlabel("$\log_{10}$ HaloMass [$M_\odot$]")
plt.xlim(10, 14)
plt.ylim(-2.2, 2)
plt.tight_layout()
plt.savefig("./Figures/Fig1C.png")
#plt.show()
plt.clf()

plt.plot(SatMass, np.log10(Out_A), 'b')
plt.plot(SatMass, np.log10(Out_B), 'b')
plt.fill_between(SatMass, np.log10(Out_B), np.log10(Out_A), alpha = 0.5)
plt.ylabel("$\log_{10} \phi$ $[Mpc^{-3} dex^{-1}]$")
plt.xlabel("$\log_{10}$ HaloMass [$M_\odot$]")
plt.xlim(10, 14)
plt.ylim(-2.2, 2)
plt.tight_layout()
plt.savefig("./Figures/Fig1D.png")
#plt.show()
plt.clf()