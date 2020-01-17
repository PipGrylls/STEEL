"""This is the original proof of concept for unevolved SHMF creating the theoritical upper limit"""
Binwidth_SDSS = 0.01
#Range of host halo massed to investigate linear will require weighting later
CentralHaloMass = np.arange(12+ np.log10(h), 15+ np.log10(h), Binwidth_SDSS) #Mvir h-1

#range of satilite masses to investigate starting at 10^11 Mh as we are looking for satilites above 10^10 M*
SatBin = 0.1
SatHaloMass = np.arange(11+ np.log10(h), 15+ np.log10(h), SatBin)

#Makes m/M as required by our Jing et al
m_M = np.array([SatHaloMass - i for i in CentralHaloMass])

#Unevolved SHMF

#Runs the model from Jing
Out = dn_dlnX(Unevolved, np.power(10, m_M))
#Weight the output to the HMF(central)
Out_Weighted = np.array([thing*HMF_fun(CentralHaloMass[i]- np.log10(h)) for i, thing in enumerate(Out)])

#abundance matching
StellarX = SEM.DarkMatterToStellarMass(SatHaloMass- np.log10(h), 0, Paramaters, ScatterOn = False)
#masscuts
StellarX_10 = StellarX[StellarX > SatiliteMassCut]
#intergrates
Integrals = np.array([trapz(thing[StellarX > SatiliteMassCut], StellarX_10) for thing in Out_Weighted])
#f=sum(Bin)/(sum(population)*binwidth)
AnalyticModel = Integrals/(np.sum(Integrals)*Binwidth_SDSS)

plt.plot(CentralHaloMass, AnalyticModel)
plt.savefig("./Figures/AnalyticMax.png")
plt.clf()
"""Proof of concept over"""
