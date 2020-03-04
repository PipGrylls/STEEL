import os
import sys
import h5py
AbsPath = str(__file__)[:-len("/CentralPostprocessing.py")]
sys.path.append(AbsPath+"/..")
import multiprocessing
import pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
from numba import jit
from matplotlib.gridspec import GridSpec
# mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
from Plots import SDSS_Plots
from Functions import Functions as F
from Functions import Functions_c as F_c
from scipy import interpolate
from scipy.integrate import cumtrapz
from itertools import cycle
from copy import copy
from colossus.cosmology import cosmology
cosmology.setCosmology("planck15")
Cosmo =cosmology.getCurrent()
HMF_fun = F.Make_HMF_Interp() #N Mpc^-3 h^3 dex^-1, Args are (Mass, Redshift)
h = Cosmo.h
h_3 = h*h*h

if "SDSS.pkl" in os.listdir("./Scripts/CentralPostprocessing"):
    Add_SDSS = pickle.load(open("./Scripts/CentralPostprocessing/SDSS.pkl", 'rb'))
else:
    Add_SDSS = SDSS_Plots.SDSS_Plots(11.5,15,0.1) #pass this halomass:min, max, and binwidth for amting the SDSS plots
    pickle.dump(Add_SDSS, open("./Scripts/CentralPostprocessing/SDSS.pkl", 'wb'))

#set plot paramaters here
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

plt.rcParams['font.size']=15


# Use LaTeX for rendering
#mpl.rcParams["text.usetex"] = True
# load the xfrac package
#mpl.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')

colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"] #matplotlibdeafults and black


#Abundance Matching Parameters
Override =\
{\
'M10':11.95,\
'SHMnorm10':0.03,\
'beta10':1.6,\
'gamma10':0.7,\
'M11':0.5,\
'SHMnorm11':-0.01,\
'beta11':-0.6,\
'gamma11':0.1\
}

AbnMtch =\
{\
'Behroozi13': False,\
'Behroozi18': False,\
'B18c':False,\
'B18t':False,\
'G18':False,\
'G18_notSE':False,\
'G19_SE':False,\
'G19_cMod':False,\
'Lorenzo18':False,\
'Moster': False,\
'Moster10': False,\
'Illustris': False,\
'z_Evo':True,\
'Scatter': 0.11,\
'Override_0': False,\
'Override_z': False,\
'Override': Override,\
'PFT': False,\
'M_PFT1': False,\
'M_PFT2': False,\
'M_PFT3': False,\
'N_PFT1': False,\
'N_PFT2': False,\
'N_PFT3': False,\
'b_PFT1': False,\
'b_PFT2': False,\
'b_PFT3': False,\
'g_PFT1': False,\
'g_PFT2': False,\
'g_PFT3': False,\
'g_PFT4': False,\
'HMevo': False,\
'HMevo_param': None\
}

Paramaters = \
{\
'AbnMtch' : AbnMtch,\
'AltDynamicalTime': 1,\
'NormRnd': 0.5,\
'SFR_Model': 'CE'\
}



#Functions Required for making plots

@jit#('double[:,:],double[:,:],double[:,:](double[:,:,:],double[:,:,:], double[:],double[:])')
def JitLoop(SHMF_Entering, Mass_Ratio_Bins, z_step, t_step, Bin):
    m, n, o = np.shape(SHMF_Entering)
    Accreted_Above_Ratio = np.zeros((m, n))
    Accreted_Above_Ratio_dz = np.zeros((m, n))
    Accreted_Above_Ratio_dt = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            Accreted_Above_Ratio[i,j] = np.sum(SHMF_Entering[i, j,Mass_Ratio_Bins[i,j]:])*Bin
            Accreted_Above_Ratio_dz[i,j] = np.sum(SHMF_Entering[i, j,Mass_Ratio_Bins[i,j]:])*Bin/z_step[i]
            Accreted_Above_Ratio_dt[i,j] = np.sum(SHMF_Entering[i, j,Mass_Ratio_Bins[i,j]:])*Bin/t_step[i]
    return Accreted_Above_Ratio, Accreted_Above_Ratio_dz, Accreted_Above_Ratio_dt

@jit#('double[:,:],double[:,:],double[:,:](double[:,:,:],double[:,:,:],double[:],double[:], double[:])')
def JitLoop2(SHMF_Entering, Mass_Ratio_Bins, SatHaloMass, z_step, t_step, Bin):
    m, n, o = np.shape(SHMF_Entering)
    Accreted_Above_Ratio = np.zeros((m, n))
    Accreted_Above_Ratio_dz = np.zeros((m, n))
    Accreted_Above_Ratio_dt = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            Accreted_Above_Ratio[i,j] = np.sum(SHMF_Entering[i,j,Mass_Ratio_Bins[i,j]:]*SatHaloMass[Mass_Ratio_Bins[i,j]:])*Bin
            Accreted_Above_Ratio_dz[i,j] = np.sum(SHMF_Entering[i,j,Mass_Ratio_Bins[i,j]:]*SatHaloMass[Mass_Ratio_Bins[i,j]:])*Bin/z_step[i]
            Accreted_Above_Ratio_dt[i,j] = np.sum(SHMF_Entering[i,j,Mass_Ratio_Bins[i,j]:]*SatHaloMass[Mass_Ratio_Bins[i,j]:])*Bin/t_step[i]
    return Accreted_Above_Ratio, Accreted_Above_Ratio_dz, Accreted_Above_Ratio_dt


class PairFractionData:
    def __init__(self, Fit_in):
        self.Fit = Fit_in[5]
        print("Tuple = ", Fit_in)
        self.RunParam = Fit_in
        self.Data_AC = F.LoadData_Mergers(Fit_in)
        self.Data_PF = F.LoadData_Pair_Frac(Fit_in)
        
        self.Accretion_History, self.z, self.AvaHaloMass, self.Surviving_Sat_SMF_MassRange = self.Data_AC       
        self.Pair_Frac, self.z, self.AvaHaloMass, self.Surviving_Sat_SMF_MassRange = self.Data_PF
        #Calculate the theoritical central SM using SMHM
        self.AvaStellarMass, self.AvaStellarMassBins = self.CreateAverageSM()
        #Calculate SM_Bin for satellites
        self.SM_Bin = self.Surviving_Sat_SMF_MassRange[1] - self.Surviving_Sat_SMF_MassRange[0]
        #Account for central bin shrinking
        self.AvaHaloMassBins = self.AvaHaloMass[:,1:] - self.AvaHaloMass[:,:-1] 
        self.AvaHaloMassBins = np.concatenate((self.AvaHaloMassBins, np.array([self.AvaHaloMassBins[:,-1]]).T), axis = 1)
        self.SMF_interp = self.Generate_SMF_interp()        
        self.z_step = self.z[1:] - self.z[:-1]
        self.t_step = Cosmo.age(self.z[:-1]) - Cosmo.age(self.z[1:])

    def CreateAverageSM(self):
        AbnMtch[self.Fit] = True
        if "PFT" in self.Fit:
            AbnMtch["PFT"] = True
        if "HMevo" in self.Fit:
            AbnMtch["HMevo"] = True
            AbnMtch["HMevo_param"] = float(self.Fit[-3:])

        AvaStellarMass = []
        for i, HM_Arr in enumerate(self.AvaHaloMass):
            AvaStellarMass.append(F.DarkMatterToStellarMass(HM_Arr-np.log10(h), self.z[i], Paramaters))
        AvaStellarMass = np.array(AvaStellarMass)

        #Where I have decreased the binsize the SMHM relation is occasionaly not monotomically increasing 
        #This smooths it out for np.digitize
    
        AvaStellarMass2 = copy(AvaStellarMass)
        for i, SM_Arr in enumerate(AvaStellarMass):
            try:
                #Check is cut is possible
                Test_Cut = 11.0
                M_Cut_bin = np.digitize(Test_Cut, SM_Arr)
            except:
                #If not smooth data
                for j in range(0, len(SM_Arr)-1):
                    if SM_Arr[j+1] <= SM_Arr[j]:
                        if j+2 == len(SM_Arr):
                            AvaStellarMass2[i, j+1] = 2*SM_Arr[j] - SM_Arr[j-1]
                        else:
                            AvaStellarMass2[i, j+1] = (SM_Arr[j] + SM_Arr[j+2])/2
                if AvaStellarMass2[i, -1] < AvaStellarMass2[i, -1]:
                    AvaStellarMass2[i, -1] = AvaStellarMass2[i, -1] + (AvaStellarMass2[i, -2] - AvaStellarMass2[i, -3])
        AvaStellarMass = AvaStellarMass2


        AvaStellarMassBins = AvaStellarMass[:,1:] - AvaStellarMass[:,:-1] 
        AvaStellarMassBins = np.concatenate((AvaStellarMassBins, np.array([AvaStellarMassBins[:,-1]]).T), axis = 1)

        AbnMtch[self.Fit] = False
        if "PFT" in self.Fit:
            AbnMtch["PFT"] = False
        
        return AvaStellarMass, AvaStellarMassBins

    def Return_Morph_Plot(self, MassRatio = 0.3, z_start = 10):
        FirstAddition = True
        P_ellip = np.zeros_like(self.AvaStellarMass)
        MMR = np.log10(MassRatio) #mergermass ratio in log10
        print(np.shape(self.AvaStellarMass)[0], np.shape(self.AvaStellarMass)[1], np.shape(self.z))
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                Maj_Merge_Bin = np.digitize(self.AvaStellarMass[i,j]+MMR, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_Frac = np.sum(self.Accretion_History[i,j,Maj_Merge_Bin:])*self.SM_Bin #sums the numberdensity of satellites causing major mergers
                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac #if this is the first step then the number turned is just the fraction
                elif (z_start > self.z[i]):
                    P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j]) #otherwise correct for the prexisting elliptical population
            if (z_start > self.z[i]):
                FirstAddition = False
        # print(self.AvaStellarMass)
        return P_ellip

    def Return_Sai_Idea_Plot(self, MassRatio = 0.3, z_start = 10, GasFracThresh = 0.15):
        FirstAddition = True

        GasFrac = np.zeros_like(self.AvaStellarMass)
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                alpha = 0.59 * ((1+self.z[i])**0.45)
                GasFrac[i,j] = 0.04*(10**self.AvaStellarMass[i,j]/4.5e11)**(-1*alpha)


        P_ellip = np.zeros_like(self.AvaStellarMass)
        P_lentic = np.zeros_like(self.AvaStellarMass)
        MMR = np.log10(MassRatio) #mergermass ratio in log10
        
        GasFracThresh = 0.06

        print(np.shape(self.AvaStellarMass)[0], np.shape(self.AvaStellarMass)[1], np.shape(self.z))
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                Maj_Merge_Bin = np.digitize(self.AvaStellarMass[i,j]+MMR, bins = self.Surviving_Sat_SMF_MassRange) 
                Major_Frac = np.sum(self.Accretion_History[i,j,Maj_Merge_Bin:])*self.SM_Bin #sums the numberdensity of satellites causing major mergers

                CurrentGasFrac = GasFrac[i,j]

                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac #if this is the first step then the number turned is just the fraction
                    if CurrentGasFrac >= GasFracThresh:
                        P_lentic[i,j] = 1 - P_ellip[i,j]
                elif (z_start > self.z[i]):
                    P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j]) #otherwise correct for the prexisting elliptical population
                    if CurrentGasFrac >= GasFracThresh:
                        P_lentic[i,j] = 1 - P_ellip[i,j]
                    else:
                        P_lentic[i,j] = 0
            if (z_start > self.z[i]):
                FirstAddition = False
        # print(self.AvaStellarMass)

        return P_lentic



    def Return_NoMerger_Plot(self, MassRatio = 0.3, z_start = 10, z_cut = 2):
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
        
        # Getting masses for galaxies
        yval = []
        for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
            testval = []
            for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
                testval.append(self.AvaStellarMass[i,j])
            yval.append(testval)

        # Creating list of total masses and current fractions of total mass
        totals = []
        fractions = []
        for i in yval:
            totals.append(i[-1])
        for i in range(len(yval)):
            fractions.append(np.array(yval[i])/totals[i])
        for i in range(len(yval)):
            # plotting fraction of total mass against redshift
            plt.plot(np.log10(self.z + 1)[::-1], np.log10(fractions[i]), label = str(i))
        
        plt.xlabel("$\log_{10}[1+z]$")
        plt.ylabel("$\log_{10}[M_{*}(z)/M_0]$")
        plt.tight_layout()
        plt.savefig('./Figures/Paper2/GalaxyGrowth.png')
        plt.savefig('./Figures/Paper2/GalaxyGrowth.pdf')
        plt.clf()

        # Defining cutoff parameters
        z_cut = 1.5
        fraction_cutoff = 0.7

        # Finding indexes of galaxies that meet the cutoff parameters
        index_of_lenticular = []
        for i in range(len(fractions)):
            for j in range(len(fractions[i])):
                if fractions[i][j] >= fraction_cutoff and self.z[j] <= z_cut:
                    index_of_lenticular.append([i,j])
        
        # making a list of the masses of all the found lenticulars
        masses_of_lenticulars =[]
        for i in index_of_lenticular:
            masses_of_lenticulars.append(yval[i[0]][i[1]])

        # making a list of the masses of all the galaxies
        allmasses = []
        for i in yval:
            for j in i:
                allmasses.append(j)

        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.5, sm_binwidth)

        # binning the data
        inds_all = np.digitize(allmasses, sm_bins)
        inds_len = np.digitize(masses_of_lenticulars, sm_bins)

        # making a dictionary where the keys are indexes of each bin, and value is number of galaxies in that bin
        unique, counts = np.unique(inds_all, return_counts = True)
        dic_all = dict(zip(unique, counts))
        unique, counts = np.unique(inds_len, return_counts = True)
        dic_len = dict(zip(unique, counts))

        # creating a list of mass fractions from the dictionary
        fracs = []
        for i in dic_all.keys():
            try:
                fracs.append(dic_len[i]/dic_all[i])
            except:
                fracs.append(0)

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


        # Pulling SDSS data to add to a plot
        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        # Making necessary cuts to the dataframe
        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Lenticulars Only
        SM_Len = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])
        Vmax_Len = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])

        Weights_Len = Vmax_Len
        Weightsum_Len = np.sum(Vmax_Len)
        totVmax_Len = Weightsum_Len/fracsky

        hist_cent_Len, edges = np.histogram(SM_Len, bins = sm_bins, weights = Vmax_Len)

        Y_Len = np.log10(np.divide(hist_cent_Len, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Len = np.power(10, Y_Len - Y_All)

        plt.plot(sm_bins[1:], F_Len, "k^", label = "SDSS", fillstyle = "none", markersize=15) # SDSS plot
        plt.plot(sm_bins, fracs[1:], "-k",label = "STEEL, z = 0.1") # Model Plot
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.legend(frameon = False)
        plt.tight_layout()
        plt.savefig('./Figures/Paper2/NoMergerLenticular.png')
        plt.savefig('./Figures/Paper2/NoMergerLenticular.pdf')
        plt.clf()

    def CookModel(self, MassRatio = 0.3, z_start = 10, z_cut = 2):

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
        
        # Getting masses for galaxies
        yval = []
        for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
            testval = []
            for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
                testval.append(self.AvaStellarMass[i,j])
            yval.append(testval)

        # Creating list of total masses and current fractions of total mass
        totals = []
        fractions = []
        for i in yval:
            totals.append(i[-1])
        for i in range(len(yval)):
            fractions.append(np.array(yval[i])/totals[i])
        for i in range(len(yval)):
            # plotting fraction of total mass against redshift
            plt.plot(np.log10(self.z + 1)[::-1], np.log10(fractions[i]), label = str(i))
        
        plt.xlabel("$\log_{10}[1+z]$")
        plt.ylabel("$\log_{10}[M_{*}(z)/M_0]$")
        plt.tight_layout()
        plt.savefig('./Figures/Paper2/GalaxyGrowthCook.png')
        plt.savefig('./Figures/Paper2/GalaxyGrowthCook.pdf')
        plt.clf()


        # Defining cutoff parameters
        Elliptical_Z_Cutoff = 2.5
        Lenticular_Z_Cutoff = 0
        Mass_Frac_Threshold = 0.7

        # Finding indexes of galaxies that meet the cutoff parameters
        index_of_elliptical = []
        for i in range(len(fractions)):
            for j in range(len(fractions[i])):
                if fractions[i][j] >= Mass_Frac_Threshold and self.z[j] >= Elliptical_Z_Cutoff:
                    index_of_elliptical.append([i,j])
        
        index_of_lenticular = []
        for i in range(len(fractions)):
            for j in range(len(fractions[i])):
                if fractions[i][j] >= Mass_Frac_Threshold and Lenticular_Z_Cutoff <= self.z[j] < Elliptical_Z_Cutoff:
                    index_of_lenticular.append([i,j])

        index_of_spiral = []
        for i in range(len(fractions)):
            for j in range(len(fractions[i])):
                if fractions[i][j] >= Mass_Frac_Threshold and self.z[j] <= Lenticular_Z_Cutoff:
                    index_of_spiral.append([i,j])

        # making a list of the masses of each galaxy type
        masses_of_ellipticals =[]
        for i in index_of_elliptical:
            masses_of_ellipticals.append(yval[i[0]][i[1]])

        masses_of_lenticulars =[]
        for i in index_of_lenticular:
            masses_of_lenticulars.append(yval[i[0]][i[1]])

        masses_of_spirals =[]
        for i in index_of_spiral:
            masses_of_spirals.append(yval[i[0]][i[1]])

        # making a list of the masses of all the galaxies
        allmasses = masses_of_lenticulars + masses_of_spirals + masses_of_ellipticals

        print(len(masses_of_lenticulars), len(masses_of_spirals), len(masses_of_ellipticals))
        print(len(masses_of_lenticulars)+ len(masses_of_spirals)+ len(masses_of_ellipticals))
        print(len(allmasses))

        sm_binwidth = 0.1
        sm_bins = np.arange(9, 12.5, sm_binwidth)

        # binning the data
        inds_all = np.digitize(allmasses, sm_bins)
        inds_ell = np.digitize(masses_of_ellipticals, sm_bins)
        inds_len = np.digitize(masses_of_lenticulars, sm_bins)
        inds_spi = np.digitize(masses_of_spirals, sm_bins)

        # making a dictionary where the keys are indexes of each bin, and value is number of galaxies in that bin
        unique, counts = np.unique(inds_all, return_counts = True)
        dic_all = dict(zip(unique, counts))

        unique, counts = np.unique(inds_ell, return_counts = True)
        dic_ell = dict(zip(unique, counts))

        unique, counts = np.unique(inds_len, return_counts = True)
        dic_len = dict(zip(unique, counts))

        unique, counts = np.unique(inds_spi, return_counts = True)
        dic_spi = dict(zip(unique, counts))

        print('*******************************************************************')
        print(dic_all)
        print('*******************************************************************')
        print(dic_ell)
        print('*******************************************************************')
        print(dic_len)
        print('*******************************************************************')
        print(dic_spi)
        print('*******************************************************************')
        # creating a list of mass fractions from the dictionary
        fracs_lent = []
        for i in dic_all.keys():
            try:
                fracs_lent.append(dic_len[i]/dic_all[i])
            except:
                fracs_lent.append(0)
        # plt.plot(sm_bins, fracs_lent[1:], "-k",label = "Lenticular") # Model Plot

        fracs_ell = []
        for i in dic_all.keys():
            try:
                fracs_ell.append(dic_ell[i]/dic_all[i])
            except:
                fracs_ell.append(0)
        plt.plot(sm_bins, fracs_ell[1:], "-r",label = "Elliptical") # Model Plot
    
        fracs_spir = []
        for i in dic_all.keys():
            try:
                fracs_spir.append(dic_spi[i]/dic_all[i])
            except:
                fracs_spir.append(0)
        # plt.plot(sm_bins, fracs_spir[1:], "-b",label = "Spiral") # Model Plot

        print('*******************************************************************')
        # print(fracs_ell)
        # print(fracs_lent)
        # print(fracs_spir)

        print(np.array(fracs_ell) + np.array(fracs_lent) + np.array(fracs_spir))
        print('*******************************************************************')


        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.5, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Ellipticals Only
        SM_Ell = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])
        Vmax_Ell = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])

        Weights_Ell = Vmax_Ell
        Weightsum_Ell = np.sum(Vmax_Ell)
        totVmax_Ell = Weightsum_Ell/fracsky

        hist_cent_Ell, edges = np.histogram(SM_Ell, bins = sm_bins, weights = Vmax_Ell)

        Y_Ell = np.log10(np.divide(hist_cent_Ell, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Ell = np.power(10, Y_Ell - Y_All)
        plt.plot(sm_bins[1:], F_Ell, "k^", label = "SDSS", fillstyle = "none", markersize=15)
        
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.legend(frameon = False)
        plt.tight_layout()
        plt.savefig('./Figures/Paper2/CookModel.png')
        plt.savefig('./Figures/Paper2/CookModel.pdf')

        plt.clf()

        
    def Return_Second_Order_Lenticular_Plot(self, MassRatio = 0.3, MassRatioS0 = 0.1, z_start = 10):
        FirstAddition = True
        FirstAdditionS0 = True

        P_ellip = np.zeros_like(self.AvaStellarMass)
        P_lentic = np.zeros_like(self.AvaStellarMass)
        
        MMR = np.log10(MassRatio) #mergermass ratio in log10
        MMRS0 = np.log10(MassRatioS0)
        
        print(np.shape(self.AvaStellarMass)[0], np.shape(self.AvaStellarMass)[1], np.shape(self.z), np.shape(P_lentic))
        
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                Maj_Merge_Bin = np.digitize(self.AvaStellarMass[i,j]+MMR, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_Frac = np.sum(self.Accretion_History[i,j,Maj_Merge_Bin:])*self.SM_Bin #sums the numberdensity of satellites causing major mergers
                
                Maj_Merge_BinS0 = np.digitize(self.AvaStellarMass[i,j]+MMRS0, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_FracS0 = np.sum(self.Accretion_History[i,j,Maj_Merge_BinS0:])*self.SM_Bin
                
                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac #if this is the first step then the number turned is just the fraction
                    P_lentic[i,j] = Major_FracS0
                elif (z_start > self.z[i]):
                    P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j]) #elliptical population continues to grow
                    P_lentic[i,j] = P_lentic[i+1,j] + Major_FracS0*(1 - P_lentic[i+1,j] - P_ellip[i+1,j]) #lenticular population grows including existing E population

            if (z_start > self.z[i]):
                FirstAddition = False
                
        return P_lentic

    
    def Return_Gas_Hard_Threshold_Plot(self, MassRatio = 0.3, MassRatioS0 = 0.1, z_start = 10, GasFracThresh = 0.5):
        FirstAddition = True
        FirstAdditionS0 = True

        GasFrac = np.zeros_like(self.AvaStellarMass)
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                alpha = 0.59 * ((1+self.z[i])**0.45)
                GasFrac[i,j] = 0.04*(10**self.AvaStellarMass[i,j]/4.5e11)**(-1*alpha)

        P_ellip = np.zeros_like(self.AvaStellarMass)
        P_lentic = np.zeros_like(self.AvaStellarMass)
        
        MMR = np.log10(MassRatio) #mergermass ratio in log10
        MMRS0 = np.log10(MassRatioS0)
        
        print(np.shape(self.AvaStellarMass)[0], np.shape(self.AvaStellarMass)[1], np.shape(self.z), np.shape(P_lentic))
        
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                Maj_Merge_Bin = np.digitize(self.AvaStellarMass[i,j]+MMR, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_Frac = np.sum(self.Accretion_History[i,j,Maj_Merge_Bin:])*self.SM_Bin #sums the numberdensity of satellites causing major mergers
                
                Maj_Merge_BinS0 = np.digitize(self.AvaStellarMass[i,j]+MMRS0, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_FracS0 = np.sum(self.Accretion_History[i,j,Maj_Merge_BinS0:])*self.SM_Bin

                CurrentGasFrac = GasFrac[i,j]

                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac
                    if CurrentGasFrac >= GasFracThresh:
                        P_lentic[i,j] = Major_FracS0

                elif (z_start > self.z[i]):
                    if CurrentGasFrac >= GasFracThresh:
                        P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j])
                        P_lentic[i,j] = P_lentic[i+1,j] + Major_FracS0*(1 - P_lentic[i+1,j] - P_ellip[i+1,j])
                    else:
                        P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j] - P_lentic[i+1,j])
            if (z_start > self.z[i]):
                FirstAddition = False
        return P_lentic


    def Return_Gas_Soft_Threshold_Plot(self, MassRatio = 0.3, MassRatioS0 = 0.1, z_start = 10, GasFracThresh = 0.5):
        FirstAddition = True
        FirstAdditionS0 = True

        GasFrac = np.zeros_like(self.AvaStellarMass)
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                alpha = 0.59 * ((1+self.z[i])**0.45)
                GasFrac[i,j] = 0.04*(10**self.AvaStellarMass[i,j]/4.5e11)**(-1*alpha)

        P_ellip = np.zeros_like(self.AvaStellarMass)
        P_lentic = np.zeros_like(self.AvaStellarMass)
        
        MMR = np.log10(MassRatio) #mergermass ratio in log10
        MMRS0 = np.log10(MassRatioS0)
        
        print(np.shape(self.AvaStellarMass)[0], np.shape(self.AvaStellarMass)[1], np.shape(self.z), np.shape(P_lentic))
        
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                Maj_Merge_Bin = np.digitize(self.AvaStellarMass[i,j]+MMR, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_Frac = np.sum(self.Accretion_History[i,j,Maj_Merge_Bin:])*self.SM_Bin #sums the numberdensity of satellites causing major mergers
                
                Maj_Merge_BinS0 = np.digitize(self.AvaStellarMass[i,j]+MMRS0, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_FracS0 = np.sum(self.Accretion_History[i,j,Maj_Merge_BinS0:])*self.SM_Bin

                CurrentGasFrac = GasFrac[i,j]

                divisor = 3.8
                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac
                    if CurrentGasFrac >= GasFracThresh:
                        P_lentic[i,j] = Major_FracS0
                    else:
                        P_lentic[i,j] = Major_FracS0 - abs(CurrentGasFrac - GasFracThresh)/divisor #arbitrary number

                elif (z_start > self.z[i]):
                    if CurrentGasFrac >= GasFracThresh:
                        P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j])
                        P_lentic[i,j] = P_lentic[i+1,j] + Major_FracS0*(1 - P_lentic[i+1,j] - P_ellip[i+1,j])
                    else:
                        P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j] - P_lentic[i+1,j])
                        P_lentic[i,j] = P_lentic[i+1,j] + Major_FracS0*(1 - P_lentic[i+1,j] - P_ellip[i+1,j]) - abs(CurrentGasFrac - GasFracThresh)/divisor #arbitrary number
                        # P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j])
            if (z_start > self.z[i]):
                FirstAddition = False
        return P_lentic


    def Return_Baryonic_Inflow_Rate(self, discmass, redshift):
        return 25 * discmass * ((1 + redshift)/3)**1.5


    def Return_New_Gas_Inflow_Plot(self, MassRatio = 0.25, MassRatioS0 = 0.05, z_start = 10, GasFracThresh = 0.5):
        print('Beginning Gas Inflow Plot function')
        FirstAddition = True
        FirstAdditionS0 = True
        GasFracThresh = 0.2
        GasFrac = np.zeros_like(self.AvaStellarMass)
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                alpha = 0.59 * ((1+self.z[i])**0.45)
                GasFrac[i,j] = 0.04*(10**self.AvaStellarMass[i,j]/4.5e11)**(-1*alpha)


        BulgeMass = np.zeros_like(self.AvaStellarMass)
        for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
            BulgeMass[np.shape(self.AvaStellarMass)[0]-1,j] = self.Return_Baryonic_Inflow_Rate(10**(self.AvaStellarMass[np.shape(self.AvaStellarMass)[0]-1,j]-11), self.z[np.shape(self.AvaStellarMass)[0]-1])* self.t_step[np.shape(self.AvaStellarMass)[0]-2]*(10**9)
        for i in range(np.shape(self.AvaStellarMass)[0]-2, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                BulgeMass[i,j] = BulgeMass[i+1, j] + self.Return_Baryonic_Inflow_Rate(10**(self.AvaStellarMass[i,j]-11), self.z[i]) * self.t_step[i]*(10**9)
        BulgeMass = np.log10(BulgeMass)

        
        print("Printing bulge Masses")
        print(BulgeMass[0])
        print("Printing galaxy Masses")
        print(self.AvaStellarMass[0])
        
        BulgeRatios = np.power(10,BulgeMass-self.AvaStellarMass)

        print('Printing bulge ratio')
        print(BulgeRatios[0])

        P_ellip = np.zeros_like(self.AvaStellarMass)
        P_lentic = np.zeros_like(self.AvaStellarMass)
        
        MMR = np.log10(MassRatio) #mergermass ratio in log10
        MMRS0 = np.log10(MassRatioS0)
        
        for i in range(np.shape(self.AvaStellarMass)[0]-1, -1, -1):
            for j in range(np.shape(self.AvaStellarMass)[1]-1, -1, -1):
                Maj_Merge_Bin = np.digitize(self.AvaStellarMass[i,j]+MMR, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_Frac = np.sum(self.Accretion_History[i,j,Maj_Merge_Bin:])*self.SM_Bin #sums the numberdensity of satellites causing major mergers
                
                Maj_Merge_BinS0 = np.digitize(self.AvaStellarMass[i,j]+MMRS0, bins = self.Surviving_Sat_SMF_MassRange) #find the bin of the Surviving_Sat_SMF_MassRange above which is major mergers
                Major_FracS0 = np.sum(self.Accretion_History[i,j,Maj_Merge_BinS0:])*self.SM_Bin

                divisor = 1.0
                cutoff = 0.127
                
                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac
                    P_lentic[i,j] = Major_FracS0
                    if GasFrac[i,j] >= GasFracThresh:
                        P_lentic[i,j] -= abs(GasFrac[i,j] - GasFracThresh)/divisor
                    FirstAddition = False
                elif z_start > self.z[i]:
                    if BulgeRatios[i,j] > cutoff:
                        add_inflow = (BulgeRatios[i,j]/100) *\
                        (1 - P_lentic[i+1,j] - P_ellip[i+1,j])
                    else: 
                        add_inflow = 0
                    if GasFrac[i,j] >= GasFracThresh:
                        add_merge = Major_FracS0*\
                        (1 - P_lentic[i+1,j] - P_ellip[i+1,j])
                        ellp_add = Major_Frac *\
                        (1 - P_ellip[i+1,j])
                    else:
                        add_merge = Major_FracS0 *\
                        (1 - P_lentic[i+1,j] - P_ellip[i+1,j]) -\
                        abs(GasFrac[i,j] - GasFracThresh)/divisor
                        if add_merge < 0:
                            add_merge = 0
                        ellp_add = Major_Frac *\
                        (1 - P_ellip[i+1,j])
                    addition = add_merge + add_inflow

                    P_ellip[i,j] = P_ellip[i+1,j] +\
                    ellp_add
                    P_lentic[i,j] = P_lentic[i+1,j] +\
                    addition
        return P_lentic

def Fit_to_Str(Fit):
    Str_Out = ""
    for i in Fit:
        Str_Out += str(i)+"_"
    return Str_Out

    
def MakeClass(Fit):
    Class = PairFractionData(Fit)
    FitName = Fit_to_Str(Fit)
    pickle.dump(Class, open("./Scripts/CentralPostprocessing/"+FitName+".pkl", 'wb'))
    return [Fit, Class]   


if __name__ == "__main__":
    
    #Make the classes===================================================================================
    M_Factors = [('1.0', True, False, True, 'G19_DPL', 'G19_SE'),\
                 ('1.0', True, False, True, 'G19_DPL', 'M_PFT1'),\
                 ('1.0', True, False, True, 'G19_DPL', 'M_PFT2'),\
                 ('1.0', True, False, True, 'G19_DPL', 'M_PFT3')]
    N_Factors = [('1.0', True, False, True, 'G19_DPL', 'G19_SE'),\
                 ('1.0', True, False, True, 'G19_DPL', 'N_PFT1'),\
                 ('1.0', True, False, True, 'G19_DPL', 'N_PFT2'),\
                 ('1.0', True, False, True, 'G19_DPL', 'N_PFT3')]
    b_Factors = [('1.0', True, False, True, 'G19_DPL', 'G19_SE'),\
                 ('1.0', True, False, True, 'G19_DPL', 'b_PFT1'),\
                 ('1.0', True, False, True, 'G19_DPL', 'b_PFT2'),\
                 ('1.0', True, False, True, 'G19_DPL', 'b_PFT3')]
    g_Factors = [('1.0', True, False, True, 'G19_DPL', 'G19_SE'),\
                 ('1.0', True, False, True, 'G19_DPL', 'g_PFT1'),\
                 ('1.0', True, False, True, 'G19_DPL', 'g_PFT2'),\
                 ('1.0', True, False, True, 'G19_DPL', 'g_PFT3')]
    
    cMod_Factors = [('1.0', True, False, True, 'G19_DPL', 'G19_cMod')]#,\
                    #('1.0', False, True, True, 'CE_PP', 'G19_cMod'),\
                    #('1.0', False, False, True, 'CE', 'G19_cMod'),\
                    #('1.0', True, True, True, 'CE_PP', 'G19_cMod')]
    Evo_Factors = [('1.0', False, False, True, 'CE', 'G19_SE'),\
                   ('1.0', False, True, True, 'CE', 'G19_SE'),\
                   ('1.0', True, True, True, 'CE', 'G19_SE')]
    DPL_Factors = [('1.0', False, False, True, 'G19_DPL', 'G19_SE'),\
                   ('1.0', False, True, True, 'G19_DPL', 'G19_SE'),\
                   ('1.0', True, True, True, 'G19_DPL', 'G19_SE'),\
                   ('1.0', True, True, True, 'G19_DPL_PP', 'G19_SE'),\
                   ('0.8', True, True, True, 'G19_DPL', 'G19_SE'),\
                   ('0.8', True, True, True, 'G19_DPL_PP', 'G19_SE'),\
                   ('1.2', True, True, True, 'G19_DPL', 'G19_SE'),\
                   ('1.2', True, True, True, 'G19_DPL_PP', 'G19_SE')]
    Ill_Factors = [('1.0', True, False, True, 'Illustris', 'Illustris')]
    HMevo_Factors = [('1.0', False, False, True, 'G19_DPL', 'G19_cMod'),\
                     ('1.0', False, False, True, 'G19_DPL', 'HMevo_alt_0.0'),\
                     ('1.0', False, False, True, 'G19_DPL', 'HMevo_alt_0.1'),\
                     ('1.0', False, False, True, 'G19_DPL', 'HMevo_alt_0.2'),\
                     ('1.0', False, False, True, 'G19_DPL', 'HMevo_alt_0.3'),\
                     ('1.0', False, False, True, 'G19_DPL', 'HMevo_alt_0.4'),\
                     ('1.0', False, False, True, 'G19_DPL', 'HMevo_alt_0.5')
                    ]
    Total_Factors = Evo_Factors + DPL_Factors + cMod_Factors + M_Factors + N_Factors + b_Factors + g_Factors + Ill_Factors + HMevo_Factors

    if False:
        ClassList = []
        SucessfulData = os.listdir("./Scripts/CentralPostprocessing/")
        for Fit in Total_Factors:
            FitName = Fit_to_Str(Fit)
            if FitName+".pkl" in SucessfulData:
                ClassList.append([Fit, pickle.load(open("./Scripts/CentralPostprocessing/"+FitName+".pkl", 'rb'))])
            else:
                try:
                    pickle.dump(PairFractionData(Fit), open("./Scripts/CentralPostprocessing/"+FitName+".pkl", 'wb'))
                except Exception as e:
                    print(Fit + "excepted with:", e)
    else:
        ClassList = []
        FitToRun = []
        SucessfulData = os.listdir("./Scripts/CentralPostprocessing/")
        for Fit in Total_Factors:
            FitName = Fit_to_Str(Fit)
            if FitName+".pkl" in SucessfulData:
                ClassList.append([Fit, pickle.load(open("./Scripts/CentralPostprocessing/"+FitName+".pkl", 'rb'))])
            else:
                if FitName not in FitToRun:
                    FitToRun.append(Fit)
        if len(FitToRun) > 0:
            print(FitToRun)
            pool = multiprocessing.Pool(processes = len(FitToRun))
            ClassList_New = pool.map(MakeClass, FitToRun)
            pool.close()
            ClassList += ClassList_New

    FitList = []
    Classes = []
    for i in ClassList:
        FitList.append(i[0])
        Classes.append(i[1])
    print(FitList)
    #=====================================================================================

    #Morphology Plot    
    if True:
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
        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.5, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Ellipticals Only
        SM_Ell = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])
        Vmax_Ell = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])

        Weights_Ell = Vmax_Ell
        Weightsum_Ell = np.sum(Vmax_Ell)
        totVmax_Ell = Weightsum_Ell/fracsky

        hist_cent_Ell, edges = np.histogram(SM_Ell, bins = sm_bins, weights = Vmax_Ell)

        Y_Ell = np.log10(np.divide(hist_cent_Ell, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Ell = np.power(10, Y_Ell - Y_All)
        plt.plot(sm_bins[1:], F_Ell, "k^", label = "SDSS", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{elliptical}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        
        MassRatio = 0.5
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        # P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 10)
        for MassRatio in [0.25, 0.375, 0.50, 0.625, 0.75, 0.875]:
            P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 2)
            plt.plot(Classes[index].AvaStellarMass[0], P_ellip[0], label = 'MassRatio = ' + "{}".format(MassRatio))

        #Create data for lorenzo
        if False:
            for i, P_Ellip in enumerate(Classes[index].Return_Morph_Plot(MassRatio, z_start = 2)):
                Output = np.vstack((Classes[index].AvaHaloMass[i], P_Ellip)).T
                FilePath = "./Test/Lorenzo/2/Halo_PEllip_{}.dat".format(Classes[index].z[i])
                np.savetxt(FilePath, Output)
            for i, P_Ellip in enumerate(Classes[index].Return_Morph_Plot(MassRatio, z_start = 3)):
                Output = np.vstack((Classes[index].AvaHaloMass[i], P_Ellip)).T
                FilePath = "./Test/Lorenzo/3/Halo_PEllip_{}.dat".format(Classes[index].z[i])
                np.savetxt(FilePath, Output)
        
        
        # z_plot = 1.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_ellip[np.digitize(z_plot, bins = Classes[index].z)], "--C0", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))
        # z_plot = 2.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_ellip[np.digitize(z_plot, bins = Classes[index].z)], "-.C3", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))
        plt.xlim(10, 12.3)
        # plt.text(10.2, 0.4, r"$\frac{M_{*, sat}}{M_{*,cen}} >$" + "{}".format(MassRatio))
        # plt.legend(frameon = False)
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig("Figures/Paper2/GalaxyMorphologies.png")
        plt.savefig("Figures/Paper2/GalaxyMorphologies.pdf")
        plt.clf()
    

    #Second Order Lenticular Morphology Plot
    '''
    This generates plots working under the assumptions that:
    1) Lenticulars are anything that isn't an elliptical
    2) Lenticulars have mass ratio range between MassRatio and MassRatioS0

    Anything else is a spiral/late-type galaxy
    '''
    if True:
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
        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.5, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Lenticulars Only
        SM_Len = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])
        Vmax_Len = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])

        Weights_Len = Vmax_Len
        Weightsum_Len = np.sum(Vmax_Len)
        totVmax_Len = Weightsum_Len/fracsky

        hist_cent_Len, edges = np.histogram(SM_Len, bins = sm_bins, weights = Vmax_Len)

        Y_Len = np.log10(np.divide(hist_cent_Len, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Len = np.power(10, Y_Len - Y_All)
        plt.plot(sm_bins[1:], F_Len, "k^", label = "SDSS", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        
        MassRatio = 0.25
        MassRatioS0 = 0.05
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_Second_Order_Lenticular_Plot(MassRatio, MassRatioS0, 2)

        
        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "STEEL, z = 0.1")
        # z_plot = 1.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_lentic[np.digitize(z_plot, bins = Classes[index].z)], "--C0", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))
        # z_plot = 2.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_lentic[np.digitize(z_plot, bins = Classes[index].z)], "-.C3", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))

        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        plt.text(10.2, 0.55, r"$\frac{M_{*, sat}}{M_{*,cen}} >$" + "{}".format(MassRatioS0))
        plt.legend(frameon = False)
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Second_Order_Lenticular.png")
        plt.savefig("Figures/Paper2/Second_Order_Lenticular.pdf")
        plt.clf()



    # Gas Fraction Restricted Lenticular Plots
    '''
    This generates plots working under the same assumptions as the second order plot.
    We also introduce the condition that to form Lenticulars, there must be a gas fraction higher than a threshold.
    Two sets of plots are generated, one with a hard threshold and one with a soft threshold:
        The hard threshold has no lenticulars forming if below gas threshold
        The soft threshold has less lenticulars forming the further below the gas threshold.
    '''
    if True:
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
        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.0, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Lenticulars Only
        SM_Len = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])
        Vmax_Len = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])

        Weights_Len = Vmax_Len
        Weightsum_Len = np.sum(Vmax_Len)
        totVmax_Len = Weightsum_Len/fracsky

        hist_cent_Len, edges = np.histogram(SM_Len, bins = sm_bins, weights = Vmax_Len)

        Y_Len = np.log10(np.divide(hist_cent_Len, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Len = np.power(10, Y_Len - Y_All)
        plt.plot(sm_bins[1:], F_Len, "kx", label = "SDSS", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))


        MassRatio = 0.25
        MassRatioS0 = 0.050
        # GasFracThresh = 0.152
        GasFracThresh = 0.107
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_Gas_Hard_Threshold_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)
        
        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "STEEL, z = 0.1")#, Lenitculars")
        plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.legend(frameon = False)
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Gas_Fraction_Hard_Threshold.png")
        plt.savefig("Figures/Paper2/Gas_Fraction_Hard_Threshold.pdf")
        plt.clf()


        #Lenticulars Only
        plt.plot(sm_bins[1:], F_Len, "k^", fillstyle = "none", markersize=15, label = 'SDSS Data')
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        
        #Ellipticals Only
        SM_Ell = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])
        Vmax_Ell = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])

        Weights_Ell = Vmax_Ell
        Weightsum_Ell = np.sum(Vmax_Ell)
        totVmax_Ell = Weightsum_Ell/fracsky

        hist_cent_Ell, edges = np.histogram(SM_Ell, bins = sm_bins, weights = Vmax_Ell)

        Y_Ell = np.log10(np.divide(hist_cent_Ell, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Ell = np.power(10, Y_Ell - Y_All)
        # plt.plot(sm_bins[1:], F_Ell, "r^", fillstyle = "none", markersize=15)

        #Spirals Only
        F_Spir = 1 - F_Len - F_Ell
        # plt.plot(sm_bins[1:], F_Spir, "b^", fillstyle = "none", markersize=15)

        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_Gas_Soft_Threshold_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)
        P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 2)
        P_spiral = 1 - P_lentic - P_ellip

        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "Lenticulars")
        # plt.plot(Classes[index].AvaStellarMass[0], P_ellip[0], "-r", label = "Ellipticals")
        # plt.plot(Classes[index].AvaStellarMass[0], P_spiral[0], "-b", label = "Spirals")

        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        # plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.legend(frameon = False, fontsize='x-small')
        # plt.text(10.8, 0.55, r"GFT = " + "{}".format(GasFracThresh), fontsize = 'x-small')
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Gas_Fraction_Soft_Threshold.png")
        plt.savefig("Figures/Paper2/Gas_Fraction_Soft_Threshold.pdf")
        plt.clf()
        
        
    # Final Model with Bulge Growth and Gas Fractions
    """
    This is the final model I worked on during my internship. It assumes:
    1) Lenticulars are formed by minor mergers with ratio 0.05<ratio<0.25
    2) Lenticulars must be formed from gas rich galaxies, otherwise they are less likely
    3) Galaxies have inflowing Baryonic mass that causes a bulge to grow
    4) Galxies with BT ratio >0.15 are also considered lenticulars
    """
    if True:
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
        plt.rcParams['font.size']=15
        plt.rcParams['lines.linewidth']=3
        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.0, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Lenticulars Only
        SM_Len = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])
        Vmax_Len = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])

        Weights_Len = Vmax_Len
        Weightsum_Len = np.sum(Vmax_Len)
        totVmax_Len = Weightsum_Len/fracsky

        hist_cent_Len, edges = np.histogram(SM_Len, bins = sm_bins, weights = Vmax_Len)

        Y_Len = np.log10(np.divide(hist_cent_Len, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Len = np.power(10, Y_Len - Y_All)
        plt.plot(sm_bins[1:], F_Len, "kx", label = "SDSS", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))

        #Ellipticals Only
        SM_Ell = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])
        Vmax_Ell = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])

        Weights_Ell = Vmax_Ell
        Weightsum_Ell = np.sum(Vmax_Ell)
        totVmax_Ell = Weightsum_Ell/fracsky

        hist_cent_Ell, edges = np.histogram(SM_Ell, bins = sm_bins, weights = Vmax_Ell)

        Y_Ell = np.log10(np.divide(hist_cent_Ell, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Ell = np.power(10, Y_Ell - Y_All)
        plt.plot(sm_bins[1:], F_Ell, "r^", fillstyle = "none", markersize=15)

        #Spirals Only
        F_Spir = 1 - F_Len - F_Ell
        plt.plot(sm_bins[1:], F_Spir, "b^", fillstyle = "none", markersize=15)

        MassRatio = 0.25
        MassRatioS0 = 0.050
        GasFracThresh = 0.127
        #index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_New_Gas_Inflow_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)
        P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 2)
        P_spiral = 1 - P_lentic - P_ellip

        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "Lenticulars")
        plt.plot(Classes[index].AvaStellarMass[0], P_ellip[0], "-r", label = "Ellipticals")
        plt.plot(Classes[index].AvaStellarMass[0], P_spiral[0], "-b", label = "Spirals")

        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        # plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.legend(frameon = False, fontsize='x-small')
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.text(10.1, 0.8, r"GFT = " + "{}".format(GasFracThresh))
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Bulge_Growth_Final.png")
        plt.savefig("Figures/Paper2/Bulge_Growth_Final.pdf")
        plt.clf()

    # Preprogrammed Lenticular Growth Plot
    '''
    This generates two plots, and models lenticular growth without any mergers.
    We posit that lenticulars form by growing to a certain fraction of final mass in a certain redshift interval
    Returns 1st plot of galaxy growth
    Returns 2nd plot of fraction of lenticulars with SDSS data also plotted
    This is currently broken as we need to use a time interval instead of redshift interval
    Use colossus function to convert redshift to time and use that as the cutoff instead of redshift
    '''
    if False:
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        Classes[index].Return_NoMerger_Plot(MassRatio, 1, 0.1)


    # Cook et al 2008 Model
    '''
    This generates a plot based loosely on the Cook et al model from 2008.
    We posit that each galaxy morphology type is preprogrammed by growth in a particular epoch
    Thus, we say that galaxies that grow to X% of their final mass in epoch 1 are ellipticals,
    those that grow to X% of final mass in epoch 2 are lenticulars, and those that grow to X%
    of final mass in epoch 3 are spirals.
    '''
    if False:
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        Classes[index].CookModel(MassRatio, 1, 0.1)

    # Sai's Model Idea
    '''
    This is similar to the gas threshold models but it starts from the first order assumption,
    that lenticulars are formed through mergers between dark matter halos with mass ratio <0.25
    We then apply the gas fraction threshold to this model
    '''
    if False:
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
        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.5, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Lenticulars Only
        SM_Len = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])
        Vmax_Len = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])

        Weights_Len = Vmax_Len
        Weightsum_Len = np.sum(Vmax_Len)
        totVmax_Len = Weightsum_Len/fracsky

        hist_cent_Len, edges = np.histogram(SM_Len, bins = sm_bins, weights = Vmax_Len)

        Y_Len = np.log10(np.divide(hist_cent_Len, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Len = np.power(10, Y_Len - Y_All)
        plt.plot(sm_bins[1:], F_Len, "k^", label = "SDSS", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))

        MassRatio = 0.25
        GasFracThresh = 0.0
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        # P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 10)
        P_lentic = Classes[index].Return_Sai_Idea_Plot(MassRatio, 2, GasFracThresh)

        
        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "STEEL, z = 0.1")

        plt.xlim(10, 12.3)
        plt.text(10.2, 0.7, r"$\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        plt.legend(frameon = False)
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig("Figures/Paper2/SaiIdea.png")
        plt.savefig("Figures/Paper2/SaiIdea.pdf")
        plt.clf()


    # Final Model with Halo Mass on x-axis
    """
    Uses the same model as the 'final model' with both gas fractions and bulge growth, but generates
    a plot that uses Halo mass on x axis.
    """
    if False:
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

        Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                  'BT','n_bulge','NewLCentSat','NewMCentSat'
                                  ,'MhaloL','probaE','probaEll',
                                'probaS0','probaSab','probaScd','TType','P_S0',
                              'veldisp','veldisperr','raSDSS7','decSDSS7']

        df = pd.read_csv('Data/Observational/Bernardi_SDSS/new_catalog_morph_flag_rtrunc.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
        goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

        df = df[goodness_cut]

        df = df[df.Vmaxwt>0]
        df.loc[df.finalflag==5,'BT']=0
        df.loc[df.finalflag==1,'BT']=1

        fracper=len(df)/670722
        skycov=8000.
        fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)

        df_cent = df[df.NewLCentSat == 1.0]
        #Add SDSS Data to plot
        sm_binwidth = 0.2
        sm_bins = np.arange(9, 12.0, sm_binwidth)

        #Total Population
        SM_All = np.array(df_cent.MsMendSerExp)
        Vmax_All = np.array(df_cent.Vmaxwt)

        Weights_All = Vmax_All
        Weightsum_All = np.sum(Vmax_All)
        totVmax_All = Weightsum_All/fracsky

        hist_cent_All, edges_All = np.histogram(SM_All, bins = sm_bins, weights = Vmax_All)

        Y_All = np.log10(np.divide(hist_cent_All, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        #Lenticulars Only
        SM_Len = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])
        Vmax_Len = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0>=0.5)])

        Weights_Len = Vmax_Len
        Weightsum_Len = np.sum(Vmax_Len)
        totVmax_Len = Weightsum_Len/fracsky

        hist_cent_Len, edges = np.histogram(SM_Len, bins = sm_bins, weights = Vmax_Len)

        Y_Len = np.log10(np.divide(hist_cent_Len, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Len = np.power(10, Y_Len - Y_All)

        plt.subplot(121)

        # plt.plot(sm_bins[1:], F_Len, "k^", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{Morphology}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))

        #Ellipticals Only
        SM_Ell = np.array(df_cent.MsMendSerExp[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])
        Vmax_Ell = np.array(df_cent.Vmaxwt[(df_cent.TType<=0)&(df_cent.P_S0<0.5)])

        Weights_Ell = Vmax_Ell
        Weightsum_Ell = np.sum(Vmax_Ell)
        totVmax_Ell = Weightsum_Ell/fracsky

        hist_cent_Ell, edges = np.histogram(SM_Ell, bins = sm_bins, weights = Vmax_Ell)

        Y_Ell = np.log10(np.divide(hist_cent_Ell, fracsky*sm_binwidth)*0.9195) #0.9195 correction of volume to Planck15

        F_Ell = np.power(10, Y_Ell - Y_All)
        # plt.plot(sm_bins[1:], F_Ell, "r^", fillstyle = "none", markersize=15)

        #Spirals Only
        F_Spir = 1 - F_Len - F_Ell
        # plt.plot(sm_bins[1:], F_Spir, "b^", fillstyle = "none", markersize=15)

        MassRatio = 0.25
        MassRatioS0 = 0.050
        GasFracThresh = 0.107
        GasFracThresh = 0.127
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_New_Gas_Inflow_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)
        P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 2)
        P_spiral = 1 - P_lentic - P_ellip

        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "Lenticulars")
        plt.plot(Classes[index].AvaStellarMass[0], P_ellip[0], "-r", label = "Ellipticals")
        plt.plot(Classes[index].AvaStellarMass[0], P_spiral[0], "-b", label = "Spirals")

        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        # plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.legend(frameon = False, fontsize='x-small')
        plt.xlim(10,12)
        plt.ylim(0,1)
        # plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.tight_layout()
        # plt.savefig("Figures/Paper2/Bulge_Growth_Final.png")
        # plt.savefig("Figures/Paper2/Bulge_Growth_Final.pdf")
        # plt.clf()

        MassRatio = 0.25
        MassRatioS0 = 0.050
        GasFracThresh = 0.107
        GasFracThresh = 0.127
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_New_Gas_Inflow_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)
        P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 2)
        P_spiral = 1 - P_lentic - P_ellip

        plt.subplot(122)
        plt.plot(Classes[index].AvaHaloMass[0], P_lentic[0], "-k",label = "Lenticulars")
        plt.plot(Classes[index].AvaHaloMass[0], P_ellip[0], "-r", label = "Ellipticals")
        plt.plot(Classes[index].AvaHaloMass[0], P_spiral[0], "-b", label = "Spirals")

        plt.xlabel("$log_{10}$ $M_{halo}$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{Morphology}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))

        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        # plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        # plt.legend(frameon = False, fontsize='x-small')
        plt.xlim(12,15)
        plt.ylim(0,1)
        # plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Bulge_Growth_Final_Halo.png")
        plt.savefig("Figures/Paper2/Bulge_Growth_Final_Halo.pdf")
        plt.clf()
