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
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
from Scripts.Plots import SDSS_Plots
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


#PairFractions Systematics Plot======================================
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

        
    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def Return_Cent_SMF(self, z_in, SMF_X = np.arange(8, 12.5, 0.01), SMF_X_Bin = 0.01, N = 10):
        AbnMtch[self.Fit] = True
        if "PFT" in self.Fit:
            AbnMtch["PFT"] = True
        HM_Bin = 0.01
        HM_Range = np.arange(9, 15, HM_Bin)
        cSMF_X, cSMF_Y = F.DM_to_SM(SMF_X, np.log10(HMF_fun(HM_Range, z_in)), HM_Range, HM_Bin, SMF_X_Bin, Paramaters, Redshift = z_in, N = 3000)    
        AbnMtch[self.Fit] = False
        if "PFT" in self.Fit:
            AbnMtch["PFT"] = False
        return self.running_mean(cSMF_X, N), self.running_mean(cSMF_Y, N), SMF_X_Bin*N

    def Generate_SMF_interp(self):
        X, SMFs, SMF_bw = self.Return_Cent_SMF(0)
        SMFs = pd.Series(SMFs).replace([np.inf, -np.inf], np.nan).interpolate().get_values().tolist()
        for i in self.z:
            SMFs = np.vstack((SMFs, pd.Series(self.Return_Cent_SMF(i)[1]).replace([np.inf, -np.inf], np.nan).interpolate().get_values().tolist()))
        SMF_interp = interpolate.interp2d(X, np.insert(self.z, 0, 0), SMFs)
        return SMF_interp
    
    def Get_CND_Masses(self, Master_interp, M = 11, z = 0.1):
        MassRange = np.arange(7, 15, 0.1)
        Master_int = Master_interp(MassRange, z)
        ThisClass_int = self.SMF_interp(MassRange, z)
        SMF_interp_Master_inv = interpolate.interp1d(Master_int, MassRange)
        SMF_interp_ThisClass_inv = interpolate.interp1d(ThisClass_int, MassRange)

        ND = Master_interp(M, z)[0]
        try:
            return SMF_interp_ThisClass_inv(ND)
        except:
            print("Error: NumberDensity", ND, "outside range for", self.Fit)
            print(M, z)

    def Return_PF_Plot(self, Master_interp, Parent_Cut = 11, Mass_Ratio = np.log10(1/4), UpperLimit = True):
        Upper_Cut = Parent_Cut + 0.6
        PairFracTot = []
        M_L, M_U = None, None
        for i, SM_Arr in enumerate(self.AvaStellarMass):
            CND_Mass = self.Get_CND_Masses(Master_interp, M = Parent_Cut, z = self.z[i])
            try:
                M_Cut_bin = np.digitize(CND_Mass, SM_Arr)
            except:
                PairFracTot.append(np.nan)
                continue
            CND_Mass_Upper = self.Get_CND_Masses(Master_interp, M = Upper_Cut, z = self.z[i])
            if UpperLimit:
                M_Cut_bin_upper = np.digitize(CND_Mass_Upper, SM_Arr)
            else:
                M_Cut_bin_upper = -1
        
            if self.Fit[-1] in ["2","3"]:
                Bin = np.digitize(2, bins = self.z)
                if i == Bin:
                    M_L = CND_Mass; M_U = CND_Mass_Upper
            if self.Fit[-1] in ["1","E","d","s"]:                
                if i == 0:
                    M_L = CND_Mass; M_U = CND_Mass_Upper
            
            Total_Pair = 0
            for j, Cent_Mass in enumerate(self.AvaHaloMass[i, M_Cut_bin:M_Cut_bin_upper]):
                Sat_Mass_Cut_bin = np.digitize(SM_Arr[M_Cut_bin + j]+Mass_Ratio, self.Surviving_Sat_SMF_MassRange)
                if UpperLimit:
                    Sat_Mass_Cut_bin_upper = np.digitize(CND_Mass_Upper, SM_Arr)
                else:
                    Sat_Mass_Cut_bin_upper = -1
                Total_Pair += np.sum(self.Pair_Frac[i, M_Cut_bin+j, Sat_Mass_Cut_bin:Sat_Mass_Cut_bin_upper])*self.SM_Bin*HMF_fun(self.AvaHaloMass[i,M_Cut_bin +j], self.z[i])*h_3*self.AvaHaloMassBins[i,M_Cut_bin +j]
            if len(self.AvaHaloMass[i,M_Cut_bin:M_Cut_bin_upper]) > 0:
                """
                print(self.z[i])
                print(self.AvaStellarMass[i,M_Cut_bin:M_Cut_bin_upper])
                print(self.AvaHaloMass[i,M_Cut_bin:M_Cut_bin_upper])
                print(HMF_fun(self.AvaHaloMass[i,M_Cut_bin:M_Cut_bin_upper], self.z[i]))
                print(self.AvaHaloMassBins[i,M_Cut_bin:M_Cut_bin_upper])
                input()
                #"""
                Total_Cent = np.sum(HMF_fun(self.AvaHaloMass[i,M_Cut_bin:M_Cut_bin_upper], self.z[i])*h_3*self.AvaHaloMassBins[i,M_Cut_bin:M_Cut_bin_upper])
                PairFracTot.append(np.divide(Total_Pair, Total_Cent))
            else:
                PairFracTot.append(np.nan)

        return self.z[1:], PairFracTot[1:], M_L, M_U
    
    def Return_Merger_Plot(self, M0, Mass_Ratio = 0.3):
        Mass_Ratio_Limit = self.AvaStellarMass + np.log10(Mass_Ratio)
        Mass_Ratio_Bins = np.digitize(Mass_Ratio_Limit, bins = self.Surviving_Sat_SMF_MassRange)[:-1]
        Accreted_Above_Ratio, Accreted_Above_Ratio_dz, Accreted_Above_Ratio_dt =  JitLoop(self.Accretion_History, Mass_Ratio_Bins, self.z_step, self.t_step, self.SM_Bin)
        Y_dt = []
        Y_dz = []
        X = []
        for i in range(len(self.z[:-1])):
            try:
                M0_bin_zi = np.digitize(M0, bins = self.AvaStellarMass[i]) -1
                Y_dt.append(np.average(Accreted_Above_Ratio_dt[i, M0_bin_zi:], weights = HMF_fun(self.AvaHaloMass[i, M0_bin_zi:], self.z[i])))
                Y_dz.append(np.average(Accreted_Above_Ratio_dz[i, M0_bin_zi:], weights = HMF_fun(self.AvaHaloMass[i, M0_bin_zi:], self.z[i])))
                #Y_dt.append(Accreted_Above_Ratio_dt[i,M0_bin_zi])
                X.append(self.z[i])
            except Exception as e:
                #print(e)
                break
        return X, Y_dt, Y_dz
    
    def ReturnInterp(self):
        return self.SMF_interp
    
    def ReturnSMHM(self, z):
        Bin = np.digitize(z, bins = self.z)
        return self.AvaHaloMass[Bin]-np.log10(h), self.AvaStellarMass[Bin]
    
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
        return P_ellip
    
    
    def Return_satSMF(self, Redshift):
        AvaHaloMass, AnalyticalModel_SMF, Surviving_Sat_SMF_MassRange, z = F.LoadData_SMFhz([self.RunParam])
        z_bins = np.digitize(Redshift, bins = z)
        return Surviving_Sat_SMF_MassRange, AnalyticalModel_SMF[0][z_bins]
    
    def Return_SSFR(self):
        Surviving_Sat_SMF_MassRange, sSFR_Range, Satellite_sSFR = F.LoadData_sSFR(self.RunParam)
        return Surviving_Sat_SMF_MassRange, sSFR_Range, Satellite_sSFR
        
        
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

                if FirstAddition and (z_start > self.z[i]):
                    P_ellip[i,j] = Major_Frac
                    if CurrentGasFrac >= GasFracThresh:
                        P_lentic[i,j] = Major_FracS0
                    else:
                        P_lentic[i,j] = Major_FracS0 - abs(CurrentGasFrac - GasFracThresh)

                elif (z_start > self.z[i]):
                    if CurrentGasFrac >= GasFracThresh:
                        P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j])
                        P_lentic[i,j] = P_lentic[i+1,j] + Major_FracS0*(1 - P_lentic[i+1,j] - P_ellip[i+1,j])
                    else:
                        P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j] - P_lentic[i+1,j])
                        P_lentic[i,j] = P_lentic[i+1,j] + Major_FracS0*(1 - P_lentic[i+1,j] - P_ellip[i+1,j]) - abs(CurrentGasFrac - GasFracThresh)
                        # P_ellip[i,j] = P_ellip[i+1,j] + Major_Frac*(1 - P_ellip[i+1,j])
            if (z_start > self.z[i]):
                FirstAddition = False
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
    
    #Pair fraction systematic plot========================================================
    if False:
        #using gridspec
        fig = plt.figure(figsize=[18,8])
        gs = GridSpec(8, 18, hspace=0.0,wspace=0.0,figure=fig)
        ax1 = fig.add_subplot(gs[0:3,0:4])
        ax2 = fig.add_subplot(gs[0:4,5:9])
        ax3 = fig.add_subplot(gs[0:4,9:13])
        ax4 = fig.add_subplot(gs[0:3,14:18])
        ax5 = fig.add_subplot(gs[5:8,0:4])
        ax6 = fig.add_subplot(gs[4:8,5:9])
        ax7 = fig.add_subplot(gs[4:8,9:13])
        ax8 = fig.add_subplot(gs[5:8,14:18])




        Master_Interp = Classes[FitList.index(('1.0', True, False, True, 'G19_DPL', 'G19_SE'))].ReturnInterp()
        #TopLeft
        colourcycler = cycle(colours)
        Max = -1; Min = 1
        for Fit in M_Factors:
            colour = next(colourcycler)
            index = FitList.index(Fit)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'M_PFT2'),('1.0', True, False, True, 'G19_DPL', 'M_PFT3'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 2
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax1.plot(Mh, Ms, "--", color = colour)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'M_PFT1'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 0.1
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax1.plot(Mh, Ms, "-", color = colour)
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Master_Interp)
            ax1.fill_between([11,14], [M_L, M_L], y2 =[M_U, M_U], alpha = 0.2, color = colour)
                   
            #For the label
            Label = r"$M"
            if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                Label = "PyMorph"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'M_PFT1'):
                Label += "_{0.1, alt}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'M_PFT2'):
                Label += "_{z+}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'M_PFT3'):
                Label += "_{z-}$"
                
            ax2.semilogy(z, PairFracTot, label = Label, color = colour)
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new





        #TopRight
        colourcycler = cycle(colours)
        for Fit in N_Factors:
            colour = next(colourcycler)
            index = FitList.index(Fit)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'N_PFT2'),('1.0', True, False, True, 'G19_DPL', 'N_PFT3'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 2
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax4.plot(Mh, Ms, "--", color = colour)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'N_PFT1'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 0.1
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax4.plot(Mh, Ms, "-", color = colour)
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Master_Interp)
            ax4.fill_between([11,14], [M_L, M_L], y2 =[M_U, M_U], alpha = 0.2, color = colour)
                               
            #For the label
            Label = r"$N"
            if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                Label = "PyMorph"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'N_PFT1'):
                Label += "_{0.1, alt}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'N_PFT2'):
                Label += "_{z+}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'N_PFT3'):
                Label += "_{z-}$"
                
            ax3.semilogy(z, PairFracTot, label = Label, color = colour)
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new


        #BottomLeft
        colourcycler = cycle(colours)
        for Fit in b_Factors:
            colour = next(colourcycler)
            index = FitList.index(Fit)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'b_PFT2'),('1.0', True, False, True, 'G19_DPL', 'b_PFT3'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 2
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax5.plot(Mh, Ms, "--", color = colour)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'b_PFT1'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 0.1
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax5.plot(Mh, Ms, "-", color = colour)
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Master_Interp, Parent_Cut = 9.5)
            ax5.fill_between([11,14], [M_L, M_L], y2 =[M_U, M_U], alpha = 0.2, color = colour)
                               
            #For the label
            Label = r"$\beta"
            if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                Label = "PyMorph"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'b_PFT1'):
                Label += "_{0.1, alt}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'b_PFT2'):
                Label += "_{z+}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'b_PFT3'):
                Label += "_{z-}$"
                
            ax6.semilogy(z, PairFracTot, label = Label, color = colour)
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new




        #BottomRight
        colourcycler = cycle(colours)
        for Fit in g_Factors:
            colour = next(colourcycler)
            index = FitList.index(Fit)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'g_PFT2'),('1.0', True, False, True, 'G19_DPL', 'g_PFT3'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 2
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax8.plot(Mh, Ms, "--", color = colour)
            if Fit in [('1.0', True, False, True, 'G19_DPL', 'g_PFT1'),('1.0', True, False, True, 'G19_DPL', 'G19_SE')]:
                z = 0.1
                Mh, Ms = Classes[index].ReturnSMHM(z)
                ax8.plot(Mh, Ms, "-", color = colour)
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Master_Interp)
            ax8.fill_between([11,14], [M_L, M_L], y2 =[M_U, M_U], alpha = 0.2, color = colour)
                               
            #For the label
            Label = r"$\gamma"
            if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                Label = "PyMorph"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'g_PFT1'):
                Label += "_{0.1, alt}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'g_PFT2'):
                Label += "_{z+}$"
            elif Fit == ('1.0', True, False, True, 'G19_DPL', 'g_PFT3'):
                Label += "_{z-}$"
                
            ax7.semilogy(z, PairFracTot, label = Label, color = colour)
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new


        #Make ticks
        OneTenth = (Max-Min)/10
        if Min == 0:
            Min = 0.0001
        ax2.xaxis.tick_top()
        ax2.set_xlim(0,3.6)
        ax2.xaxis.set_label_position("top")
        ax2.set_ylim(Min, Max + 2*OneTenth)

        ax3.yaxis.tick_right()
        ax3.xaxis.tick_top()
        ax3.set_xlim(0,3.6)
        ax3.set_ylim(Min, Max + 2*OneTenth)

        ax6.set_xlim(0,3.6)
        ax6.set_ylim(Min, Max + 2*OneTenth)

        ax7.yaxis.tick_right()
        ax7.set_xlim(0,3.6)
        ax7.set_ylim(Min, Max + 2*OneTenth)


        ax1.set_ylim(9, 12)
        ax1.set_xlim(11,14)

        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.set_ylim(9, 12)
        ax4.set_xlim(11,14)

        ax5.set_ylim(9, 12)
        ax5.set_xlim(11,14)

        ax8.yaxis.tick_right()
        ax8.yaxis.set_label_position("right")
        ax8.set_ylim(9, 12)
        ax8.set_xlim(11,14)



        #make legends
        ax2.legend(loc = 8, frameon = False, ncol = 2)#bbox_to_anchor=(-1.3, -0.1),
        ax3.legend(loc = 8, frameon = False, ncol = 2)#bbox_to_anchor=(2.3, -0.1), 
        ax6.legend(loc = 9, frameon = False, ncol = 2) # bbox_to_anchor=(-1.3, 1),
        ax7.legend(loc = 8, frameon = False, ncol = 2) #, bbox_to_anchor=(2.3, 1)

        # Set labels
        Label_FS = 15
        ax2.text(0.1, Max - 2*OneTenth, "M", fontsize=Label_FS)
        ax3.text(0.1, Max - 2*OneTenth, "N", fontsize=Label_FS)
        ax6.text(0.1, Max - 2*OneTenth, r"$\beta$", fontsize=Label_FS)
        ax7.text(0.1, Max - 2*OneTenth, r"$\gamma$", fontsize=Label_FS)
        ax1.text(11.1, 11.75, "M", fontsize=Label_FS)
        ax4.text(11.1, 11.75, "N", fontsize=Label_FS)
        ax5.text(11.1, 11.75, r"$\beta$", fontsize=Label_FS)
        ax8.text(11.1, 11.75, r"$\gamma$", fontsize=Label_FS)
        fig.text(0.5, 0.04, 'z', ha='center', va='center')
        fig.text(0.25, 0.5, '$\mathrm{f_{pair}}$', ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.99, 'z', ha='center', va='center')
        fig.text(0.75, 0.5, '$\mathrm{f_{pair}}$', ha='center', va='center', rotation= -90)
        ax1.set_xlabel("$\mathrm{log_{10}}$ $\mathrm{M_h}$ $\mathrm{[M_\odot]}$")
        ax1.set_ylabel("$\mathrm{log_{10}}$ $\mathrm{M_*}$ $\mathrm{[M_\odot]}$")
        ax4.set_xlabel("$\mathrm{log_{10}}$ $\mathrm{M_h}$ $\mathrm{[M_\odot]}$")
        ax4.set_ylabel("$\mathrm{log_{10}}$ $\mathrm{M_*}$ $\mathrm{[M_\odot]}$", rotation=-90)
        ax4.yaxis.set_label_coords(1.18, 0.5)
        ax5.set_xlabel("$\mathrm{log_{10}}$ $\mathrm{M_h}$ $\mathrm{[M_\odot]}$")
        ax5.set_ylabel("$\mathrm{log_{10}}$ $\mathrm{M_*}$ $\mathrm{[M_\odot]}$")
        ax8.set_xlabel("$\mathrm{log_{10}}$ $\mathrm{M_h}$ $\mathrm{[M_\odot]}$")
        ax8.set_ylabel("$\mathrm{log_{10}}$ $\mathrm{M_*}$ $\mathrm{[M_\odot]}$", rotation=-90)
        ax8.yaxis.set_label_coords(1.18, 0.5)


        plt.tight_layout()
        plt.savefig("Figures/Paper3/PairFractionSystematic.png")
        plt.savefig("Figures/Paper3/PairFractionSystematic.pdf")
        plt.clf()
    #====================================================================

    #Make the Data comparision PF plot===================================
    if False:
        f, SubPlots = plt.subplots(1, 2, figsize = (10, 4))
        Master_Interp = Classes[FitList.index(('1.0', True, False, True, 'G19_DPL', 'G19_cMod'))].ReturnInterp()
        colourcycler = cycle(colours)
        Max = -1; Min = 1
        Fits = [('1.0', True, False, True, 'G19_DPL', 'G19_SE'), ('1.0', True, False, True, 'G19_DPL', 'G19_cMod')]
        #Fits = [('1.0', True, False, True, 'G19_DPL', 'G19_SE'), ('1.0', True, False, True, 'Illustris', 'Illustris')]
        ModelPlots = []
        for Fit in Fits:
            colour = next(colourcycler)
            index = FitList.index(Fit)

            lines = ["-", ":"]
            linecycler = cycle(lines)
            Redshifts = [0.1,2]#[0.1,1,2,3]
            for i, z in enumerate(Redshifts):
                line = next(linecycler)
                Mh, Ms = Classes[index].ReturnSMHM(z)
                if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                    SubPlots[0].plot(Mh, Ms, line, color = colour)
                else:
                    SubPlots[0].plot(Mh, Ms, line, color = colour)


            Classes[index].ReturnInterp()
            #z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Master_Interp, Parent_Cut = 10, UpperLimit = True)
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Classes[index].ReturnInterp(), Parent_Cut = 10, UpperLimit = True)
            if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                ModelPlots.append(SubPlots[1].semilogy(z, PairFracTot, "-.", color = colour)[0])
            else:
                ModelPlots.append(SubPlots[1].semilogy(z, PairFracTot, "-.", color = colour)[0])
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new
                
            #"""
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Master_Interp, Parent_Cut = 11, UpperLimit = True)
            if Fit == ('1.0', True, False, True, 'G19_DPL', 'G19_SE'):
                ModelPlots.append(SubPlots[1].semilogy(z, PairFracTot, "-", label = r"$> 10^{11} M_{\odot}$", color = colour)[0])
            else:
                ModelPlots.append(SubPlots[1].semilogy(z, PairFracTot, "-", label = " ", color = colour)[0])
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new
            #"""
        
        MundyPlots = []
        """
        f0, m, N = 0.028, 0.80, 0.5
        MundyPlots.append(SubPlots[1].semilogy(np.arange(z[0], z[-1], 0.4), (f0*np.power(1+np.arange(z[0], z[-1], 0.4), m)),  "+",label = r"$> 10^{10} M_{\odot}$", mfc = None)[0])
        """
        f0, m, N = 0.024, 0.78, 0.5
        MundyPlots.append(SubPlots[1].semilogy(np.arange(z[0], z[-1], 0.4), (f0*np.power(1+np.arange(z[0], z[-1], 0.4), m)),  "xk",label = r"$> 10^{11} M_{\odot}$")[0])
        
        """
        #Add illustris
        #z_10 = [0, 0.1, 0.5, 0.7, 1, 1.5, 2, 3]
        #PF_ill_10 = [-2.48, -2.50, -2.35, -2.26, -2.07,-2.54,-2.19,-2.83]
        z_10 = [0.1, 0.5, 1, 1.5, 2, 3]
        PF_ill_10 = [-2.41,-2.42,-2.26,-2.27,-2.37,-2.63]
        #z_11 = [0, 0.5, 1, 2, 3]
        #PF_ill_11 = [-2.91,-3.17,-2.60, -2.24,-2.25]
        SubPlots[1].semilogy(z_10, np.power(10, PF_ill_10), "x", color = 'k', label = 'Illustris TNG')
        #SubPlots[1].semilogy(z_11, np.power(10, PF_ill_11), "-.", color = 'k')
        SMHM_ill = np.load(AbsPath+'/../Data/Observational/Illustris/SMHM_fillbtwn_99.npy')
        SubPlots[0].fill_between(SMHM_ill[0], SMHM_ill[1], SMHM_ill[2], color = 'k', alpha = 0.5, label = 'Illustris TNG')
        
        #Sneaky Labels
        SubPlots[0].plot([0,1], [0,1],"-", color = "C0", label = "STEEL: PyMorph")
        SubPlots[0].plot([0,1], [0,1],"-", color = "C1", label = "STEEL: Illustris")
        
        
        SubPlots[0].legend(frameon = False)
        SubPlots[1].legend(frameon = False)
        #"""
        
        #"""
        #Sneaky Labels
        Leg1 = []
        Leg1.append(SubPlots[0].plot([0,1], [0,1],"x", color = "k", label = "Mundy+17")[0])
        Leg1.append(SubPlots[0].plot([0,1], [0,1],"-", color = "C0", label = "PyMorph")[0])
        Leg1.append(SubPlots[0].plot([0,1], [0,1],"-", color = "C1", label = "cmodel")[0])
        Leg2 = []
        Leg2.append(SubPlots[0].plot([0,1], [0,1],"-", color = "k", label = "z = 0.1")[0])
        Leg2.append(SubPlots[0].plot([0,1], [0,1],":", color = "k", label = "z = 2.0")[0])
        Leg3 = []
        Leg3.append(SubPlots[1].plot([],[],"-", color = "k", label = r"M$_\odot$: 10$^{11}$")[0])
        Leg3.append(SubPlots[1].plot([],[],"-.", color = "k", label = r"M$_\odot$: 10$^{10}$")[0])
        
        Legend1 = SubPlots[0].legend(handles = Leg1, frameon = False, loc = 2)
        SubPlots[0].add_artist(Legend1)
        Legend2 = SubPlots[0].legend(handles = Leg2, frameon = False, loc = 4)
        #ßSubPlots[0].add_artist(Legend2)
        Legend3 = SubPlots[1].legend(handles = Leg3, frameon = False, ncol = 2, loc = 9)
        SubPlots[1].add_artist(Legend3)
        #"""
        
        if Min <= 0:
            Min = 0.0001
        SubPlots[1].set_ylim(Min, Max*10)#+0.1)
        SubPlots[1].set_xlim(0.0, 3.5)
        SubPlots[1].set_xlabel("z")
        SubPlots[1].set_ylabel("$\mathrm{f_{pair}}$")
        SubPlots[0].set_ylim(9, 12.5)
        SubPlots[0].set_xlim(11, 15)
        SubPlots[0].set_xlabel("$\mathrm{log_{10}}$ $\mathrm{M_h}$ $\mathrm{[M_\odot]}$")
        SubPlots[0].set_ylabel("$\mathrm{log_{10}}$ $\mathrm{M_*}$ $\mathrm{[M_\odot]}$")
        plt.tight_layout()
        plt.savefig("Figures/Paper3/PairFractionData.png")
        plt.savefig("Figures/Paper3/PairFractionData.pdf")
        #plt.savefig("Figures/Paper3/Test.png")
        #plt.savefig("Figures/Paper3/Test.pdf")
        plt.clf()

        
        
    #Make the Mevo PF plot===================================
    if False:
        f, SubPlots = plt.subplots(1, 2, figsize = (10, 4))
        colourcycler = cycle(colours)
        Max = -1; Min = 1
        ModelPlots = []
        LegGz = []
        GzFacs = ["cmodel", "0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
        for i, Fit in enumerate(HMevo_Factors):
            if Fit == ('1.0', False, False, True, 'G19_DPL', 'G19_cMod'):
                index = FitList.index(Fit)
                Mh, Ms = Classes[index].ReturnSMHM(0.1)
                SubPlots[0].plot(Mh, Ms, ":", color = "k")
            colour = next(colourcycler)
            index = FitList.index(Fit)
            Mh, Ms = Classes[index].ReturnSMHM(2)
            if Fit == ('1.0', False, False, True, 'G19_DPL', 'G19_cMod'):
                LegGz.append(SubPlots[0].plot(Mh, Ms, "-", color = colour, label = GzFacs[i])[0])
            else:
                LegGz.append(SubPlots[0].plot(Mh, Ms, "-", color = colour, label = r"$\gamma_{z}$ = "+GzFacs[i])[0])

            Classes[index].ReturnInterp()
            z, PairFracTot, M_L, M_U = Classes[index].Return_PF_Plot(Classes[index].ReturnInterp(), Parent_Cut = 11, UpperLimit = True)
            ModelPlots.append(SubPlots[1].semilogy(z, PairFracTot, "-", color = colour)[0])
            Max_new = np.nanmax(PairFracTot); Min_new = np.nanmin(PairFracTot)
            if Max_new > Max:
                Max = Max_new
            if Min_new < Min:
                Min = Min_new
        
        MundyPlots = []
        f0, m, N = 0.024, 0.78, 0.5
        MundyPlots.append(SubPlots[1].semilogy(np.arange(z[0], z[-1], 0.4), (f0*np.power(1+np.arange(z[0], z[-1], 0.4), m)),  "xk",label = r"$> 10^{11} M_{\odot}$")[0])
        
        #"""
        #Sneaky Labels
        Leg1 = []
        Leg1.append(SubPlots[1].plot([], [],"x", color = "k", label = "Mundy+17")[0])
        Leg2 = []
        Leg2.append(SubPlots[0].plot([], [],":", color = "k", label = "z = 0.1")[0])
        Leg2.append(SubPlots[0].plot([], [],"-", color = "k", label = "z = 2.0")[0])
        Leg3 = []
        Leg3.append(SubPlots[1].plot([],[],"-", color = "k", label = r"M$_\odot$: 10$^{11}$")[0])
        
        LegendGz = SubPlots[0].legend(handles = LegGz, frameon = False, loc = 2, ncol = 2, prop={'size': 10})
        SubPlots[0].add_artist(LegendGz)
        Legend1 = SubPlots[1].legend(handles = Leg1, frameon = False, loc = 2)
        SubPlots[1].add_artist(Legend1)
        Legend2 = SubPlots[0].legend(handles = Leg2, frameon = False, loc = 4)
        #SubPlots[0].add_artist(Legend2)
        Legend3 = SubPlots[1].legend(handles = Leg3, frameon = False, loc = 4)
        #SubPlots[1].add_artist(Legend3)
        #"""
        
        if Min <= 0:
            Min = 0.0001
        SubPlots[1].set_ylim(Min, Max*10)#+0.1)
        SubPlots[1].set_xlim(0.0, 3.5)
        SubPlots[1].set_xlabel("z")
        SubPlots[1].set_ylabel("$\mathrm{f_{pair}}$")
        SubPlots[0].set_ylim(9, 12.5)
        SubPlots[0].set_xlim(11, 15)
        SubPlots[0].set_xlabel("$\mathrm{log_{10}}$ $\mathrm{M_h}$ $\mathrm{[M_\odot]}$")
        SubPlots[0].set_ylabel("$\mathrm{log_{10}}$ $\mathrm{M_*}$ $\mathrm{[M_\odot]}$")
        plt.tight_layout()
        plt.savefig("Figures/Paper3/PairFractionHMevo.png")
        plt.savefig("Figures/Paper3/PairFractionHMevo.pdf")
        plt.clf()
        
        
        
    #MergerRate Plot    
    if False:
        def Mundy_MR(z, R, m, c = None):
            if c == None:
                return R*np.power(1+z, m)
            else:
                return R*np.power(1+z, m)*np.exp(-c*z)
            
        MassRatio = 0.3   
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twiny()#add time axis on top
        for Fit in [('1.0', True, True, True, 'G19_DPL', 'G19_SE')]:#, 'G18_0.8Dyn']:
            lines = ["--","-", "-.", ":"]
            linecycler = cycle(lines)
            colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
            colourcycler = cycle(colours)
            index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
            for j, M0 in enumerate([10.0, 11.0]):
                colour = next(colourcycler)
                line = next(linecycler)           
                X, Y_dt, Y_dz = Classes[index].Return_Merger_Plot(M0)
                #print(np.shape(X), np.shape(Y_dt), np.shape(Y_dz))
                ax.semilogy(X[1:], Y_dt[1:], line, label = r"$M_{*, cen} >$"+"$10^{%s}$"%M0+r"$M_{\odot}$", color = colour, linewidth = 2.2)

            #ax.semilogy(np.take(X, np.arange(0, len(X), 10)), Mundy_MR(np.take(np.array(X), np.arange(0, len(X), 10)), 1.73*(10**-2), 4.13, c = 1.41), "+",label = r"$> 10^{10} M_{\odot}$")
            #ax.semilogy(np.take(X, np.arange(0, len(X), 10)), Mundy_MR(np.take(np.array(X), np.arange(0, len(X), 10)), 1.79*(10**-2), 2.79, c = 0.93), "x",label = r"$> 10^{11} M_{\odot}$")
            ax.semilogy(np.take(X, np.arange(0, len(X), 10)), Mundy_MR(np.take(np.array(X), np.arange(0, len(X), 10)), 1.73*(10**-2), 4.13, c = 1.41), "+",label = r"$> 10^{10} M_{\odot}$")
            ax.semilogy(np.take(X, np.arange(0, len(X), 10)), Mundy_MR(np.take(np.array(X), np.arange(0, len(X), 10)), 1.79*(10**-2), 2.79, c = 0.93), "x",label = r"$> 10^{11} M_{\odot}$")
        ax.set_ylim(10**-3, 10**0.5)
        ax.set_xlim(0.1, 2.5)
        ax2.set_xlim(ax.get_xlim())
        #Extra ticks
        upper_x_tick_loc = [Cosmo.lookbackTime(2, inverse = True), Cosmo.lookbackTime(3, inverse = True), Cosmo.lookbackTime(5, inverse = True), Cosmo.lookbackTime(7, inverse = True), Cosmo.lookbackTime(11, inverse = True)]
        ax2.set_xticks(upper_x_tick_loc)
        ax2.set_xticklabels([2,3,5,7,11])
        ax2.set_xlabel("Lookback Time [Gyr]")
        ax2.minorticks_off()
        ax.set_xlabel("$z$")
        ax.set_ylabel("$dN/dt$ $[Gyr^{-1}]$")
        ax.text(0.2, 10**-2.7, r"$\frac{M_{*, sat}}{M_{*,cen}} >$" + "{}".format(MassRatio))
        ax.legend(loc = 9, frameon = False, ncol = 2, title = "            STEEL G19                        Mundy+ 17    ")
        plt.tight_layout()
        plt.savefig("Figures/Paper2/GalaxyMergerRate.png")
        plt.savefig("Figures/Paper2/GalaxyMergerRate.pdf")
        plt.clf()
        
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
        
        MassRatio = 0.25
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        # P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 3)
        P_ellip = Classes[index].Return_Morph_Plot(MassRatio, 2)

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
        
        
        plt.plot(Classes[index].AvaStellarMass[0], P_ellip[0], "-k",label = "STEEL, z = 0.1")
        # z_plot = 1.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_ellip[np.digitize(z_plot, bins = Classes[index].z)], "--C0", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))
        # z_plot = 2.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_ellip[np.digitize(z_plot, bins = Classes[index].z)], "-.C3", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))
        plt.xlim(10, 12.3)
        plt.text(10.2, 0.4, r"$\frac{M_{*, sat}}{M_{*,cen}} >$" + "{}".format(MassRatio))
        plt.legend(frameon = False)
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
        GasFracThresh = 0.09
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_Gas_Hard_Threshold_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)

        
        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "STEEL, z = 0.1")
        # z_plot = 1.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_lentic[np.digitize(z_plot, bins = Classes[index].z)], "--C0", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))
        # z_plot = 2.0
        # plt.plot(Classes[index].AvaStellarMass[np.digitize(z_plot, bins = Classes[index].z)], P_lentic[np.digitize(z_plot, bins = Classes[index].z)], "-.C3", alpha = 0.9,label = "STEEL, z = {}".format(z_plot))


        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.legend(frameon = False)
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Gas_Fraction_Hard_Threshold.png")
        plt.savefig("Figures/Paper2/Gas_Fraction_Hard_Threshold.pdf")
        plt.clf()


        plt.plot(sm_bins[1:], F_Len, "k^", label = "SDSS", fillstyle = "none", markersize=15)
        plt.xlabel("$log_{10}$ $M_*$ [$M_\odot$]")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        plt.ylabel("$f_{lenticular}$")#, fontproperties = mpl.font_manager.FontProperties(size = 15))
        
        MassRatio = 0.25
        MassRatioS0 = 0.05
        GasFracThresh = 0.09
        
        # index = FitList.index(('1.0', True, True, True, 'G19_DPL', 'G19_SE'))
        index = FitList.index(('1.0', False, False, True, 'CE', 'G19_SE'))
        P_lentic = Classes[index].Return_Gas_Soft_Threshold_Plot(MassRatio, MassRatioS0, 2, GasFracThresh)

        
        plt.plot(Classes[index].AvaStellarMass[0], P_lentic[0], "-k",label = "STEEL, z = 0.1")

        # plt.text(10.2, 0.8, "{}".format(MassRatioS0) + r"< $\frac{M_{*, sat}}{M_{*,cen}} <$" + "{}".format(MassRatio))
        plt.text(10.2, 0.55, r"GFT = " + "{}".format(GasFracThresh))
        plt.legend(frameon = False)
        plt.xlim(10,12)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig("Figures/Paper2/Gas_Fraction_Soft_Threshold.png")
        plt.savefig("Figures/Paper2/Gas_Fraction_Soft_Threshold.pdf")
        plt.clf()

    #Satellite Accretion plot
    def SFR(M, z):
        s0 = 0.6 + 1.22*(z) - 0.2*(z**2)
        logM0 = 10.3 + 0.753*(z) - 0.15*(z**2)
        Gamma = -(1.3 - 0.1*(z))# - 0.03*(z[i]**2))#including -ve here to avoid it later
        log10MperY = s0 - np.log10(1 + np.power(np.power(10, (M - logM0) ), Gamma))
        return log10MperY
    def SFR_s_fit(SM, z):
        """
        Calculates Starformation rate
        Args:
        SM: Stellar Mass [log10 Msun]
        z: Redshift
        Returns:
        SFR: Star formation rate [log10 Msun yr-1]
        """

        #Schreiber 2015
        m = SM-9; r = np.log10(1+z)
        m0, a0, a1, m1, a2 = 0.75, 1.75, 0.3, 0.36, 1.75 # 0.5 (up down), 1.5(z up down), 0.3 (bend left right), 0.36 (bend tightness), 2.5 (bend tightness z)
        Max = m-m1-a2*r
        Max[Max<0] = 0
        return m-m0+a0*r-a1*np.power(Max, 2)
    if False:     
        #for k, Fit in enumerate([('1.0', True, True, True, 'G19_DPL', 'G19_SE')]):
        f, SubPlots = plt.subplots(3, 3, figsize = (12,7), sharex = 'col', sharey = 'row')
        #colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
        colours = ["C0", "C2", "C3", "C6"]
        colourcycler = cycle(colours)
        #[HMevo_Factors[0], HMevo_Factors[1], HMevo_Factors[2], HMevo_Factors[6]]
        for k, Fit in enumerate([HMevo_Factors[0], HMevo_Factors[2], HMevo_Factors[3], HMevo_Factors[6]]):
            colour = next(colourcycler)
            #f, SubPlots = plt.subplots(3, 3, figsize = (12,7), sharex = 'col', sharey = 'row')
            #colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
            #colourcycler = cycle(colours)
            DataClass = Classes[FitList.index(Fit)]
            z_intp = DataClass.z[DataClass.z < 4]
            SatelliteMasses = np.power(10, DataClass.Surviving_Sat_SMF_MassRange)
            Mass_Accretion_PerCentral = np.zeros_like(DataClass.AvaStellarMass)
            Mass_Accretion_PerCentral_Minor = np.zeros_like(DataClass.AvaStellarMass)
            Mass_Accretion_PerCentral_Major = np.zeros_like(DataClass.AvaStellarMass)
            
            
            #Open the file from Joel
            File = open(AbsPath+"/../Data/Observational/Leja_SFR/sfh_stack_pip.pickle", "rb")
            #Load the data from pickle, note latin1 required due to python 2 to python 3
            Data = pickle.load(File, encoding='latin1')
            for i in range(np.shape(DataClass.AvaStellarMass)[0]-1, -1, -1):
                for j in range(np.shape(DataClass.AvaStellarMass)[1]-1, -1, -1): 
                    #CutOff = np.digitize(DataClass.AvaStellarMass[i,j] - 4, DataClass.Surviving_Sat_SMF_MassRange)-1#set a mass ratio limit 
                    CutOff = np.digitize(9, DataClass.Surviving_Sat_SMF_MassRange)-1#set a mass ratio limit 
                    if CutOff<0:CutOff = 0  
                    MassAcc = np.sum(DataClass.Accretion_History[i,j,CutOff:]*SatelliteMasses[CutOff:])*DataClass.SM_Bin*0.612 #Calculates the total acreted stellar mass per central mass factor of 0.612 from moster 2018 assuming in any given merger ~40% of the mass of the satellite is distributed to the ICM
                    if (j == None):
                        print(MassAcc)
                    if MassAcc > 0:
                        Mass_Accretion_PerCentral[i,j] = MassAcc
                    else:
                        Mass_Accretion_PerCentral[i,j] = 0
                        
                        
            for i in range(np.shape(DataClass.AvaStellarMass)[0]-1, -1, -1):
                for j in range(np.shape(DataClass.AvaStellarMass)[1]-1, -1, -1): 
                    MergerThreshold = np.digitize(DataClass.AvaStellarMass[i,j]+np.log10(0.3), bins = DataClass.Surviving_Sat_SMF_MassRange)-1
                    MassAcc_Minor = np.sum(DataClass.Accretion_History[i,j,:MergerThreshold]*SatelliteMasses[:MergerThreshold])*DataClass.SM_Bin*0.612 #Calculates the total acreted stellar mass per central mass     factor of 0.612 from moster 2018 assuming in any given merger ~40% of the mass of the satellite is distributed to the ICM
                    MassAcc_Major = np.sum(DataClass.Accretion_History[i,j,MergerThreshold:]*SatelliteMasses[MergerThreshold:])*DataClass.SM_Bin*0.612 #Calculates the total acreted stellar mass per central mass     factor of 0.612 from moster 2018 assuming in any given merger ~40% of the mass of the satellite is distributed to the ICM
                    if (j == None):
                        print(MassAcc)
                    if MassAcc_Minor > 0:
                        Mass_Accretion_PerCentral_Minor[i,j] = MassAcc_Minor
                    else:
                        Mass_Accretion_PerCentral_Minor[i,j] = 0
                    if MassAcc_Major > 0:
                        Mass_Accretion_PerCentral_Major[i,j] = MassAcc_Major
                    else:
                        Mass_Accretion_PerCentral_Major[i,j] = 0
                        
                        
            #colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
            #colourcycler = cycle(colours)
            
            #Output the CE from the diffrence between unity and the galaxy accretion rate
            Masses_for_CE = [np.digitize(i, bins = DataClass.AvaStellarMass[0])-1 for i in np.arange(9, np.max(DataClass.AvaStellarMass[0]), 0.1)]
            for i_, i in enumerate(Masses_for_CE):
                #Central Mass Growth interp
                CentralMass = np.power(10, DataClass.AvaStellarMass[:,i])
                CentralMassGrowth = CentralMass[:-1] - CentralMass[1:]
                CentralMassGrowth = np.insert(CentralMassGrowth, -1,CentralMassGrowth[-1])
                dt_CMG = Cosmo.lookbackTime(DataClass.z[1:]) - Cosmo.lookbackTime(DataClass.z[:-1]) 
                dt_CMG = np.insert(dt_CMG, len(dt_CMG)-1, dt_CMG[-1]) # timesteps in gyr
                Cent_Loss_Rate = np.zeros_like(CentralMassGrowth)
                for j, Mass in enumerate(CentralMassGrowth):
                    if j > 1:
                        f_loss = 0.05*np.log(1+np.divide((np.flip(np.cumsum(np.flip(dt_CMG[:j])))*(10**3)), 1.4)) #Moster+18
                        loss_rate = np.divide(np.insert(f_loss[1:] - f_loss[:-1], -1, 0)*Mass, dt_CMG[:j]*(10**9)) #Msun yr-1
                        Cent_Loss_Rate[:j] = Cent_Loss_Rate[:j] + loss_rate #Msun yr-1
                CMG_dt = np.divide(CentralMassGrowth, dt_CMG*(10**9)) - Cent_Loss_Rate #Central Mass Growth dM/dt Msun yr-1
                CM_interp = interpolate.interp1d(DataClass.z, CentralMass)
                CMG_dt_interp = interpolate.interp1d(DataClass.z, CMG_dt) 
                N = 3
                X_acc_hz, Y_acc_hz = np.convolve(DataClass.z, np.ones((N,))/N, mode='valid'), np.convolve( np.divide(Mass_Accretion_PerCentral[:,i], CentralMassGrowth), np.ones((N,))/N, mode='valid')
                if i_ > 0:
                    z_CE= np.vstack((z_CE, X_acc_hz[:-1]))
                    Mass_CE= np.vstack((Mass_CE, np.convolve(CentralMass, np.ones((N,))/N, mode='valid')[1:]))
                    SFR_CE= np.vstack((SFR_CE, (1-Y_acc_hz[:-1])*np.convolve(CMG_dt, np.ones((N,))/N, mode='valid')[:-1]))
                    AccRt = np.vstack((AccRt , (Y_acc_hz[:-1])*np.convolve(np.divide(CentralMassGrowth, dt_CMG*(10**9)), np.ones((N,))/N, mode='valid')[:-1]))
                else:
                    z_CE=X_acc_hz[:-1]
                    Mass_CE=np.convolve(CentralMass, np.ones((N,))/N, mode='valid')[1:]
                    SFR_CE=(1-Y_acc_hz[:-1])*np.convolve(CMG_dt, np.ones((N,))/N, mode='valid')[:-1]
                    AccRt =(Y_acc_hz[:-1])*np.convolve(np.divide(CentralMassGrowth, dt_CMG*(10**9)), np.ones((N,))/N, mode='valid')[:-1]
            np.save("Scripts/CentralPostprocessing/HaloMassTrackCE", np.vstack((Mass_CE.flatten(), SFR_CE.flatten(), z_CE.flatten(), AccRt.flatten())))
            np.save("Scripts/CentralPostprocessing/HaloMassTrackCE_M", Mass_CE)
            np.save("Scripts/CentralPostprocessing/HaloMassTrackCE_SFR", SFR_CE)
            np.save("Scripts/CentralPostprocessing/HaloMassTrackCE_Z", z_CE)
            np.save("Scripts/CentralPostprocessing/HaloMassTrackCE_AccRt", AccRt)
            
            #Useful redshift bins
            zbinpt5 = np.digitize(0.5, bins = DataClass.z)
            zbinpt6 = np.digitize(0.6, bins = DataClass.z)
            zbin1 = np.digitize(1, bins = DataClass.z)
            zbin2 = np.digitize(2, bins = DataClass.z)
            zbin3 = np.digitize(3, bins = DataClass.z)
            zbin4 = np.digitize(4, bins = DataClass.z)
            zbin5 = np.digitize(5, bins = DataClass.z)
            
            for i_, i in enumerate([np.digitize(12, bins = DataClass.AvaStellarMass[0])-1, np.digitize(11.5, bins = DataClass.AvaStellarMass[0])-1, np.digitize(11, bins = DataClass.AvaStellarMass[0])-1]):
            #for i_, i in enumerate([np.digitize(10.5, bins = DataClass.AvaStellarMass[0])-1, np.digitize(9.5, bins = DataClass.AvaStellarMass[0])-1, np.digitize(9, bins = DataClass.AvaStellarMass[0])-1]):
                #colour = next(colourcycler)
                

                
                #for printing masses of the MPB at diffrent redhsifts
                if True:
                    print(" z0 Mass:", DataClass.AvaStellarMass[0,i])
                    print(" z0.5:", round(DataClass.AvaStellarMass[zbinpt5,i], 2),\
                          " z0.6:", round(DataClass.AvaStellarMass[zbinpt6,i], 2),\
                          " z1:", round(DataClass.AvaStellarMass[zbin1,i], 2),\
                          " z2:", round(DataClass.AvaStellarMass[zbin2,i], 2),\
                          " z3:", round(DataClass.AvaStellarMass[zbin3,i], 2),\
                          " z4:", round(DataClass.AvaStellarMass[zbin4,i], 2),\
                          " z5:", round(DataClass.AvaStellarMass[zbin5,i], 2))
                    print("\n")
                
                #Central Mass Growth interp
                CentralMass = np.power(10, DataClass.AvaStellarMass[:,i])
                CentralMassGrowth = CentralMass[:-1] - CentralMass[1:]
                CentralMassGrowth = np.insert(CentralMassGrowth, -1,CentralMassGrowth[-1])
                dt_CMG = Cosmo.lookbackTime(DataClass.z[1:]) - Cosmo.lookbackTime(DataClass.z[:-1]) 
                dt_CMG = np.insert(dt_CMG, -1, dt_CMG[-1]) # timesteps in gyr
                CMG_dt = np.divide(CentralMassGrowth, dt_CMG*(10**9)) #Central Mass Growth dM/dt Msun yr-1    
                CM_interp = interpolate.interp1d(DataClass.z, CentralMass)
                CMG_dt_interp = interpolate.interp1d(DataClass.z, CMG_dt)
                
                #Create the SFH using the central with mass accertion
                #Moving average here to smooth out the scatters in the instantaneous rate
                N = 3
                X_acc_hz_SFH, Y_acc_hz_SFH = np.convolve(DataClass.z, np.ones((N,))/N, mode='valid'), np.convolve(Mass_Accretion_PerCentral[:,i], np.ones((N,))/N, mode='valid')
                
                Accretion_Interp = interpolate.interp1d(DataClass.z, np.flip(np.cumsum(np.flip(Mass_Accretion_PerCentral[:,i], 0))), 0)
                dt_acc_hz = Cosmo.lookbackTime(X_acc_hz_SFH[1:]) - Cosmo.lookbackTime(X_acc_hz_SFH[:-1]) 
                dt_acc_hz = np.insert(dt_acc_hz, len(dt_acc_hz)-1, dt_acc_hz[-1]) # timesteps in gyr
                Accretion_Interp_dt = interpolate.interp1d(X_acc_hz_SFH, np.divide(Y_acc_hz_SFH, dt_acc_hz*(10**9)), fill_value = "extrapolate")#dM/dt Msun yr-1
                
                #Imputs for the SFH code
                z_start = 5
                z_for_SFH = np.flip(np.arange(np.min(X_acc_hz_SFH), z_start, 0.01), 0)
                t = F.RedshiftToTimeArr(0) - F.RedshiftToTimeArr(z_for_SFH)
                d_t = t[:-1] -t[1:]
                d_t = np.insert(d_t, 0, d_t[0])
                M_acc_dot = Accretion_Interp_dt(z_for_SFH)
                MaxGas, Tquench, Tau_f = 100, -1, 0
                
                M_out, M_dot, M_dot_noacc, SFH, GMLR = F_c.Starformation_Centrals(DataClass.AvaStellarMass[zbin5,i], t, d_t, z_for_SFH, M_acc_dot, MaxGas, Tquench, Tau_f, SFR_Model = "G19_DPL", Scatter_On = 0)
                M_out, M_dot, M_dot_noacc, SFH, GMLR = np.power(10, np.array(M_out)), np.array(M_dot), np.array(M_dot_noacc), np.array(SFH), np.array(GMLR)
                np.save("Scripts/CentralPostprocessing/GalaxyTracks{}".format(round(DataClass.AvaStellarMass[0,i],1)), np.vstack((z_for_SFH, M_out, M_dot_noacc, GMLR)))
                #Msun, Myr-1, Myr-1      , M  , Myr-1
                #print(M_out, "\n", M_dot, "\n", M_dot_noacc, "\n", SFH, "\n", GMLR)
                #Cumlative SFH
                Mass = np.cumsum(SFH) +np.power(10, DataClass.AvaStellarMass[zbin5,i])
                Mass_Intp = interpolate.interp1d(z_for_SFH, Mass)
                
                #Panel 1: Cumlative total of mass from satellite accretion or SFH
                #Total
                SubPlots[0, i_].plot(DataClass.z, DataClass.AvaStellarMass[:,i], "-", color = colour)
                #SFH
                #SubPlots[0, i_].plot(z_for_SFH, np.log10(Mass), ":", color = colour)
                #Accretion
                SubPlots[0, i_].plot(DataClass.z, np.flip(np.log10(np.cumsum(np.flip(Mass_Accretion_PerCentral[:,i], 0))), 0), "--", color = colour)
                #SubPlots[0, i_].plot(DataClass.z, np.flip(np.log10(np.cumsum(np.flip(Mass_Accretion_PerCentral_Minor[:,i], 0))), 0), "-.", color = colour)
                #SubPlots[0, i_].plot(DataClass.z, np.flip(np.log10(np.cumsum(np.flip(Mass_Accretion_PerCentral_Major[:,i], 0))), 0), "-.", color = colour)
                #set to true for making the cartoon insert
                if i_ == 0 and False:
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
                    plt.rcParams['font.size']=22
                    plt.rcParams['lines.linewidth']=5
                    CartoonFig = plt.figure()
                    CartoonPlot = CartoonFig.add_subplot(1,1,1) 
                    CartoonPlot.plot(DataClass.z, DataClass.AvaStellarMass[:,i], "-C1", label = " Total")
                    CartoonPlot.plot(DataClass.z, np.flip(np.log10(np.cumsum(np.flip(Mass_Accretion_PerCentral[:,i], 0))), 0), "--C1", label = " Satellite\n Accretion")
                    CartoonPlot.set_xscale('log')
                    CartoonPlot.set_xlim(0.1, 3)
                    CartoonPlot.set_ylim(10, 12)
                    CartoonPlot.set_xticks([0.1,0.5,1,2])
                    CartoonPlot.set_xticklabels(["0.1","0.5","1", "2"])
                    CartoonPlot.set_xlabel("z")
                    CartoonPlot.set_ylabel(r"log10 M$_*$ [M$_{\odot}]$")
                    CartoonPlot.legend(frameon=False)
                    CartoonFig.tight_layout()
                    CartoonFig.savefig("Figures/Paper2/GrowthHistCartoon_diff.png")
                    CartoonFig.clf()
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
                    plt.rcParams['lines.linewidth']=2
                    plt.rcParams['font.size']=15
            
                #Panel 2: Fraction of total mass from satellite accretion or SFH since z = 3                
                #The ratio from SFH
                SFH_zbin3 = np.digitize(3, bins = z_for_SFH)
                Ratio_SFH = np.divide(Mass[SFH_zbin3:]-Mass[SFH_zbin3], CM_interp(z_for_SFH[SFH_zbin3:])-CM_interp(z_for_SFH[SFH_zbin3]))
                #SubPlots[1, i_].plot(z_for_SFH[SFH_zbin3:], Ratio_SFH, ":", color = colour)                
                
                #The ratio from Satellite Accretion
                Ratio_Acc = np.divide(Accretion_Interp(z_for_SFH[SFH_zbin3:]) - Accretion_Interp(z_for_SFH[SFH_zbin3]), CM_interp(z_for_SFH[SFH_zbin3:])-CM_interp(z_for_SFH[SFH_zbin3]))
                SubPlots[1, i_].plot(z_for_SFH[SFH_zbin3:], Ratio_Acc, "--", color = colour)
                
                #Total
                #SubPlots[1, i_].plot(z_for_SFH[SFH_zbin3:], Ratio_SFH+Ratio_Acc, "-", color = colour)
                
                #Panel 3: Instaneous mass rates
                #Moving averages here to smooth out the scatters in the instantaneous rates
                N = 3
                X_acc_hz, Y_acc_hz = np.convolve(DataClass.z, np.ones((N,))/N, mode='valid'), np.convolve( np.divide(Mass_Accretion_PerCentral[:,i], CentralMassGrowth), np.ones((N,))/N, mode='valid')
                SubPlots[2, i_].plot(X_acc_hz[9:], Y_acc_hz[9:], "--", color = colour)
                #SubPlots[2, i_].plot(z_for_SFH, np.divide(M_dot_noacc, CMG_dt_interp(z_for_SFH)), ":", color = colour)
                #SubPlots[2, i_].plot(z_for_SFH, np.divide(M_dot, CMG_dt_interp(z_for_SFH)), "-", color = colour)               
                
                #Adding crosses from leja
                """
                #Calculate the SFR by multipying the median sSFR by the mass
                SFR_06 = Data['0.6']['sfr_med']*np.power(10, np.full((7, np.size(Data['0.6']['mvec'])) , Data['0.6']['mvec']).T)
                #Get the redshift of the SFH steps by calculating the total lookback time then convering back to redshift 
                z_06 = Cosmo.lookbackTime(Cosmo.lookbackTime(0.6)+np.mean(10**Data['0.6']['agebins']/10**9, axis = 1), inverse = True)
                
                #Mass at z = 0.6
                if DataClass.AvaStellarMass[zbinpt6,i] < np.max(Data['0.6']['mvec']):
                    M_bin = np.digitize(DataClass.AvaStellarMass[zbinpt6,i], bins = Data['0.6']['mvec'])
                    print(z_06[:-2])
                    print(SFR_06[M_bin][:-2])
                    print(CMG_dt_interp(z_06[:-2]))
                    print(np.divide(SFR_06[M_bin][:-2], CMG_dt_interp(z_06[:-2])))
                    SubPlots[2, i_].plot(z_06[:-2], np.divide(SFR_06[M_bin][:-2], CMG_dt_interp(z_06[:-2])), "x", color = colour)
                    
                #Calculate the SFR by multipying the median sSFR by the mass
                SFR_1 = Data['1.0']['sfr_med']*np.power(10, np.full((7, np.size(Data['1.0']['mvec'])) , Data['1.0']['mvec']).T)
                #Get the redshift of the SFH steps by calculating the total lookback time then convering back to redshift 
                z_1 = Cosmo.lookbackTime(Cosmo.lookbackTime(1.0)+np.mean(10**Data['1.0']['agebins']/10**9, axis = 1), inverse = True)
                
                #Mass at z = 1
                if DataClass.AvaStellarMass[zbin1,i] < np.max(Data['1.0']['mvec']):
                    M_bin = np.digitize(DataClass.AvaStellarMass[zbin1,i], bins = Data['1.0']['mvec'])
                    print(z_1[:-2])
                    print(SFR_1[M_bin][:-2])
                    print(CMG_dt_interp(z_1[:-2]))
                    print(np.divide(SFR_1[M_bin][:-2], CMG_dt_interp(z_1[:-2])))
                    SubPlots[2, i_].plot(z_1[:-2], np.divide(SFR_1[M_bin][:-2], CMG_dt_interp(z_1[:-2])), "x", color = colour)
                    
                #Calculate the SFR by multipying the median sSFR by the mass
                SFR_2 = Data['2.0']['sfr_med']*np.power(10, np.full((7, np.size(Data['2.0']['mvec'])) , Data['2.0']['mvec']).T)
                #Get the redshift of the SFH steps by calculating the total lookback time then convering back to redshift 
                z_2 = Cosmo.lookbackTime(Cosmo.lookbackTime(2.0)+np.mean(10**Data['2.0']['agebins']/10**9, axis = 1), inverse = True)
                
                #Mass at z = 2.0
                if DataClass.AvaStellarMass[zbin2,i] < np.max(Data['2.0']['mvec']):
                    M_bin = np.digitize(DataClass.AvaStellarMass[zbin2,i], bins = Data['2.0']['mvec'])
                    print(z_2[:-2])
                    print(SFR_2[M_bin][:-2])
                    print(CMG_dt_interp(z_2[:-2]))
                    print(np.divide(SFR_2[M_bin][:-2], CMG_dt_interp(z_2[:-2])))
                    SubPlots[2, i_].plot(z_2[:-2], np.divide(SFR_2[M_bin][:-2], CMG_dt_interp(z_2[:-2])), "x", color = colour)
                #"""
                
                
                #plots off axis for labels  
                #SubPlots[2, i_].plot([7,8,9], [0.5, 0.5, 0.5], "-",label = "$M_{*,cen} = $"+"$10^{%.3g}$"%DataClass.AvaStellarMass[0,i]+"$M_{\odot}$", color = colour)
                SubPlots[0, i_].set_title("$M_{*,cen} = $"+"$10^{%.3g}$"%DataClass.AvaStellarMass[0,i]+"$M_{\odot}$")
            
            
            #Unity lines
            SubPlots[1, 0].axhline(1, 0.001, 3, linestyle = "-", color = "k", alpha = 0.5) 
            SubPlots[2, 0].axhline(1, 0.001, 3, linestyle = "-", color = "k", alpha = 0.5)
            SubPlots[1, 1].axhline(1, 0.001, 3, linestyle = "-", color = "k", alpha = 0.5) 
            SubPlots[2, 1].axhline(1, 0.001, 3, linestyle = "-", color = "k", alpha = 0.5)
            SubPlots[1, 2].axhline(1, 0.001, 3, linestyle = "-", color = "k", alpha = 0.5) 
            SubPlots[2, 2].axhline(1, 0.001, 3, linestyle = "-", color = "k", alpha = 0.5)
            
            #Adding Illustris
            """
            colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
            colourcycler = cycle(colours)
            for i_, i in enumerate([12,11.5,11]):#
                colour = next(colourcycler)
                z = np.load("./Data/Observational/Illustris/z_{}.npy".format(i))
                Mcen_l, Mcen_u = np.load("./Data/Observational/Illustris/Mcen_{}.npy".format(i))
                Macc_l, Macc_u = np.load("./Data/Observational/Illustris/Macc_Mcen_{}.npy".format(i))
                Macc_Mcen_l, Macc_Mcen_u = np.load("./Data/Observational/Illustris/Macc_{}.npy".format(i))

                SubPlots[0, i_].fill_between(z,  Mcen_l, Mcen_u, alpha = 0.25, color = colour)
                SubPlots[0, i_].fill_between(z, Macc_l, Macc_u, alpha = 0.25, facecolor = "none", hatch = "X", edgecolor = colour)
                SubPlots[1, i_].fill_between(z, Macc_Mcen_l, Macc_Mcen_u, alpha = 0.25, color = colour)
                SubPlots[2, i_].plot([10, 11],[1, 1])
            #"""
            #Adding Moster
            """
            colours = ["C2", "C1", "C0"]
            colourcycler = cycle(colours)
            Emerge = h5py.File(AbsPath+"/../Data/Observational/Moster_EMERGE/mainbranches.S85.h5")
            for i, Masscut in enumerate(list(Emerge.keys())):
                colour = next(colourcycler)
                Switch = True
                data = Emerge[Masscut]
                Trees = list(data.keys())
                for tree_id in Trees:
                    Tree = Emerge[Masscut][tree_id]
                    SF = Tree['Scale_factor']
                    SF_cut = np.digitize(0.2, bins = SF)+1
                    SF = SF[:SF_cut]
                    SM = Tree['Stellar_mass'][:SF_cut]
                    SM_insitu = Tree['Insitu_mass'][:SF_cut]
                    SFR = Tree['SFR'][:SF_cut]
                    CenSat = Tree['Type'][:SF_cut]
                    H_ID = Tree['Halo_ID'][:SF_cut]
                    D_ID = Tree['Desc_ID'][:SF_cut]

                    SM_growth = np.power(10, SM[0:-2]) - np.power(10, SM[1:-1])
                    SMis_growth = np.power(10, SM_insitu[0:-2]) - np.power(10, SM_insitu[1:-1])
                    t = Cosmo.lookbackTime((1/SF)-1)
                    z = Cosmo.lookbackTime((t[1:-1] + t[0:-2])/2, inverse = True)
                    delta_t = t[1:-1] - t[0:-2]
                    SM_dt = SM_growth/delta_t
                    SMis_dt = SMis_growth/delta_t

                    if Switch:
                        Shape = np.shape(CenSat)
                        z_stack = z
                        M_stack = np.power(10, SM[0:-2]) + np.power(10, SM[1:-1])/2
                        Mis_stack = np.power(10, SM_insitu[0:-2]) + np.power(10, SM_insitu[1:-1])/2
                        Mes_stack = (np.power(10, SM[0:-2]) + np.power(10, SM[1:-1])/2) - (np.power(10, SM_insitu[0:-2]) + np.power(10, SM_insitu[1:-1])/2)
                        rsf_stack = np.divide(SFR[0:-2]+SFR[1:-1],2)*10**9/SM_dt
                        ris_stack = SMis_dt/SM_dt
                        Switch = False
                    elif Shape == np.shape(CenSat):
                        z_stack = np.vstack((z_stack, z))
                        M_stack = np.vstack((M_stack, np.power(10, SM[0:-2]) + np.power(10, SM[1:-1])/2))
                        Mis_stack = np.vstack((Mis_stack, np.power(10, SM_insitu[0:-2]) + np.power(10, SM_insitu[1:-1])/2))
                        Mes_stack = np.vstack((Mes_stack, (np.power(10, SM[0:-2]) + np.power(10, SM[1:-1])/2) - (np.power(10, SM_insitu[0:-2]) + np.power(10, SM_insitu[1:-1])/2)))
                        rsf_stack = np.vstack((rsf_stack, np.divide(SFR[0:-2]+SFR[1:-1],2)*10**9/SM_dt))
                        ris_stack = np.vstack((ris_stack,SMis_dt/SM_dt))
                M_ava = np.log10(np.mean(M_stack, axis = 0)) + np.log10(h)
                Mis_ava = np.log10(np.mean(Mis_stack, axis = 0)) + np.log10(h)
                Mes_ava = np.log10(np.mean(Mes_stack, axis = 0)) + np.log10(h)
                SubPlots[0,2-i].plot(z_stack[0], M_ava, "-", color = "k")#colour)
                SubPlots[0,2-i].plot(z_stack[0], Mis_ava, ":", color = "k")#colour)
                SubPlots[0,2-i].plot(z_stack[0], Mes_ava, "--", color = "k")#colour)
                
                z_cut = np.digitize(3, z_stack[0])+1
                
                Mava_3 = np.power(10,M_ava[:z_cut]) - np.power(10,M_ava[z_cut])
                Misava_3 = np.divide(np.power(10,Mis_ava[:z_cut]) - np.power(10,Mis_ava[z_cut]), Mava_3)
                Mesava_3 = np.divide(np.power(10,Mes_ava[:z_cut]) - np.power(10,Mes_ava[z_cut]), Mava_3)
                SubPlots[1,2-i].plot(z_stack[0, :z_cut], Misava_3, ":", color = "k")#colour)
                SubPlots[1,2-i].plot(z_stack[0, :z_cut], Mesava_3, "--", color = "k")#colour)
                
            #"""
            
            #Adding Behroozi
            """
            for i, file in enumerate(["stats_a0.911185_absolute_sm_11.500_cen.dat", "stats_a0.911185_absolute_sm_11.000_cen.dat"]):
                print(file)
                Header = ["type", "SF", "a", "b", "c", "avg", "avg_err", "sd", "counts"]
                df = pd.read_csv(AbsPath+"/../Data/Observational/Behroozi_UnviM/sfh_stats/"+file, sep = " ", skiprows = 17, names = Header, usecols = [0,1,2,3,4,6,8,10,12])
                df2 = df.pivot(index = 'SF', columns = "type", values = 'avg')
                for j in df2.columns:
                    df2[j] = df2[j].map(lambda x: float(str(x)[:-1]))
                df2 = df2.assign(z = lambda x: (1/x.index.values)-1)
                z= df2['z'].values
                t = Cosmo.lookbackTime(1/df2.index.values - 1)
                dt = (t[:-1]-t[1:])*(10**9)
                SM = df2['sm_mp'].values
                SM_dt = (SM[1:]-SM[:-1])/dt
                z_med = Cosmo.lookbackTime((t[:-1]+t[1:])/2, inverse = True)
                SFR = df2['sfr_mp'].values*0.59 #inst mass recycling
                SFR_med = (SFR[:-1] + SFR[1:])/2
                Ratio = SFR_med/SM_dt
                SubPlots[0,i+1].plot(z, np.log10(SM), "-", color = "k")
                SubPlots[2,i+1].plot(z_med, Ratio, ":", color = "k")
            #"""
            
            #Adding Menci
            """
            Acc11 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/Macc_11.dat").T
            Acc11pt5 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/Macc_11.5.dat").T
            Acc12 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/Macc_12.dat").T
            Main11 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/Mmain_11.dat").T
            Main11pt5 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/Mmain_11.5.dat").T
            Main12 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/Mmain_12.dat").T
            Ratio11 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/ratio_M11.dat").T
            Ratio11pt5 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/ratio_M11.5.dat").T
            Ratio12 = np.loadtxt(AbsPath+"/../Data/Observational/Nicola_SAM/ratio_M12.dat").T
            
            SubPlots[0,0].plot(Main12[0], np.log10(Main12[1]) + 12, color = "k")
            SubPlots[0,1].plot(Main11pt5[0], np.log10(Main11pt5[1]) + 11.5, color = "k")
            SubPlots[0,2].plot(Main11[0], np.log10(Main11[1]) + 11, color = "k")
            
            SubPlots[0,0].plot(Acc12[0], np.log10(Acc12[1]),"--", color = "k")
            SubPlots[0,1].plot(Acc11pt5[0], np.log10(Acc11pt5[1]),"--", color = "k")
            SubPlots[0,2].plot(Acc11[0], np.log10(Acc11[1]), "--", color = "k")
            
            SubPlots[1,0].plot(Ratio12[0], Ratio12[1], "--", color = "k")
            SubPlots[1,1].plot(Ratio11pt5[0], Ratio11pt5[1], "--", color = "k")
            SubPlots[1,2].plot(Ratio11[0], Ratio11[1], "--", color = "k")
            #"""
            
            
            
            #Line labels
            #SubPlots[0,2].plot([4,5,6], [0.5, 0.5, 0.5], "--",label = "Accretion", color = "k")
            #SubPlots[0,2].plot([4,5,6], [0.5, 0.5, 0.5], ":", label = "SFH", color = "k")
            #SubPlots[0,2].plot([4,5,6], [0.5, 0.5, 0.5], "-", label = "Total", color = "k")
            
            #Legends
            SubPlots[0,2].legend(ncol = 2,frameon = False, loc = 9, fontsize = 12)
            SubPlots[2,0].legend(ncol = 1, frameon = False, loc = 1, fontsize = 12)
            SubPlots[2,1].legend(ncol = 1, frameon = False, loc = 1, fontsize = 12) 
            SubPlots[2,2].legend(ncol = 1, frameon = False, loc = 1, fontsize = 12)
            
            #Log axis
            SubPlots[0,0].set_xscale('log')
            SubPlots[1,0].set_xscale('log')
            SubPlots[2,0].set_xscale('log')
            SubPlots[1,0].set_yscale('log')
            SubPlots[2,0].set_yscale('log')
            SubPlots[0,1].set_xscale('log')
            SubPlots[1,1].set_xscale('log')
            SubPlots[2,1].set_xscale('log')
            SubPlots[1,1].set_yscale('log')
            SubPlots[2,2].set_yscale('log')
            SubPlots[0,2].set_xscale('log')
            SubPlots[1,2].set_xscale('log')
            SubPlots[2,2].set_xscale('log')
            SubPlots[1,2].set_yscale('log')
            SubPlots[2,2].set_yscale('log')  
            
            """
            #Arrows
            SubPlots[0,0].arrow(0.35, 11.3, 0.0, 0.3, head_width = 0.04, head_length = 0.15, length_includes_head = True, fill = None)
            SubPlots[1,0].arrow(0.85, 2.5*(10**-1), 0.0, 0.2, head_width = 0.1, head_length = 0.075, length_includes_head = True, fill = None)
            SubPlots[2,0].arrow(1.92, 0.25, 0.0, 0.1, head_width = 0.2, head_length = 0.05, length_includes_head = True, fill = None)
            SubPlots[2,1].arrow(1.1, 0.3, 0.0, 0.1, head_width = 0.1, head_length = 0.05, length_includes_head = True, fill = None)
            #ArrowLabels
            SubPlots[0,0].text(0.35, 11.1, "A")
            SubPlots[1,0].text(0.85, 2*(10**-1), "B")
            SubPlots[2,0].text(1.9, 0.2, "C")
            SubPlots[2,1].text(1.1, 0.25, "D")
            #"""
            
            #Ticks
            #X
            SubPlots[2,0].set_xticks([0.1,0.5,1,2])
            SubPlots[2,0].set_xticklabels(["0.1","0.5","1", "2"])
            SubPlots[2,1].set_xticks([0.1,0.5,1,2])
            SubPlots[2,1].set_xticklabels(["0.1","0.5","1", "2"])
            SubPlots[2,2].set_xticks([0.1,0.5,1,2,3])
            SubPlots[2,2].set_xticklabels(["0.1","0.5","1","2","3"])
            
                        
            SubPlots[0,0].set_yticks([9, 10, 11, 12])
            SubPlots[0,0].set_yticklabels(["9","10", "11","12"])
            SubPlots[0,1].set_yticks([9, 10, 11, 12])
            SubPlots[0,1].set_yticklabels(["9","10", "11","12"])
            SubPlots[0,2].set_yticks([9, 10, 11, 12])
            SubPlots[0,2].set_yticklabels(["9","10", "11","12"])            
            
            Ticks = [0.1, 0.5, 1]
            Labels = ["0.1", "0.5", "1"]
            SubPlots[2,0].set_yticks(Ticks)
            SubPlots[2,0].set_yticklabels(Labels)
            SubPlots[2,1].set_yticks(Ticks)
            SubPlots[2,1].set_yticklabels(Labels)
            SubPlots[2,2].set_yticks(Ticks)
            SubPlots[2,0].set_yticklabels(Labels)


            #Axis Limits
            SubPlots[2,0].set_xlim(0.1, 3)
            SubPlots[0,0].set_ylim(9, 12.5)
            SubPlots[1,0].set_ylim(0.02, 2)
            SubPlots[2,0].set_ylim(0.1, 2)
            SubPlots[2,1].set_xlim(0.1, 3)
            SubPlots[0,1].set_ylim(9, 12.5)
            SubPlots[1,1].set_ylim(0.02, 2)
            SubPlots[2,1].set_ylim(0.1, 2)
            SubPlots[2,2].set_xlim(0.1, 3)
            SubPlots[0,2].set_ylim(9, 12.5)
            SubPlots[1,2].set_ylim(0.02, 2)
            SubPlots[2,2].set_ylim(0.1, 2)

            #Axis Labels
            SubPlots[2,0].set_xlabel("z")
            SubPlots[0,0].set_ylabel(r"log10 M$_*$ M$_{\odot}$")
            #SubPlots[1,0].set_ylabel(r"$\sum_{i=3}^{0}  M_{X,i} \div \sum_{i=3}^{0}  M_{cen,i}$")
            SubPlots[1,0].set_ylabel(r"$\frac{M_{X,z}-M_{X,3}}{M_{cen,z}-M_{cen,3}}$") 
            SubPlots[2,0].set_ylabel(r"$\dot{M}_{X} \div \dot{M}_{cen}$")
            #Axis Labels
            SubPlots[2,1].set_xlabel("z")
            SubPlots[2,2].set_xlabel("z")
            #SubPlots[0,1].set_ylabel(r"log10 Mass $M_{\odot}$")
            #SubPlots[1,1].set_ylabel(r"$\frac{M_{*,acc}}{M_{*,cen}}$")
            #SubPlots[2,1].set_ylabel(r"$\frac{dM_{*,acc}/dt}{dM_{*,cen}/dt}$")
            
            #Adjust and Save
            plt.subplots_adjust(hspace=0, wspace=0)
            #plt.tight_layout()
            
            #plt.savefig("Figures/Paper2/SatelliteAccretion{}.png".format(Fit_to_Str(Fit)))
            #plt.savefig("Figures/Paper2/SatelliteAccretion{}.pdf".format(Fit_to_Str(Fit)))
            #plt.clf()
        colours = ["C0", "C2", "C3", "C6"]
        colourcycler = cycle(colours)
        SubPlots[0,0].plot([],[], "-", color = next(colourcycler), label = "cmodel")
        for i in [HMevo_Factors[2], HMevo_Factors[3], HMevo_Factors[6]]:
            SubPlots[0,0].plot([],[], "-", color = next(colourcycler), label = r"$\gamma_{z}$ = "+i[5][-3:])
        
        SubPlots[0,2].plot([4,5,6], [0.5, 0.5, 0.5], "--",label = "Accretion", color = "k")
        SubPlots[0,2].plot([4,5,6], [0.5, 0.5, 0.5], "-", label = "Total", color = "k")
        SubPlots[0,0].legend(ncol = 1, frameon = False, loc = 3, fontsize = 12)
        SubPlots[0,2].legend(ncol = 2,frameon = False, loc = 9, fontsize = 12)
        SubPlots[2,0].legend(ncol = 1, frameon = False, loc = 1, fontsize = 12)
        SubPlots[2,1].legend(ncol = 1, frameon = False, loc = 1, fontsize = 12) 
        SubPlots[2,2].legend(ncol = 1, frameon = False, loc = 1, fontsize = 12)
        plt.savefig("Figures/Paper3/SatelliteAccretion.png".format(Fit_to_Str(Fit)))
        plt.savefig("Figures/Paper3/SatelliteAccretion.pdf".format(Fit_to_Str(Fit)))
        plt.clf()
            
            
            

    
    #Make the SMF
    if False:
        colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
        colourcycler = cycle(colours)
        Redshifts = [0,1.5,3]
        f, SubPlots = plt.subplots(1, len(Redshifts), figsize = (12,4), sharex = True, sharey = 'row')
        for i, Fit in enumerate([('1.0', True, True, True, 'G19_DPL', 'G19_SE')]):
            colour = next(colourcycler)
            DataClass = Classes[FitList.index(Fit)]
            for j, z_ in enumerate(Redshifts):
                Surviving_Sat_SMF_MassRange, AnalyticalModel_SMF = DataClass.Return_satSMF(z_)
                SubPlots[j].plot(Surviving_Sat_SMF_MassRange, np.log10(AnalyticalModel_SMF), '--', color = colour, label = Fit)
                if i == 0:
                    Xcen, Ycen, Bin = DataClass.Return_Cent_SMF(z_)#, SMF_X = Surviving_Sat_SMF_MassRange)
                    SubPlots[j].plot(Xcen, Ycen, '-.', color = "k")
                    SubPlots[j].text(11.5, -2, "z = {}".format(round(z_,2)))
        SubPlots[0].set_ylim(-6, -1.5)
        SubPlots[0].set_xlim(9, 12.5)
        SubPlots[0].legend(loc = 3, frameon = False)
        
        SubPlots[0].set_ylabel("$log_{10}  \phi$ $[Mpc^{-3} dex^{-1}]$")
        SubPlots[1].set_xlabel(r"log10 Mass $M_{\odot}$")
        
        plt.tight_layout()
        plt.savefig("Figures/Paper2/SMF.png")
        plt.savefig("Figures/Paper2/SMF.pdf")
        plt.clf()
        
    #Make the sSFR distribution
    if False:
        f, SubPlots = plt.subplots(1, 3, figsize = (10,3), sharey = True)
        FirstPass = True
        No_Leg = False
        lines = ["-", "--","-."]
        colours = colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "k"]
        linecycler = cycle(lines)
        colourcycler = cycle(colours)
        x,y=0,0
        Tdyn_Factors = [('1.0', True, True, True, 'G19_DPL', 'G19_SE')] #['G19_SE_DPL_NOCE_SF', 'G19_SE_DPL_NOCE_SF_Strip']
        
        MassRatio = 0.25
        for i, Fit in enumerate(Tdyn_Factors):
            index = FitList.index(Fit)
            P_ellip = np.repeat(Classes[index].Return_Morph_Plot(MassRatio, 3)[0], 100)
            DataClass = Classes[FitList.index(Fit)]
            Paramaters['SFR_Model'] = 'G19_DPL'
            
            Central_SM = np.repeat(DataClass.AvaStellarMass[0], 100)
            Central_SFR = F.StarFormationRate(Central_SM, 0.1, Paramaters, ScatterOn = True, Quenching = True, P_ellip = P_ellip)
            Central_sSFR = Central_SFR - Central_SM
            Central_wt = HMF_fun(Central_SM, 0.1)*np.repeat(DataClass.AvaHaloMassBins[0], 100)
            
            Line = next(linecycler)
            Colour = next(colourcycler)
            Surviving_Sat_SMF_MassRange, sSFR_Range, Satellite_sSFR = DataClass.Return_SSFR()
            bin_w = Surviving_Sat_SMF_MassRange[1]-Surviving_Sat_SMF_MassRange[0]
            x = 0
            for l,u in [(10,10.5),(10.5,11.3),(11.3,12.5)]:#[(10,10.5),(10.5,11),(11,12)]:
                #Satellites
                Weights = np.sum(Satellite_sSFR[np.digitize(l, bins = Surviving_Sat_SMF_MassRange):np.digitize(u, bins = Surviving_Sat_SMF_MassRange)], axis = 0)
                N_Ntot = Weights/(np.sum(Weights)*bin_w)
                #Centrals
                
                N_Ntot_cen = np.histogram(Central_sSFR[np.digitize(l, bins = DataClass.AvaStellarMass[0])*100:np.digitize(u, bins = DataClass.AvaStellarMass[0])*101], bins = sSFR_Range, weights = Central_wt[np.digitize(l, bins = DataClass.AvaStellarMass[0])*100:np.digitize(u, bins = DataClass.AvaStellarMass[0])*101], density = True)[0]

                SubPlots[x].set_title("{}-{}".format(l,u) + "$M_{*, sat}$")
                if FirstPass == True:    
                    if i == 0 and x == 0:
                        No_Leg = False
                    else:
                        No_Leg = True
                    A = Add_SDSS.sSFR_Plot(l, u, SubPlots[x], No_Leg = No_Leg)
                    B = Add_SDSS.sSFR_Plot_Cen(l, u, SubPlots[x], No_Leg = No_Leg)
                if x==1:
                    if len(Tdyn_Factors) == 1:
                        Label = "Satellites:\nDynamical Quenching"

                    else:
                        Label = "{}".format(Factor)
                    SubPlots[x].plot(sSFR_Range, N_Ntot, Line, color = Colour,label = Label, alpha = 0.75)
                    SubPlots[x].plot(sSFR_Range[:-1], N_Ntot_cen, "--", color = 'k',label = "Centrals", alpha = 0.75)
                else:
                    SubPlots[x].plot(sSFR_Range, N_Ntot, Line, color = Colour,alpha = 0.75)
                    SubPlots[x].plot(sSFR_Range[:-1], N_Ntot_cen, "--", color = 'k', alpha = 0.75)
                x +=1
            FirstPass = False

        SubPlots[0].set_xlim(-13, -9.0)
        SubPlots[1].set_xlim(SubPlots[0].get_xlim())
        SubPlots[2].set_xlim(SubPlots[0].get_xlim())
        SubPlots[0].legend(loc = 2, frameon = False, fontsize = 12)
        SubPlots[1].legend(loc = 9, frameon = False, fontsize = 12)

        SubPlots[1].set_xlabel("log10(sSFR) [$yr^{-1}$]", fontproperties = mpl.font_manager.FontProperties(size = 15))
        #print(ax.get_xticks())  
        SubPlots[0].set_xticklabels(["-13", "-12", "-11", "-10", ""])
        SubPlots[1].set_xticklabels(["-13", "-12", "-11", "-10", ""])
        SubPlots[2].set_xticklabels(["-13", "-12", "-11", "-10", "-9"])

        #when you come looking for how to do this
        #ticks = plt.xticks(); plt.xticks(ticks[:-1])


        plt.subplots_adjust(wspace=0, hspace=0)
        #tik.TickHelper
        #plt.tight_layout()
        plt.savefig("Figures/Paper2/SSFR.png", bbox_inches='tight')
        plt.savefig("Figures/Paper2/SSFR.pdf", bbox_inches='tight')   
