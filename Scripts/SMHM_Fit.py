#Author: Philip Grylls
#Intent: Provide a script through which a parameterisation of the stellar halo mass relation at all redshifts is derived given SMF and HMF as input
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions as F #imports my functions file
import SDSS_Plots #imports the SDSS file
import multiprocessing
from numba import vectorize
from scipy.interpolate import interp1d, interp2d
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from fast_histogram import histogram1d, histogram2d
cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()
h = Cosmo.h
h_3 = h*h*h
if "SDSS_Plots.pkl" in os.listdir("./Bernardi_SDSS"):
    Add_SDSS = pickle.load(open("./Bernardi_SDSS/SDSS_Plots.pkl", 'rb'))
else:
    Add_SDSS = SDSS_Plots.SDSS_Plots(11.5,15,0.1) #pass this halomass:min, max, and binwidth for amting 
    pickle.dump(Add_SDSS, open("./Bernardi_SDSS/SDSS_Plots.pkl", 'wb'))



class HaloMassFunction:
    #A class in which to put all possible halo mass function (HMF) varients
    #Each HMF should return N Mpc^-3 h^3 dex^-1
    def __init__(self):
        #Set the standard HMF range for the fitting
        self.HMR_L = 10
        self.HMR_U = 16
        self.HMR_BW = 0.1
        self.HaloMassRange = np.arange(self.HMR_L, self.HMR_U , self.HMR_BW) #log10 Mvir h-1 Msun
        self.HaloMassFunction = F.Make_HMF_Interp() #loads the HMF/COLOSSUS hmf from STEEL
    
    #Functions for returning defaults
    def ReturnDefaultMassRange(self):
        return self.HaloMassRange, self.HMR_L, self.HMR_U , self.HMR_BW
    
    """
    Functions for returning HMF
    Each Function should take as arguments the following:
    z: A float or array of floats at the redshifts of intrest, shape (N,)
    HMF: Optional keyworded argument. A float or array of floats, shapes (M,) (N,M)
    The output should then be:
    If N = M = 1, output shape (1,)
    If N > 1, M = 1, output shape (N,)
    If N = 1, M > 1, output shape (M,)
    If N > 1, M > 1, output shape (N, M)
    Output should have units N Mpc^-3 h^3 dex-1
    """
    def hmf_vir(self, z):
        """
        Returns the tinker10 HMF in viral coordinates using Make_HMF_Interp(), which uses hmf then COLOSSUS to convert to virial masses
        Arguments:
            z: A float or array of floats at the redshifts of intrest, Units: Redshift
            HMF: Optional keyworded argument. A float or 1/2d array of floats, Units: log10 Mvir Msun
        Returns: 
            HMF_Fun: A array of dimensions given above for input halo-redshift numberdensties
        """
        HMR = self.HaloMassRange
        #Check if M is 2d if so make z the smae dimensions and then flatten
        Flat = False
        if np.size(np.shape(HMR)) == 2:
            assert np.shape(HMR)[0] == np.size(z)
            z = (np.full(np.shape(HMR.T), z).T).flatten()
            OutputShape = np.shape(HMR)
            HMR = HMR.flatten()
            Flat = True
    
        HMF_fun = self.HaloMassFunction(HMR, z)
        
        
        if np.size(HMR) == 1 and np.size(z) == 1:
            assert np.shape(HMF_fun) == (1,)
            return HMF_fun
        elif np.size(HMR) == 1:
            HMF_fun = HMF_fun[:,0]
            assert np.shape(HMF_fun) == np.shape(z)
            return HMF_fun
        elif np.size(z) == 1:
            assert np.shape(HMF_fun) == np.shape(HMR)
            return HMF_fun
        elif Flat:
            HMF_fun = np.reshape(HMF_fun, OutputShape)
            assert np.shape(HMF_fun) == OutputShape
            return HMF_fun
        else:
            assert np.shape(HMF_fun) == (np.size(z), np.size(HMR))
            return HMF_fun
    
    def SHMF_STEEL(self, z_in):
        RunParam = (1.0, False, False, True, 'CE', 'G18')
        z, SubHaloMass, NumberDensities = F.LoadData_MultiEpoch_SubHalos(RunParam)
        z_bin = np.digitize(z_in, bins = z) - 1
        return SubHaloMass, NumberDensities[z_bin], z
            


class StellarMassFunction:
    #A class in which to put all possible stellar mass function (SMF) varients
    #Each SMF should return N Mpc^-3 dex^-1
    def __init__(self):
        #Set the standard HMF range for the fitting
        self.SMR_L = 9
        self.SMR_U = 12.5
        self.SMR_BW = 0.1
        self.StellarMassRange = np.arange(self.SMR_L, self.SMR_U , self.SMR_BW) #log10 Mvir h-1 Msun
    
    #Functions for returning defaults
    def ReturnDefaultMassRange(self):
        return self.StellarMassRange, self.SMR_L, self.SMR_U , self.SMR_BW
    
    def Bernardi_SDSS(self):
        """
        Returns a Sersic Exponential Fit SMF from SDSS-DR7 Meert+(2015, 1016), using the high mass fit Bernardi+(2013, 2017a,b)
        Arguments:
            SMR: Optional Keyworded. Stellar mass range, must have cinsistant binwiths. Units log10 Msun
        Returns: 
            Y_t: Total SMF log10 N Mpc^-3 dex^-1
            Y_t_e: Error on total SMF
            Y_sat: Satellite SMF log10 N Mpc^-3 dex^-1
            Y_sat_e: Error on Satellite SMF
            Y_cen: Central SMF log10 N Mpc^-3 dex^-1
            Y_cen_e: Error on Central SMF
        """
        SMR = self.StellarMassRange
        SMF_Bins, Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e = Add_SDSS.SMF_Data(SMF_Bins = np.insert(SMR, 0,np.min(SMR) - self.SMR_BW)-0.05)
        Redshifts = [0.1]
        return Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e, Redshifts
    
    def cModel_SDSS(self):
        """
        Returns a Sersic Exponential Fit SMF from SDSS-DR7 Meert+(2015, 1016), using the high mass fit Bernardi+(2013, 2017a,b)
        Arguments:
            SMR: Optional Keyworded. Stellar mass range, must have cinsistant binwiths. Units log10 Msun
        Returns: 
            Y_t: Total SMF log10 N Mpc^-3 dex^-1
            Y_t_e: Error on total SMF
            Y_sat: Satellite SMF log10 N Mpc^-3 dex^-1
            Y_sat_e: Error on Satellite SMF
            Y_cen: Central SMF log10 N Mpc^-3 dex^-1
            Y_cen_e: Error on Central SMF
        """
        SMR = self.StellarMassRange
        SMF_Bins, Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e = Add_SDSS.SMF_Data(SMF_Bins = np.insert(SMR, 0,np.min(SMR) - self.SMR_BW)-0.05, OverridePhoto = "MsMendCmodel")
        Redshifts = [0.1]
        return Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e, Redshifts
    
    def Davidzon_17(self, Corr = False):
        SMR = self.StellarMassRange
        DavDat = []
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax0.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax1.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax2.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax3.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax4.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax5.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax6.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax7.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        #DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax8.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        #DavDat.append(pd.read_csv("./Dav17_SMF/mf_mass2b_fl5b_tot_Vmax9.dat", header=None, delim_whitespace=True, names = ["M", "Phi", "err_up", "err_down"]))
        Redshifts = [0.37,0.668,0.938,1.286,1.735,2.220,2.683,3.271]
        
        DavDat_Interp = []
        
        if Corr:
            Shift = 0.15
        else:
            Shift = 0
        
        for i in DavDat:
            DavDat_Interp.append(interp1d(i.M + Shift, i.Phi, bounds_error = False, fill_value = np.nan)(SMR))
        
        return DavDat_Interp, Redshifts
    
    #def Bernardi_SDSS_cmod(self):
    #    SMR = self.StellarMassRange
    #    Mendal = np.loadtxt("./Lorenzo_SDSS/SMF_MsMendelcModel.txt")        
    #    return interp1d(Mendal[:,0], np.log10(Mendal[:,1]), bounds_error = False, fill_value = np.nan)(SMR), [0.1]
        
    def Bernardi_SDSS_Mous(self):
        SMR = self.StellarMassRange
        Moustakas = np.loadtxt("./Lorenzo_SDSS/moustakas.dat")
        return interp1d(Moustakas[:,0], Moustakas[:,1], bounds_error = False, fill_value = np.nan)(SMR), [0.1]

    

class HaloMassToStellarMass:
    #A class in which to put different Stellar Mass Halo Mass (SMHM) relations
    #Functions should come in pairs one to define the fit one to return that runtion and an inital guess array for fitting
   
    def Moster(self, DM, Inputs, z, Pairwise = False):
        M10, M11, SHMnorm10, SHMnorm11, beta10, beta11, gamma10, gamma11, Scatter = Inputs
        zparameter = np.divide(z-0.1, z+1)
        #putting the parameters together for inclusion in the Moster 2010 equation
        M = M10 + M11*zparameter
        N = SHMnorm10 + SHMnorm11*zparameter
        b = beta10 + beta11*zparameter
        g = gamma10 + gamma11*zparameter
        
        # Moster 2010 eq2
        if Pairwise:
            SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
        elif ((np.shape(DM) == np.shape(z)) or np.shape(z) == (1,) or np.shape(z) == ()):
            SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
        else:
            if (np.shape(DM)[0] != np.shape(z)[0]):
                M = np.full((np.size(DM), np.size(z)), M).T
                N = np.full((np.size(DM), np.size(z)), N).T
                b = np.full((np.size(DM), np.size(z)), b).T
                g = np.full((np.size(DM), np.size(z)), g).T
                DM = np.full((np.size(z), np.size(DM)), DM)
            SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))

        Scatter_Arr = np.random.normal(scale = Scatter, size = np.shape(SM))
        return( np.log10(SM) + Scatter_Arr)
    
    def ReturnMosterFit(self):
        Inputs = [  11.95,   0.4,        0.032,          -0.02,       1.61,         -0.6,       0.54,        -0.1,       0.15]
        Bounds = [[11,13], [0,2], [0.02, 0.04], [-0.03, -0.01], [1.2, 1.8], [-0.9, -0.5], [0.5, 0.7], [-0.2, 0.2], [0.1, 0.2]]
        return self.Moster, Inputs, Bounds
    
    
class Fitting_Functions:
    
    def __init__(self, SMHM = "Moster", SMF = ["Bernardi16", "Davidzon17_corr"], HMF_central = "HMF_Collosus", HMF_sub = "STEEL"):
        #Inherit the other three classes
        self.SMF_Class = StellarMassFunction()
        self.HMF_Class = HaloMassFunction()
        self.SMHM_Class = HaloMassToStellarMass()
        
        #Arrays constructed for interpolation
        self.SMF_Redshifts = []
        self.SMF_Central = []
        self.SMF_Total = []
        self.SMF_Satellite = []
        self.SMF_Central_Err = []
        self.SMF_Total_Err = []
        self.SMF_Satellite_Err = []
        self.SHMF = []
        self.CHMF = []
        self.Low_z = []
        self.High_z = []
        
        self.HMR = self.HMF_Class.ReturnDefaultMassRange()
        self.SMR = self.SMF_Class.ReturnDefaultMassRange()
        #Pick SMHM relation and initial parameters
        if SMHM == "Moster":
            self.SMHM_Model, self.Parameters, self.Bounds = self.SMHM_Class.ReturnMosterFit()
        else:
            print("Error: SMHM model not defined")
        #Pick SMF data
        if "Bernardi16" in SMF:
            Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e, Redshifts = self.SMF_Class.Bernardi_SDSS() 
            self.SMF_Redshifts += Redshifts
            self.SMF_Central.append(Y_cen)
            self.SMF_Total.append(Y_t)
            self.SMF_Satellite.append(Y_sat)
            self.SMF_Central_Err.append(Y_cen_e)
            self.SMF_Total_Err.append(Y_t_e)
            self.SMF_Satellite_Err.append(Y_sat)     
        if "cModel" in SMF:
            Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e, Redshifts = self.SMF_Class.cModel_SDSS() 
            self.SMF_Redshifts += Redshifts
            self.SMF_Central.append(Y_cen)
            self.SMF_Total.append(Y_t)
            self.SMF_Satellite.append(Y_sat)
            self.SMF_Central_Err.append(Y_cen_e)
            self.SMF_Total_Err.append(Y_t_e)
            self.SMF_Satellite_Err.append(Y_sat)
        if "Moustakas" in SMF:
            Mous, Redshifts = self.SMF_Class.Bernardi_SDSS_Mous()
            self.SMF_Redshifts += Redshifts
            self.SMF_Total += Mous
        if "Davidzon17" in SMF:
            DavDat, Redshifts = self.SMF_Class.Davidzon_17()
            self.SMF_Redshifts += Redshifts
            self.SMF_Total += DavDat
        if "Davidzon17_corr" in SMF:
            DavDat, Redshifts = self.SMF_Class.Davidzon_17(Corr = True)
            self.SMF_Redshifts += Redshifts
            self.SMF_Total += DavDat

        
        #for each redshift step of SMF data get the HMF and SHMF
        for z in self.SMF_Redshifts:
            if HMF_central == "HMF_Collosus":
                self.CHMF.append(self.HMF_Class.hmf_vir(z))
            if HMF_sub == "STEEL":
                self.SHMF.append(self.HMF_Class.SHMF_STEEL(z))
        
        if SMHM == "Moster": 
            #Make a Parameter Spaces to search
            for i0 in np.arange(self.Bounds[0][0], self.Bounds[0][1], (self.Bounds[0][1] - self.Bounds[0][0])/10):
                for i1 in np.arange(self.Bounds[2][0], self.Bounds[2][1], (self.Bounds[2][1] - self.Bounds[2][0])/10):
                    for i2 in np.arange(self.Bounds[4][0], self.Bounds[4][1], (self.Bounds[4][1] - self.Bounds[4][0])/10):
                        for i3 in np.arange(self.Bounds[6][0], self.Bounds[6][1], (self.Bounds[6][1] - self.Bounds[6][0])/10):
                            self.Low_z.append([i0, i1, i2, i3, self.Parameters[8]])
            for i0 in np.arange(self.Bounds[1][0], self.Bounds[1][1], (self.Bounds[1][1] - self.Bounds[1][0])/10):
                for i1 in np.arange(self.Bounds[3][0], self.Bounds[3][1], (self.Bounds[3][1] - self.Bounds[3][0])/10):
                    for i2 in np.arange(self.Bounds[5][0], self.Bounds[5][1], (self.Bounds[5][1] - self.Bounds[5][0])/10):
                        for i3 in np.arange(self.Bounds[7][0], self.Bounds[7][1], (self.Bounds[7][1] - self.Bounds[7][0])/10):
                            self.High_z.append([i0, i1, i2, i3])
                
    
def DM_to_SM(SMF_X, SMF_Bin, Halo_MR, HMF_Bin, HMF, Paramaters, Redshift, SMHM_Model, N = 1000, Pairwise = False, GPU = False):
    """   
    Args:
        SMF_X: Stellar Mass Function Mass Range log10[$M_\odot$]
        HMF: Halo Mass Function Weights log10[$\Phi$ Mpc^{-3} h^3]
        Halo_MR: Halo Mass Function Mass Range log10[$M_\odot$ h^{-1}]
        HMF_Bin: Binwidth of Halo_MR
        SMF_Bin: Binwidth of SMF_X
        Parameters: Dictonary of thing to pass to DarkMatterToStellarMass see afformentioned for details
        Redshift: z
        N: number of times to use
        UseAlt: Bool To switch to other Alt DM_to_SM
    Returns:
        SMF_X: Stellar Mass Function Mass Range log10[$M_\odot$], SMF numberdensties Phi [Mpc^-3] 
    """

    DM_In = np.repeat(Halo_MR - np.log10(h), N) #log Mh [Msun]
    Wt = np.repeat(np.divide(HMF*h_3*HMF_Bin, N), N) #Phi/N [Mpc^-3]
    if Pairwise:
        Redshift = np.repeat(Redshift, N)

    if GPU:
        M10, M11, SHMnorm10, SHMnorm11, beta10, beta11, gamma10, gamma11, Scatter = Paramaters
        SM = SMHM_Model(DM_In, Redshift, M10, M11, SHMnorm10, SHMnorm11, beta10, beta11, gamma10, gamma11, Scatter) #log M* [Msun]
        Scatter_Arr = np.random.normal(scale = Scatter, size = np.shape(SM))
        SM = np.log10(SM) + Scatter_Arr
    else:
        SM = SMHM_Model(DM_In, Paramaters, Redshift, Pairwise = Pairwise) #log M* [Msun]
    
    SMF_Y, Bin_Edge = np.histogram(SM, bins = np.append(SMF_X, SMF_X[-1]+SMF_Bin)-0.05, weights = Wt, density = False) #Phi [Mpc^-3], M* [Msun]
    """
    plt.plot(DM_In, SM)
    plt.savefig("./SMHM_Fitting/SMHM_Test.png")
    plt.clf()
    plt.plot(Halo_MR, np.log10(HMF))
    plt.savefig("./SMHM_Fitting/HMF_Test.png")
    plt.clf()
    plt.plot(SMF_X, np.log10(np.divide(SMF_Y, SMF_Bin)))
    plt.savefig("./SMHM_Fitting/SMF_Test.png")
    plt.clf()
    """
    
    return np.log10(np.divide(SMF_Y, SMF_Bin)) #M* [Msun], Phi [Mpc^-3] 




def MultiProcessWrapper_Lowz(Inputs):
    """
    Takes the input to multiprocessing and makes it usable for DM_to_SM then does the fit.
    Args:
        Inputs: A tuple containing the following:
            [M,N,b,g], z, HMF_Range, HMF_Bin, HMF_Numberdensitys, SMF_Range, SMF_Bin, SMF_Numberdensities, SMHM_Model
    Returns:
    List: [Goodness of Fit, [M,N,b,g]]
    
    """
    Params, z, HMR, HM_Bin, CHMF_0, SMR, SM_Bin, CSMF_0, SMHM_Model = Inputs
    Params = [Params[0], 0, Params[1], 0, Params[2], 0, Params[3], 0, Params[4]]
    SMF_From_SMHM = DM_to_SM(SMR, SM_Bin, HMR, HM_Bin, CHMF_0, Params, z, SMHM_Model)
    mask = np.logical_and(np.isfinite(SMF_From_SMHM), np.isfinite(CSMF_0))
    if len(SMF_From_SMHM[mask])/len(SMF_From_SMHM) > 0.9:
        Fit = np.sqrt(np.sum(np.power(SMF_From_SMHM[mask] - CSMF_0[mask], 2))/len(SMF_From_SMHM[mask]))
    else:
        Fit = np.inf
    return [Fit, Params]



@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64,float64,float64,float64,float64)'])
def Moster_GPU(DM, z, M10, M11, SHMnorm10, SHMnorm11, beta10, beta11, gamma10, gamma11, Scatter):
    zparameter = (z-0.1)/(z+1)
    #putting the parameters together for inclusion in the Moster 2010 equation
    M = M10 + M11*zparameter
    N = SHMnorm10 + SHMnorm11*zparameter
    b = beta10 + beta11*zparameter
    g = gamma10 + gamma11*zparameter
        
    # Moster 2010 eq2
    SM =  (10**DM) * (2*N*((((10**(DM-M))** -b) + ((10**(DM-M))** g))** -1.0))
    
    return SM


def MultiProcessWrapper_Highz(Inputs):
    """
    Takes the input to multiprocessing and makes it usable for DM_to_SM then does the fit.
    Args:
        Inputs: A tuple containing the following:
            [M,N,b,g], z, HMF_Range, HMF_Bin, HMF_Numberdensitys, SMF_Range, SMF_Bin, SMF_Numberdensities, SMHM_Model
    Returns:
    List: [Goodness of Fit, [M,N,b,g]]
    
    """
    Params_lz, Params_hz, z, HMR, HM_Bin, HMF_wts, SMR, SM_Bin, SMF, SMHM_Model = Inputs
    Params = [Params_lz[0], Params_hz[0], Params_lz[2], Params_hz[1], Params_lz[4], Params_hz[2], Params_lz[6], Params_hz[3], Params_lz[8]]
    Fit = 0
    
    #print(np.shape(HMR))
    #print(np.shape(SMF))
    
    for i in range(len(HMR)):
        SMF_From_SMHM = DM_to_SM(SMR, SM_Bin, HMR[i], HM_Bin, HMF_wts[i], Params, z[i], SMHM_Model, Pairwise = True, GPU = False)
        mask = np.logical_and(np.isfinite(SMF_From_SMHM), np.isfinite(SMF[i]))
        Fit += np.sqrt(np.sum(np.power(SMF_From_SMHM[mask] - SMF[i][mask], 2))/len(SMF_From_SMHM[mask]))
        #if len(SMF_From_SMHM[mask])/len(SMF_From_SMHM) > 0.5:
        #    Fit += np.sqrt(np.sum(np.power(SMF_From_SMHM[mask] - SMF[i][mask], 2))/len(SMF_From_SMHM[mask]))
        # else:
        #     Fit += np.inf
    return [Fit, Params]


if __name__ == "__main__":
    #Make FttingFuctionsCass
    FittingClass = Fitting_Functions(SMHM = "Moster", SMF = ["Bernardi16", "Davidzon17_corr"], HMF_central = "HMF_Collosus", HMF_sub = "STEEL")
    #FittingClass = Fitting_Functions(SMHM = "Moster", SMF = ["cModel", "Davidzon17"], HMF_central = "HMF_Collosus", HMF_sub = "STEEL")
    #Fit at redshift 0.1:
    z = 0.1
    index_0 = np.digitize(z, bins = FittingClass.SMF_Redshifts) - 1
    HMR, HM_min, HM_max, HM_bin = FittingClass.HMR
    CHMF_0 = FittingClass.CHMF[index_0]
    SHMF_0 = FittingClass.SHMF[index_0]
    SMR, SM_min, SM_max, SM_bin = FittingClass.SMR
    TSMF_0 = FittingClass.SMF_Total[index_0]
    CSMF_0 = FittingClass.SMF_Central[index_0]
    SSMF_0 = FittingClass.SMF_Satellite[index_0]
    SMHM_Model = FittingClass.SMHM_Model
    ParameterList = [(Params, z, HMR, HM_bin, CHMF_0, SMR, SM_bin, CSMF_0, SMHM_Model) for Params in FittingClass.Low_z]
    
    
    """
    #Test_Params = [11.95,0.032,1.61,0.54, 0.15]
    Test_Params = [11.999999999999996,  0.03199999999999999,  1.5000000000000002,  0.56, 0.15]
    print(MultiProcessWrapper_Lowz((Test_Params, z, HMR, HM_bin, CHMF_0, SMR, SM_bin, CSMF_0, SMHM_Model))) # fortesting
    Params = [Test_Params[0], 0, Test_Params[1], 0, Test_Params[2], 0, Test_Params[3], 0, Test_Params[4]]
    plt.plot(SMR, DM_to_SM(SMR, SM_bin, HMR, HM_bin, CHMF_0, Params, z, SMHM_Model))
    plt.plot(SMR, CSMF_0, label = "Cent")
    plt.plot(SMR, SSMF_0, label = "Sat")
    plt.plot(SMR, TSMF_0, label = "Total")
    plt.savefig("./SMHM_Fitting/z0SMF_Test.png")
    plt.clf()
    """    
    pool = multiprocessing.Pool(processes = 20)
    PoolReturn = pool.map(MultiProcessWrapper_Lowz, ParameterList)
    pool.close()
    #Take the RMS total slice and find index coressponding to minimum RMS
    pickle.dump(PoolReturn, open("./SMHM_Fitting/PoolReturn_lz_SerExp.pkl", 'wb'))
    RMS_Arr = [i[0] for i in PoolReturn]  
    index_min = np.argmin(RMS_Arr)
    #print minimum result
    print(PoolReturn[index_min])
    Params_lz = PoolReturn[index_min][1]
    #Params = [Params[0], 0, Params[1], 0, Params[2], 0, Params[3], 0, Params[4]]
    plt.plot(SMR, DM_to_SM(SMR, SM_bin, HMR, HM_bin, CHMF_0, Params_lz, z, SMHM_Model), label = "Fit")
    plt.plot(SMR, CSMF_0, label = "Cent")
    plt.plot(SMR, SSMF_0, label = "Sat")
    plt.plot(SMR, TSMF_0, label = "Total")
    plt.savefig("./SMHM_Fitting/z0SMF_SerExp.png")
    plt.clf()
    #"""
    
    #fit at redshift > 0.1
    SMHM_Model = FittingClass.SMHM_Model
    HaloMasses = [[] for i in range(len(FittingClass.SMF_Redshifts))]
    HaloWts = [[] for i in range(len(FittingClass.SMF_Redshifts))]
    HaloRed = [[] for i in range(len(FittingClass.SMF_Redshifts))]
    
    #make repeating list of DM length of the redshift steps in steel
    for i, z_ in enumerate(FittingClass.SMF_Redshifts):
        #Centrals
        HaloMasses[i] += list(HMR)
        HaloWts[i] += list(FittingClass.CHMF[i])
        HaloRed[i] += [z_ for temp in range(len(FittingClass.CHMF[i]))]
        #Subhaloes
        for j, z_sub_inf in enumerate(FittingClass.SHMF[0][2]):
            HaloMasses[i] += list(FittingClass.SHMF[i][0])
            HaloWts[i] += list(FittingClass.SHMF[i][1][j])
            HaloRed[i] += [z_sub_inf for temp in range(len(FittingClass.SHMF[i][1][j]))]
    HaloMasses = np.array(HaloMasses)
    HaloWts = np.array(HaloWts)
    HaloRed = np.array(HaloRed)
    
    
    
    
    """
    #Test_Params = [11.95,0.032,1.61,0.54, 0.15]
    Params_lz = Params
    Test_Params = [0.4, -0.02,  -0.6,  -0.1]
    #Test_Params = [0,0,0,0]
    print(MultiProcessWrapper_Highz((Params_lz, Test_Params, HaloRed, HaloMasses, HM_bin, HaloWts, SMR, SM_bin, FittingClass.SMF_Total, Moster_GPU))) # fortesting
    Params = [Params_lz[0], Test_Params[0], Params_lz[1], Test_Params[1], Params_lz[2], Test_Params[2], Params_lz[3], Test_Params[3], Params_lz[4]]
    f, SubPlots = plt.subplots(3, 3, figsize = (12,12), sharex = True ,sharey = True)
    k = 0
    for i in range(3):
        for j in range(3):         
            SubPlots[i][j].plot(SMR, DM_to_SM(SMR, SM_bin, HaloMasses[k], HM_bin, HaloWts[k], Params, HaloRed[k], SMHM_Model, Pairwise = True), label = "Fit")
            SubPlots[i][j].plot(SMR, FittingClass.SMF_Total[k], label = "Total")
            k+=1
    SubPlots[0][0].legend()
    plt.savefig("./SMHM_Fitting/HighzSMF_Test.png")
    plt.clf()
    """
    ParameterList = [(Params_lz, Params_hz, HaloRed, HaloMasses, HM_bin, HaloWts, SMR, SM_bin, FittingClass.SMF_Total, SMHM_Model) for Params_hz in FittingClass.High_z]
    pool = multiprocessing.Pool(processes = 20)
    PoolReturn = pool.map(MultiProcessWrapper_Highz, ParameterList)
    pool.close()
    #Take the RMS total slice and find index coressponding to minimum RMS
    pickle.dump(PoolReturn, open("./SMHM_Fitting/PoolReturn_hz_SerExp.pkl", 'wb'))
    RMS_Arr = [i[0] for i in PoolReturn]  
    index_min = np.argmin(RMS_Arr)
    #print minimum result
    print(PoolReturn[index_min])
    Params = PoolReturn[index_min][1]
    f, SubPlots = plt.subplots(3, 3, figsize = (12,12), sharex = True ,sharey = True)
    k = 0
    for i in range(3):
        for j in range(3):         
            SubPlots[i][j].plot(SMR, DM_to_SM(SMR, SM_bin, HaloMasses[k], HM_bin, HaloWts[k], Params, HaloRed[k], SMHM_Model, Pairwise = True), label = "Fit")
            SubPlots[i][j].plot(SMR, FittingClass.SMF_Total[k], label = "Total")
            k+=1
    plt.savefig("./SMHM_Fitting/HighzSMF_Ser_Exp.png")
    plt.clf()
    #"""
    
    
    #Make FttingFuctionsCass
    FittingClass = Fitting_Functions(SMHM = "Moster", SMF = ["cModel", "Davidzon17"], HMF_central = "HMF_Collosus", HMF_sub = "STEEL")
    #Fit at redshift 0.1:
    z = 0.1
    index_0 = np.digitize(z, bins = FittingClass.SMF_Redshifts) - 1
    HMR, HM_min, HM_max, HM_bin = FittingClass.HMR
    CHMF_0 = FittingClass.CHMF[index_0]
    SHMF_0 = FittingClass.SHMF[index_0]
    SMR, SM_min, SM_max, SM_bin = FittingClass.SMR
    TSMF_0 = FittingClass.SMF_Total[index_0]
    CSMF_0 = FittingClass.SMF_Central[index_0]
    SSMF_0 = FittingClass.SMF_Satellite[index_0]
    SMHM_Model = FittingClass.SMHM_Model
    ParameterList = [(Params, z, HMR, HM_bin, CHMF_0, SMR, SM_bin, CSMF_0, SMHM_Model) for Params in FittingClass.Low_z]
    
    
    """
    #Test_Params = [11.95,0.032,1.61,0.54, 0.15]
    Test_Params = [11.999999999999996,  0.03199999999999999,  1.5000000000000002,  0.56, 0.15]
    print(MultiProcessWrapper_Lowz((Test_Params, z, HMR, HM_bin, CHMF_0, SMR, SM_bin, CSMF_0, SMHM_Model))) # fortesting
    Params = [Test_Params[0], 0, Test_Params[1], 0, Test_Params[2], 0, Test_Params[3], 0, Test_Params[4]]
    plt.plot(SMR, DM_to_SM(SMR, SM_bin, HMR, HM_bin, CHMF_0, Params, z, SMHM_Model))
    plt.plot(SMR, CSMF_0, label = "Cent")
    plt.plot(SMR, SSMF_0, label = "Sat")
    plt.plot(SMR, TSMF_0, label = "Total")
    plt.savefig("./SMHM_Fitting/z0SMF_Test.png")
    plt.clf()
    """    
    pool = multiprocessing.Pool(processes = 20)
    PoolReturn = pool.map(MultiProcessWrapper_Lowz, ParameterList)
    pool.close()
    #Take the RMS total slice and find index coressponding to minimum RMS
    pickle.dump(PoolReturn, open("./SMHM_Fitting/PoolReturn_lz_cMod.pkl", 'wb'))
    RMS_Arr = [i[0] for i in PoolReturn]  
    index_min = np.argmin(RMS_Arr)
    #print minimum result
    print(PoolReturn[index_min])
    Params_lz = PoolReturn[index_min][1]
    #Params = [Params[0], 0, Params[1], 0, Params[2], 0, Params[3], 0, Params[4]]
    plt.plot(SMR, DM_to_SM(SMR, SM_bin, HMR, HM_bin, CHMF_0, Params_lz, z, SMHM_Model), label = "Fit")
    plt.plot(SMR, CSMF_0, label = "Cent")
    plt.plot(SMR, SSMF_0, label = "Sat")
    plt.plot(SMR, TSMF_0, label = "Total")
    plt.savefig("./SMHM_Fitting/z0SMF_cMod.png")
    plt.clf()
    #"""
    
    #fit at redshift > 0.1
    SMHM_Model = FittingClass.SMHM_Model
    HaloMasses = [[] for i in range(len(FittingClass.SMF_Redshifts))]
    HaloWts = [[] for i in range(len(FittingClass.SMF_Redshifts))]
    HaloRed = [[] for i in range(len(FittingClass.SMF_Redshifts))]
    
    #make repeating list of DM length of the redshift steps in steel
    for i, z_ in enumerate(FittingClass.SMF_Redshifts):
        #Centrals
        HaloMasses[i] += list(HMR)
        HaloWts[i] += list(FittingClass.CHMF[i])
        HaloRed[i] += [z_ for temp in range(len(FittingClass.CHMF[i]))]
        #Subhaloes
        for j, z_sub_inf in enumerate(FittingClass.SHMF[0][2]):
            HaloMasses[i] += list(FittingClass.SHMF[i][0])
            HaloWts[i] += list(FittingClass.SHMF[i][1][j])
            HaloRed[i] += [z_sub_inf for temp in range(len(FittingClass.SHMF[i][1][j]))]
    HaloMasses = np.array(HaloMasses)
    HaloWts = np.array(HaloWts)
    HaloRed = np.array(HaloRed)
    
    
    
    
    """
    #Test_Params = [11.95,0.032,1.61,0.54, 0.15]
    Params_lz = Test_Params
    Test_Params = [0.4, -0.02,  -0.6,  -0.1]
    #Test_Params = [0,0,0,0]
    print(MultiProcessWrapper_Highz((Params_lz, Test_Params, HaloRed, HaloMasses, HM_bin, HaloWts, SMR, SM_bin, FittingClass.SMF_Total, SMHM_Model))) # fortesting
    Params = [Params_lz[0], Test_Params[0], Params_lz[1], Test_Params[1], Params_lz[2], Test_Params[2], Params_lz[3], Test_Params[3], Params_lz[4]]
    f, SubPlots = plt.subplots(3, 3, figsize = (12,12), sharex = True ,sharey = True)
    k = 0
    for i in range(3):
        for j in range(3):         
            SubPlots[i][j].plot(SMR, DM_to_SM(SMR, SM_bin, HaloMasses[k], HM_bin, HaloWts[k], Params, HaloRed[k], SMHM_Model, Pairwise = True), label = "Fit")
            SubPlots[i][j].plot(SMR, FittingClass.SMF_Total[k], label = "Total")
            k+=1
    SubPlots[0][0].legend()
    plt.savefig("./SMHM_Fitting/HighzSMF_Test.png")
    plt.clf()
    """
    ParameterList = [(Params_lz, Params_hz, HaloRed, HaloMasses, HM_bin, HaloWts, SMR, SM_bin, FittingClass.SMF_Total, SMHM_Model) for Params_hz in FittingClass.High_z]
    pool = multiprocessing.Pool(processes = 20)
    PoolReturn = pool.map(MultiProcessWrapper_Highz, ParameterList)
    pool.close()
    #Take the RMS total slice and find index coressponding to minimum RMS
    pickle.dump(PoolReturn, open("./SMHM_Fitting/PoolReturn_hz_cMod.pkl", 'wb'))
    RMS_Arr = [i[0] for i in PoolReturn]  
    index_min = np.argmin(RMS_Arr)
    #print minimum result
    print(PoolReturn[index_min])
    Params = PoolReturn[index_min][1]
    f, SubPlots = plt.subplots(3, 3, figsize = (12,12), sharex = True ,sharey = True)
    k = 0
    for i in range(3):
        for j in range(3):         
            SubPlots[i][j].plot(SMR, DM_to_SM(SMR, SM_bin, HaloMasses[k], HM_bin, HaloWts[k], Params, HaloRed[k], SMHM_Model, Pairwise = True), label = "Fit")
            SubPlots[i][j].plot(SMR, FittingClass.SMF_Total[k], label = "Total")
            k+=1
    plt.savefig("./SMHM_Fitting/HighzSMF_cMod.png")
    plt.clf()
    #"""