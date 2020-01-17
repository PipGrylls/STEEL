with open ('../default.conf') as conf_file:
    input_params = conf_file.readlines()
for i in range(0, len(input_params)):
    if 'path_to_STEEL' in input_params[i]:
        for j in range(0, len(input_params[i])):
            if input_params[i][len(input_params[i])-1-j] == "\'":
                if input_params[i][len(input_params[i])-2-j] == '/':
                    exec(input_params[i])
                    break
                else:
                    exec( input_params[i][0:len(input_params[i])-1-j] + "/\'")
                    break
        break

import os
import sys
AbsPath = str(__file__)[:-len("/SMHM_Fit_MCMC.py")]+"/.."
sys.path.append(AbsPath)
sys.path.append(path_to_STEEL+'Functions') #Hao
sys.path.append(path_to_STEEL+'Scripts/Plots') #Hao
import numpy as np
import pandas as pd
import pickle
import emcee
import corner
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import Functions as F
from colossus.cosmology import cosmology
import SDSS_Plots


"""
Pay attention to the following parameters for the DM to SM conversion
"""
Override =\
{\
'M10':12.0,\
'SHMnorm10':0.032,\
'beta10':1.5,\
'gamma10':0.56,\
'M11':0.6,\
'SHMnorm11':-0.014,\
'beta11':-2,\
'gamma11':0.08\
}
AbnMtch =\
{\
'Behroozi13': False,\
'Behroozi18': False,\
'B18c':False,\
'B18t':False,\
'G18':False,\
'G18_notSE':False,\
'G19_SE':True,\
'G19_cMod':False,\
'Lorenzo18':False,\
'Moster': True,\
'Moster10': False,\
'Illustris': False,\
'z_Evo':True,\
'Scatter': 0.15,\
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
Parameters = \
{\
'AbnMtch' : AbnMtch,\
'AltDynamicalTime': 1,\
'NormRnd': 0.5,\
'SFR_Model': 'G19_DPL',\
'PreProcessing': False,\
'AltDynamicalTimeB': False\
}


HMF_fun = F.Make_HMF_Interp()
cosmology.setCosmology("planck15")
Cosmo =cosmology.getCurrent()
h = Cosmo.h
h_3 = h*h*h


def Moster_SMHM(DM, Params, Pairwise=True): #SMHM
    M10, M11, SHMnorm10, SHMnorm11, beta10, beta11, gamma10, gamma11, Scatter = Params
    M = M10 #+ M11*zparameter
    N = SHMnorm10 #+ SHMnorm11*zparameter
    b = beta10 #+ beta11*zparameter
    g = gamma10 #+ gamma11*zparameter

    SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
    
    Scatter_Arr = np.random.normal(scale = Scatter, size = np.shape(SM))
    return( np.log10(SM) + Scatter_Arr)


def identify_idx_z(z_desired, z):
    if (z_desired == 0.):
        z_idx=0
    elif (z_desired > 0. and z_desired < np.max(z)):
        for i in range(0,z.size):
            if (z[i]>z_desired):
                z_idx=i
                break
        z2=z[z_idx]; z1=z[z_idx-1]
        diff1=np.abs(z1-z_desired)
        diff2=np.abs(z2-z_desired)
        if (diff1<=diff2):
            z_idx=z_idx-1
    #elif (z_desired == 6.):
        #z_idx=z.size-1
    elif (z_desired > np.max(z)):
        Err_msg = 'The desired value {} in z_desired is larger than the maximum value in z!\nCannot identify z index!'.format(z_desired)
        sys.exit(Err_msg)
    return z_idx


class HaloMassFunction:
    
    def __init__(self):
        self.HMR_L = 10.
        self.HMR_U = 16.
        self.HMR_BW = 0.1
        self.HaloMassRange = np.arange(self.HMR_L, self.HMR_U , self.HMR_BW) #log10 Mvir h-1 Msun
        self.HaloMassFunction = F.Make_HMF_Interp() #loads the HMF/COLOSSUS hmf from STEEL
        
    def ReturnDefaultMassRange(self):
        return self.HaloMassRange, self.HMR_L, self.HMR_U , self.HMR_BW


class StellarMassFunction:

    def __init__(self):
        self.SMR_L = 9.
        self.SMR_U = 12.5
        self.SMR_BW = 0.1
        self.StellarMassRange = np.arange(self.SMR_L, self.SMR_U , self.SMR_BW) #log10 Mvir h-1 Msun
        
    def ReturnDefaultMassRange(self):
        return self.StellarMassRange, self.SMR_L, self.SMR_U , self.SMR_BW
        
    def Leja19(self, logm, z0=np.arange(0.2,3.1,0.2)):
        #logm = self.StellarMassRange[:,None]
        """
        logm and z0 array-like please
        z0 should be included between 0.2 and 3
        """
        # Continuity model median parameters + 1-sigma uncertainties.
        pars = {'logphi1': [-2.44, -3.08, -4.14],
        'logphi1_err': [0.02, 0.03, 0.1],
        'logphi2': [-2.89, -3.29, -3.51],
        'logphi2_err': [0.04, 0.03, 0.03],
        'logmstar': [10.79,10.88,10.84],
        'logmstar_err': [0.02, 0.02, 0.04],
        'alpha1': [-0.28],
        'alpha1_err': [0.07],
        'alpha2': [-1.48],
        'alpha2_err': [0.1]}
        # Draw samples from posterior assuming independent Gaussian uncertainties.
        # Then convert to mass function at `z=z0`.
        draws = {}
        ndraw = 1000 #increase for better quality, 1000 recommended by Leja et  al. 2019
        PHI = np.zeros((z0.size, logm.size))
        PHI_interp = []
        for j in range(0, z0.size):
            for par in ['logphi1', 'logphi2', 'logmstar', 'alpha1', 'alpha2']:
                samp = np.array([np.random.normal(median,scale=err,size=ndraw) for median, err in zip(pars[par], pars[par+'_err'])])
                if par in ['logphi1', 'logphi2', 'logmstar']:
                    draws[par] = F.parameter_at_z0(samp,z0[j])
                else:
                    draws[par] = samp.squeeze()
            # Generate Schechter functions.
            phi1 = F.schechter(logm, draws['logphi1'], # primary component
            draws['logmstar'], draws['alpha1'])
            phi2 = F.schechter(logm, draws['logphi2'], # secondary component
            draws['logmstar'], draws['alpha2'])
            phi = phi1 + phi2 # combined mass function
            # Compute median and 1-sigma uncertainties as a function of mass.
            #phi_50, phi_84, phi_16 = np.percentile(phi, [50, 84, 16], axis=1)
            for i in range(0,logm.size):
                PHI[j,i] = np.mean(phi[i,:])
            PHI_interp.append(interp1d(logm[:,0], PHI[j,:], bounds_error =  False))
        #return PHI, z0 #array-like [phi/Mpc^-3/dex]
        return PHI_interp, z0


class HaloMassToStellarMass:
    
    def Moster(self, DM, Input, z, Pairwise=True): #SMHM
        M10, M11, SHMnorm10, SHMnorm11, beta10, beta11, gamma10, gamma11, Scatter = Input
        zparameter = np.divide(z-0.1, z+1)
        M = M10 #+ M11*zparameter
        N = SHMnorm10 #+ SHMnorm11*zparameter
        b = beta10 #+ beta11*zparameter
        g = gamma10 #+ gamma11*zparameter
        
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
        Bounds = [[11,13], [0,2], [0.02, 0.04], [-0.03, -0.01], [1.2, 1.8], [-0.9, -0.5], [0.5, 0.7], [-0.2, 0.2], [0.05, 0.2]]
        return self.Moster, Inputs, Bounds


class Fitting_Functions:

    def __init__(self, SMHM="Moster", SMF="Leja19", z_des=0):
        self.SMF_Class = StellarMassFunction()
        self.HMF_Class = HaloMassFunction()
        self.SMHM_Class = HaloMassToStellarMass()
        
        self.SMF_Redshifts = []
        self.SMF_Total = []
        self.Low_z = []
        self.High_z = []
        
        if "Leja19" in SMF:
            HM_min = 11.0; HM_max = 16.6; HM_bin=0.1
            HM_range = np.arange(HM_min+np.log10(h), HM_max+np.log10(h), HM_bin)
            Redshifts, HM_Range = F.Get_HM_History(HM_range, HM_min, HM_max, HM_bin)
            z_idx = identify_idx_z(z_des, Redshifts)
            self.z = Redshifts[z_idx]
            SM_range = F.DarkMatterToStellarMass(HM_Range, self.z, Parameters, ScatterOn = True, Scatter = 0.15, Pairwise = True)
            SM_idx_min=0; SM_idx_max=0
            for i in range(0, SM_range[z_idx,:].size):
                if SM_range[z_idx,i]>=8. and SM_idx_min==0:
                    SM_idx_min=i
                if SM_range[z_idx,i]>=12. and SM_idx_max==0:
                    SM_idx_max=i-1
            #self.HMR = HM_Range[z_idx,:], HM_min, HM_max, HM_bin
            #self.SMR = SM_range[z_idx,:], np.min(SM_range), np.max(SM_range), np.abs(np.mean(SM_range[0]))
            self.HMR = HM_Range[z_idx,SM_idx_min:SM_idx_max], HM_min, HM_max, HM_bin
            self.SMR = SM_range[z_idx,SM_idx_min:SM_idx_max], np.min(SM_range), np.max(SM_range), np.abs(np.mean(SM_range[0]))
            """z_idx_min=0; z_idx_max=0
            for i in range(0,self.z.size):
                if self.z[i]>=0.2 and z_idx_min==0:
                    z_idx_min = i#; print(z_idx_min)
                if self.z[i]>3. and z_idx_max==0:
                    z_idx_max = i-1#; print(z_idx_max)
            Leja_SMF, Redshifts = self.SMF_Class.Leja19(logm=SM_range[z_idx_min:z_idx_max,:][:, None],  z0=self.z[z_idx_min:z_idx_max])"""
            Leja_SMF, Redshifts = self.SMF_Class.Leja19(logm=SM_range[z_idx,:][:,None], z0=np.array([self.z]))
            self.SMF_Redshifts.append(self.z)
            self.SMF_Total.append(Leja_SMF)
            
        if SMHM == "Moster":
            self.SMHM_Model, self.Parameters, self.Bounds = self.SMHM_Class.ReturnMosterFit()
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
                            
                            
def DM_to_SM(SMF_X, SMF_Bin, HMR, HMF_bin, Params, Redshift, SMHM_Model):
    """
    Args:
        SMF_X: Stellar Mass Function Mass Range log10[$M_\odot$]
        HMF: Halo Mass Function Weights [$\Phi$ Mpc^{-3} h^3]
        HMR: Halo Mass Function Mass Range log10[$M_\odot$ h^{-1}]
        HMF_bin: Binwidth of HMR
        SMF_bin: Binwidth of SMF_X
        Params: Dictonary of thing to pass to DarkMatterToStellarMass see afformentioned for details
        Redshift: z
        N: number of times to use
        UseAlt: Bool To switch to other Alt DM_to_SM
    Returns:
        SMF_X: Stellar Mass Function Mass Range log10[$M_\odot$], SMF numberdensties Phi [Mpc^-3]
    """
    DM = HMR - np.log10(h)
    #Wt = np.array(HMF) *h_3*HMF_bin
    SM = SMHM_Model(DM, Params, Redshift, Pairwise = True) #log M* [Msun]
    bins = np.append(SMF_X, SMF_X[-1]+SMF_Bin)-(SMF_Bin/2)
    SMF_Y, Bin_Edge = np.histogram(SM, density = False) #Phi [Mpc^-3], M* [Msun]
    return SMF_X, np.log10(np.divide(SMF_Y, SMF_Bin)) #M* [Msun], Phi [Mpc^-3]


def ln_Lkl(theta, Params, z, HMR, HM_bin, SMR, SM_bin, SMFt, SMHM_Model):
    #z, HMR, HM_bin, SMR, SM_bin, SMFt, SMHM_Model = Input
    Params = [Params[0], 0, Params[1], 0, Params[2], 0, Params[3], 0, Params[4]]
    theta = [theta[0], 0, theta[1], 0, theta[2], 0, theta[3], 0, 0.15]
    #SMF_from_SMHM = DM_to_SM(SMR, SM_bin, HMR, HM_bin, Params, z, SMHM_Model)
    #SMF_from_Model = SMFt
    #Lkl = -0.5* np.sum((SMF_from_SMHM[0][:]-SMF_from_Model[0][0](SMR[:]))**2)
    SMHM_fid = DM_to_SM(SMR, SM_bin, HMR, HM_bin, Params, z, SMHM_Model)
    SMHM_obs = Moster_SMHM(HMR, theta, Pairwise=True)
    if True in np.isnan(SMHM_obs):
        Lkl = -np.inf
    else:
        Lkl = -0.5* np.sum((SMHM_fid[0][:]-SMHM_obs[:])**2)
    #print(SMHM_obs)
    #print('{} {} {} {} {}'.format(Lkl, theta[0], theta[1], theta[2], theta[3]))
    """plt.ion(); plt.figure(1)
    plt.plot(HMR, np.log10(SMHM_fid[0][:]))
    plt.plot(HMR, np.log10(SMHM_obs[:]), ',')
    plt.show()"""
    return Lkl



if __name__ == "__main__":
    
    z = 1.
    FittingClass = Fitting_Functions(SMHM="Moster", SMF="Leja19", z_des=z)
    
    #z = np.array(FittingClass.SMF_Redshifts)[0]
    SMR, SM_min, SM_max, SM_bin = FittingClass.SMR
    HMR, HM_min, HM_max, HM_bin = FittingClass.HMR
    SMHM_Model = FittingClass.SMHM_Model
    Params = 11.91,0.029,2.09,0.64, 0.15 #cmodel
    Input = [Params, z, HMR, HM_bin, SMR, SM_bin, FittingClass.SMF_Total, SMHM_Model]
    ndim, nwalkers = 4, 16
    #pos = [[0.6,-0.014,-0.7,0.03] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    pos = [[11.91,0.029,2.09,0.64] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_Lkl, args=Input, threads=20)
    sampler.run_mcmc(pos, 500000)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
    Mz, Nz, bz, gz = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
    #pickle.dump(samples, open("./Data/Model/Output/Other/SMHM_Fitting/MCMC_Chain_hz.pkl", 'wb'))
    print("Mz:",Mz)
    print("Nz:",Nz)
    print("bz:",bz)
    print("gz:",gz)
    fig = corner.corner(samples, labels=[r"$M_z$", r"$N_z$", r"$\beta_z$", r"$\gamma_z$"], truths = [Mz[0], Nz[0], bz[0], gz[0]],\
                        color = "C0", truth_color = "k", smooth = True, quantiles=[0.16, 0.84],\
                        show_titles = True, title_fmt= ".3f")
    fig.savefig("MCMC_Leja.png")
    fig.clf()






