"""
STEEL
Author: Philip Grylls
If this code or its output is used please cite Grylls+2019 "A statistical semi-empirical model: satellite galaxies in groups and clusters" and any relavent followup papers by the authors.
If you wish to devlop the code please contact pipgryllsastro"at"gmail.com or if unavalible F.Shankar"at"soton.ac.uk
"""

import time
import os
import numpy as np
from fast_histogram import histogram1d, histogram2d
import matplotlib as mpl
mpl.use('agg')
import hmf
from Functions import Functions as F
import multiprocessing
from numba import jit
from colossus.cosmology import cosmology
from colossus.halo.mass_defs import changeMassDefinition as CMD
from colossus.halo.mass_defs import pseudoEvolve as PE
from colossus.lss import mass_function
from colossus.halo.concentration import concentration as get_c
from colossus.halo.mass_so import M_to_R
from halotools import empirical_models
from astropy.cosmology import Planck15 as Cosmo_AstroPy
plt = mpl.pyplot
T1 = time.time()
cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()
h = Cosmo.h
h_3 = h*h*h
HMF_fun = F.Make_HMF_Interp()


HighRes = False # Set to True for better HM/SM resolution, but takes MUCH longer

# Cuts in Satellite Mass
SM_Cuts = [9, 9.5, 10, 10.5, 11, 11.45] #[9,10,11]
# When using Abundance matching do N realisations to capture upscatter effects
N = 5

#Abundance Matching Parameters
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
'G19_SE':False,\
'G19_cMod':False,\
'Lorenzo18':False,\
'Moster': False,\
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
'g_PFT4': False\
}

Paramaters_Glob = \
{\
'AbnMtch' : AbnMtch,\
'AltDynamicalTime': 1,\
'NormRnd': 0.5,\
'SFR_Model': 'CE',\
'PreProcessing': False,\
'AltDynamicalTimeB': False\
}

# Subhalomass function parameters macc/M0
Unevolved = {\
'gamma' : 0.22,\
'alpha' : -0.91,\
'beta' : 6,\
'omega' : 3,\
'a' : 1,\
}

# HaloMass Limits and Bins
AnalyticHaloMass_min = 11.0
AnalyticHaloMass_max = 16.6

if HighRes:
    AnalyticHaloBin = 0.05
else:
    AnalyticHaloBin = 0.1

AHB_2 = AnalyticHaloBin*AnalyticHaloBin
AnalyticHaloMass = np.arange(AnalyticHaloMass_min + np.log10(h), AnalyticHaloMass_max + np.log10(h), AnalyticHaloBin)
# Units are Mvir h-1

# This is the Halomass groWth history
# Generates redshfit steps that are small enough to avoid systematics.
z, AvaHaloMass_wz = F.Get_HM_History(AnalyticHaloMass, AnalyticHaloMass_min, AnalyticHaloMass_max, AnalyticHaloBin)
AvaHaloMass = AvaHaloMass_wz[:, 1:]

# Account for central bin shrinking
AvaHaloMassBins = AvaHaloMass[:,1:] - AvaHaloMass[:,:-1]
AvaHaloMassBins = np.concatenate((AvaHaloMassBins, np.array([AvaHaloMassBins[:,-1]]).T), axis = 1)

# Arrays for tracing the time and indexing efficently the time to z = 0
Times = F.RedshiftToTimeArr(z)
Time_To_0 = Times[0] - Times
#=========================Creating SubHalos=====================================

"""Creating subhalo mass arrays going back in time"""
# range of satilite masses (Slightly lower max and much lower min than AnaHaloMass)

if HighRes:
    Min_Corr = -3 # For Continuity satellites
else:
    Min_Corr = -1
SatHaloMass = np.arange(AnalyticHaloMass_min + Min_Corr + np.log10(h), AnalyticHaloMass_max - 0.1 + np.log10(h), AnalyticHaloBin)
SHM_min, SHM_max = np.min(SatHaloMass), np.max(SatHaloMass)
# Units are Mvir h-1

"""for each array create SHMF"""
# Shapes
a, b = np.shape(AvaHaloMass)
c = np.shape(SatHaloMass)[0]
SubHaloFile = "SHMFs_Entering_{}{}{}{}{}{}{}.npy".format(AnalyticHaloMass_min+Min_Corr, AnalyticHaloMass_max, AnalyticHaloBin, h, a, b, c)
if SubHaloFile in os.listdir(path="./Data/Model/Input/"):
    SHMFs_Entering = np.load("./Data/Model/Input/"+SubHaloFile)
else:
    # Make  m_M to FOR uSHMF from Jing et al
    m_M = np.array([[SatHaloMass - AvaHaloMass[i][j] for j in range(b)] for i in range(a)])
    # Create SHMF arrays (no redshift evolution)
    SHMFs = np.array([[F.dn_dlnX(Unevolved, np.power(10, m_M[i][j])) for j in range(b)] for i in range(a)])
    # Calculate the number density of halos that fall in at each timestep
    SHMFs_Entering = np.array([[SHMFs[:, i][j] - SHMFs[:, i][j+1] for i in range(b)] for j in range(a-1)])

    np.save("./Data/Model/Input/"+"SHMFs_Entering_{}{}{}{}{}{}{}".format(AnalyticHaloMass_min+Min_Corr, AnalyticHaloMass_max, AnalyticHaloBin, h, a, b, c), SHMFs_Entering)

#===============Calculating Surviving Subhalos and Galaxies=====================
# for each accreted halo calculate if it survives to z = 0 given tdyn
# Abundance match the galaxy in and count the number of satilites above SM_Cut

def OneRealization(Factor_Stripping_SF, ParamOverRide = False, AltParam = None):

    # For the high redhsift fits
    if ParamOverRide:
        Paramaters = AltParam
    else:
        Paramaters = Paramaters_Glob
        print("Starting:", Factor_Stripping_SF)

    """Runs the Code for one set of parameters"""

    ######################
    # Parameter Management

    # Split the Running Paramters here for clarity later
    if Factor_Stripping_SF[0][-4:] == "_Alt":
        Factor = float(Factor_Stripping_SF[0][:-4])
        Paramaters['AltDynamicalTimeB'] = True
    else:
        Factor = Factor_Stripping_SF[0]

    Stripping = Factor_Stripping_SF[1]
    SF = Factor_Stripping_SF[2]
    Stripping_DM = False #Future use
    AbnMtch['z_Evo'] = Factor_Stripping_SF[3]
    #Pass the alterd dynamical time into the dictonary for function input
    Paramaters["AltDynamicalTime"] = float(Factor)
    #Switch between CE SFR and observed SFR
    if Factor_Stripping_SF[4][-3:] == "_PP":
        Paramaters['PreProcessing'] = True
        Paramaters['SFR_Model'] = Factor_Stripping_SF[4][:-3]
    else:
        Paramaters['SFR_Model'] = Factor_Stripping_SF[4]
    #Choice of Abundance matching
    AbnMtch[Factor_Stripping_SF[5]] = True
    if "PFT" in Factor_Stripping_SF[5]:
        AbnMtch["PFT"] = True
    ###################

    # ///////////
    # Array Prep

    # Data output arrays that are saved into the folders created above
    # Saving usSHMF's at each redshift step
    SurvivingSubhalos = np.full((a, c), 0.)
    SurvivingSubhalos_Stripped = np.full((a, c-1), 0.)
    SurvivingSubhalos_ByParent = np.full((a, b, c), 0.)
    SurvivingSubhalos_Stripped_ByParent = np.full((a, b, c-1), 0.)
    SurvivingSubhalos_z_z = np.full((a,a,c), 0.)

    # For saving surviving satellite galaxies
    if HighRes:
        SatBin = 0.05
        Surviving_Sat_SMF_MassRange = np.arange(6.5, 13.1, SatBin)#For Continuity
        SatM_min, SatM_max, SatM_len = 6.5, 13.0, np.size(Surviving_Sat_SMF_MassRange)-1
    else:
        SatBin = 0.1
        Surviving_Sat_SMF_MassRange = np.arange(9, 13.1, SatBin)
        SatM_min, SatM_max, SatM_len = 9.0, 13.0, np.size(Surviving_Sat_SMF_MassRange)-1

    # For the total numberdensities of each satilite mass for SMF
    Surviving_Sat_SMF_Weighting_Totals = np.zeros(np.size(Surviving_Sat_SMF_MassRange[:-1]))
    Surviving_Sat_SMF_Weighting_Totals_highz = np.zeros((a, len(Surviving_Sat_SMF_MassRange[:-1])))
    #2d array where i is parent halomass and j is Surviving_Sat_SMF_MassRange
    Surviving_Sat_SMF_Weighting = np.zeros((b, np.size(Surviving_Sat_SMF_MassRange[:-1])))
    Surviving_Sat_SMF_Weighting_highz = np.zeros( (a, b, len(Surviving_Sat_SMF_MassRange[:-1])) )
    #For saving satilite massases and associated halo/subhalo masses
    Sat_SMHM = np.zeros((a, c+1, len(Surviving_Sat_SMF_Weighting_Totals))) #redshift, subhalo, SM
    Sat_SMHM_Host = np.zeros((a, b+1, len(Surviving_Sat_SMF_Weighting_Totals))) #redshift, parent halo, SM

    #Saving sSFR for galaxies
    sSFR_Range = np.arange(-14, -8, 0.1)
    sSFR_min, sSFR_max, sSFR_len = -14, -8, np.size(sSFR_Range)-1
    Satilite_sSFR = np.zeros((len(Surviving_Sat_SMF_MassRange[:-1]), len(sSFR_Range[:-1])))
    #Saving bulk stars made per central halo per satellite mass bin
    Total_StarFormation = [[[[] for k in range(0, len(Surviving_Sat_SMF_MassRange[:-1]))] for j in range(0, b)] for i in range(0, a)]

    #Saving elliptical probabilities
    P_Elliptical = np.full((a, b), 0.)
    #saving sersic index
    Analyticalmodel_SI = np.full((a, b), 1.)
    #saving infall redshifts
    z_infall = np.full((a, len(Surviving_Sat_SMF_MassRange[:-1])), 0.0)
    #saving mergers
    Accretion_History = np.full((a, b, len(Surviving_Sat_SMF_MassRange)-1), 0.) #Array to host the subhalos merging in a given host bin for a given redshift
    Accretion_History_Halo = np.full((a, b, c), 0.)
    #saving pairfractions
    Pair_Frac = np.full((a, b, len(Surviving_Sat_SMF_MassRange)-1), 0.) #Array to host the subhalos merging in a given host bin for a given redshift
    Pair_Frac_Halo = np.full((a, b, c), 0.)

    # ////////////


    #Loop over redshift steps from high z to z = 0
    for i in range(a-2, -1, -1):
        if ParamOverRide == False:
            if i%10 == 0 or i == (a-2):
                print("Still Running:", Factor_Stripping_SF, "{}/{}".format(a-i, a))
        #This is the time to redshift 0
        TTZ0 = Time_To_0[i]
        # Loop over the AvaHaloMass
        for j in range(b):
            # Loop Over the Subhalo masses
            for k in range(c):
                # Only calculate where host is bigger than sat, nothing physical just about shape of arrays.
                if AvaHaloMass[i][j] > SatHaloMass[k]:
                    # Calculate the merger time for this bin of subhalo mass=========
                    # Masses are virial, little h cancles out so dependance unnecessary
                    Tdyf = F.DynamicalFriction(AvaHaloMass[i][j], SatHaloMass[k], z[i], Paramaters)
                    z_bin = np.digitize(Tdyf + Times[i], Times) #index for T_Merge
                    if Tdyf < TTZ0:
                        z_Merge = z[z_bin]
                        z_Merge_Bin = z_bin
                    else:
                        z_Merge_Bin = -1 #flag this galaxy as never merging in observable universe
                        z_bin = 0
                    #z_bin indexes on a/i/z[] as the time the subhalo merges========

                    #Strip the subhalos ============================================
                    if (z_bin < i):
                        #Dark Matter Stripping
                        if Stripping_DM:
                            NewHaloMass, DM_StrippingFraction = F.HaloMassLoss_w(SatHaloMass[k], AvaHaloMass[z_bin:i,j], z, z_bin, i)
                    #Save satilite halos that are remaining=========================

                    #create weightlist==============================================
                    if i != 0 and z_bin != i:
                        Arr2D = HMF_fun(AvaHaloMass[z_bin:i, j], z[z_bin:i])
                        if len(Arr2D.shape) > 1:
                            WeightList = np.diag(np.fliplr(Arr2D))*(SHMFs_Entering[i][j][k])*(AvaHaloMassBins[z_bin:i,j]*AnalyticHaloBin) # N Mpc^-3 h^3
                        else:
                            WeightList = Arr2D*(SHMFs_Entering[i][j][k])*(AvaHaloMassBins[z_bin:i,j]*AnalyticHaloBin) # N Mpc^-3 h^3
                        WeightList_SubOnly = np.full_like(z[z_bin:i], SHMFs_Entering[i][j][k]*AnalyticHaloBin) #N per central
                    else:
                        #Makes sure acretion in final redshift step is included
                        WeightList = (HMF_fun(AvaHaloMass[i, j], z[i]))*(SHMFs_Entering[i][j][k])*(AvaHaloMassBins[i,j]*AnalyticHaloBin) # N Mpc^-3 h^3
                    ###CHECK Z_bin == i ==0
                    #This creates the Unevolved Surviving Subhalo Mass Function
                    #Unstripped (Unevolved Surviving)
                    if ((Stripping_DM == False) and (Stripping or SF) == False):
                        Bin = k
                        ix = [np.arange(z_bin, i), np.full_like(np.arange(z_bin, i), Bin)]
                        SurvivingSubhalos[ix] = SurvivingSubhalos[ix] + WeightList/AnalyticHaloBin # N Mpc^-3 h^3 dex^-1
                        ix = [np.arange(z_bin, i), np.full_like(Bin, j), Bin]
                        SurvivingSubhalos_ByParent[ix] = SurvivingSubhalos_ByParent[ix] + WeightList/AnalyticHaloBin # N Mpc^-3 h^3 dex^-1
                        SurvivingSubhalos_z_z[z_bin:i, i, k] = SurvivingSubhalos_z_z[z_bin:i, i, k] + WeightList/AnalyticHaloBin# N Mpc^-3 h^3 dex^-1
                    #Stripped (Evolved Surviving)
                    if (i !=0):
                        if Stripping_DM:
                            #Wt_Corr = np.histogram2d(z[z_bin:i], NewHaloMass, bins=(z[z_bin:i+1], SatHaloMass), normed = False)[0]
                            Wt_Corr = histogram2d(np.arange(z_bin, i, 1), NewHaloMass, (i-z_bin, c), ((z_bin, i),(SHM_min, SHM_max)))
                            SurvivingSubhalos_Stripped[z_bin:i] = SurvivingSubhalos_Stripped[z_bin:i] + np.divide(np.multiply(WeightList, Wt_Corr.T).T , AnalyticHaloBin) # N Mpc^-3 h^3 dex^-1
                            SurvivingSubhalos_Stripped_ByParent[z_bin:i, j] = SurvivingSubhalos_Stripped_ByParent[z_bin:i, j] + np.divide(np.multiply(WeightList, Wt_Corr.T).T , AnalyticHaloBin) # N Mpc^-3 h^3 dex^-1
                    #Subhalos Saved========================================


                    #Calculate N galaxies from abundace matching====================
                    SM_Sat = F.DarkMatterToStellarMass(np.full(N, SatHaloMass[k]-np.log10(h)), z[i], Paramaters, ScatterOn=True) #Mass Msun
                    #SM_Sizes = F.DarkMastterToStellarRadius()    Chris

                    # Calculate the mass after stripping and starformation
                    if (z_bin < i):
                        #Stellar Mass Stripping/SF
                        MassBefore = np.mean(np.power(10, SM_Sat))
                        #print(SM_Sat)
                        if SF and Stripping:
                            StripFactor = F.StellarMassLoss(AvaHaloMass[i,j], SatHaloMass[k], SM_Sat.T, np.flip(Time_To_0[z_bin:i]), Tdyf, factor_only = True) #Mass Msun
                            SM_Sat, sSFR = F.StarFormation(SM_Sat, TTZ0, Tdyf, z[i], z[z_bin], z, SatHaloMass[k], AvaHaloMass[z_bin:i,j], Paramaters, StripFactor = StripFactor, Stripping = True) #New Stellar Mass log10 Msun and sSFR log10 yr-1 of galaxies (shape (i-z_bin), i)
                        elif SF:
                            SM_Sat, sSFR = F.StarFormation(SM_Sat, TTZ0, Tdyf, z[i], z[z_bin], z, SatHaloMass[k], AvaHaloMass[z_bin:i,j], Paramaters) #New Stellar Mass log10 Msun and sSFR log10 yr-1 of galaxies (shape (i-z_bin), i)
                        elif Stripping:
                            SM_Sat = F.StellarMassLoss(AvaHaloMass[i,j], SatHaloMass[k], SM_Sat, np.flip(Time_To_0[z_bin:i]), Tdyf).T #Mass Msun

                        #saving the Total mass made in each scenario for galaxies that have merged
                        if (Stripping or SF):
                            MassAfter = np.mean(np.power(10, SM_Sat[:,-1]))
                            bin_ = np.digitize(np.log10(MassBefore), bins = Surviving_Sat_SMF_MassRange)
                            if 0 < bin_ <len(Surviving_Sat_SMF_MassRange[:-1]) and z_Merge_Bin != -1:
                                Total_StarFormation[z_bin][j][bin_].append(MassAfter - MassBefore)
                    #We now have stripped halo mass and Satilite Masses=============

                    #Saving the Specific Starformation Rate at redshift 0.1===========
                    if z_bin == 0:
                        if SF:
                            if len(np.shape(SM_Sat)) == 1:
                                Satilite_sSFR = Satilite_sSFR + (histogram2d(SM_Sat, sSFR[:,-1], (SatM_len, sSFR_len),  ((SatM_min, SatM_max),(sSFR_min, sSFR_max)))/N)*WeightList[0]*h_3
                            else:
                                Satilite_sSFR = Satilite_sSFR + (histogram2d(SM_Sat[:,-1], sSFR[:,-1], (SatM_len, sSFR_len),  ((SatM_min, SatM_max),(sSFR_min, sSFR_max)))/N)*WeightList[0]*h_3
                    #Specific Starformation Rate Saved==============================

                    #Build up the SMF/Fractional Plot at redshift 0=================
                    if z_Merge_Bin == -1: #Galaxy has not merged
                        if len(np.shape(SM_Sat)) == 1:
                            Wt_Corr = np.divide(histogram1d(SM_Sat, SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                        else:
                            try:
                                Wt_Corr = np.divide(histogram1d(SM_Sat[:,-1], SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                            except:
                                print(SM_Sat)
                        #SMF
                        try:
                            Surviving_Sat_SMF_Weighting_Totals = Surviving_Sat_SMF_Weighting_Totals + np.divide(WeightList[0]*h_3*Wt_Corr, SatBin) #N Mpc^-3 dex-1
                        except:
                            print(WeightList, type(WeightList))
                        #Fractional
                        Surviving_Sat_SMF_Weighting[j] = Surviving_Sat_SMF_Weighting[j] + np.divide(WeightList[0]*h_3*Wt_Corr, SatBin) #N Mpc^-3 dex-1
                        #infall redshifts
                        z_infall[i] = z_infall[i] + np.divide(WeightList[0]*h_3*Wt_Corr, SatBin)
                    #===============================================================

                    #Build up the SMF/Fractional Plot at High z=====================
                    #Create weights for the Surviving_Sat_SMF_MassRange Bins
                    if z_bin != i and i !=0:
                        if len(np.shape(SM_Sat)) == 1:
                            Wt_Corr = np.divide(histogram1d(SM_Sat, SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                            Wt_Corr = np.full((len(Surviving_Sat_SMF_Weighting_Totals_highz[z_bin:i]), len(Wt_Corr)), Wt_Corr)
                        else:
                            Counterpart = np.multiply(np.ones_like(SM_Sat), np.arange(z_bin,i,1)).T
                            Wt_Corr = np.flipud(np.divide(histogram2d(Counterpart.flatten(), SM_Sat.T.flatten(), (i-z_bin,SatM_len), ((z_bin, i),(SatM_min, SatM_max))), N))
                        #SMF
                        Surviving_Sat_SMF_Weighting_Totals_highz[z_bin:i] = Surviving_Sat_SMF_Weighting_Totals_highz[z_bin:i] + np.divide(np.multiply(WeightList*h_3, Wt_Corr.T).T, SatBin) #N Mpc^-3 dex-1
                        #Fractional
                        Surviving_Sat_SMF_Weighting_highz[z_bin:i, j] = Surviving_Sat_SMF_Weighting_highz[z_bin:i, j] + np.divide(np.multiply(WeightList*h_3, Wt_Corr.T).T, SatBin) #N Mpc^-3 dex-1
                    #===============================================================

                    #code below here does not run in the SMHM relation fitting======
                    if ParamOverRide:
                        continue
                    #===============================================================


                    #satellite SMHM relation at all redshifts=======================
                    if z_bin != i and i !=0:
                        if len(np.shape(SM_Sat)) == 1:
                            #Wt_Corr = np.divide(np.histogram(SM_Sat, bins=Surviving_Sat_SMF_MassRange)[0], N) #Weight per bin from scatter in SM-HM
                            Wt_Corr = np.divide(histogram1d(SM_Sat, SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                            Wt_Corr = np.full((len(Surviving_Sat_SMF_Weighting_Totals_highz[z_bin:i]), len(Wt_Corr)), Wt_Corr)
                        else:
                            #Counterpart = np.multiply(np.ones_like(SM_Sat), z[z_bin:i]).T
                            #Wt_Corr = np.flipud(np.divide(np.histogram2d(Counterpart.flatten(), SM_Sat.T.flatten(), bins=(z[z_bin:i+1], Surviving_Sat_SMF_MassRange), normed = False)[0], N))
                            Counterpart = np.multiply(np.ones_like(SM_Sat), np.arange(z_bin,i,1)).T
                            Wt_Corr = np.flipud(np.divide(histogram2d(Counterpart.flatten(), SM_Sat.T.flatten(), (i-z_bin,SatM_len), ((z_bin, i),(SatM_min, SatM_max))), N))
                        #SMF
                        Sat_SMHM[z_bin:i,k] = Sat_SMHM[z_bin:i,k] + np.divide(np.multiply(WeightList*h_3, Wt_Corr.T).T, SatBin) #N Mpc^-3 dex-1
                        Sat_SMHM_Host[z_bin:i,j] = Sat_SMHM_Host[z_bin:i,j] + np.divide(np.multiply(WeightList*h_3, Wt_Corr.T).T, SatBin) #N Mpc^-3 dex-1
                    else:
                        #Wt_Corr = np.divide(np.histogram(SM_Sat, bins=Surviving_Sat_SMF_MassRange)[0], N) #Weight per bin from scatter in SM-HM
                        Wt_Corr = np.divide(histogram1d(SM_Sat, SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                        Sat_SMHM[i][k] = Sat_SMHM[i][k] + np.divide(WeightList[0]*h_3*Wt_Corr, SatBin)
                        Sat_SMHM_Host[i][j] = Sat_SMHM_Host[i][j] + np.divide(WeightList[0]*h_3*Wt_Corr, SatBin)
                    #===============================================================

                    #Calculate merger rate per masstrack============================
                    if z_Merge_Bin != -1:
                        if len(np.shape(SM_Sat)) == 1:
                            Wt_Corr = np.divide(histogram1d(SM_Sat, SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                        else:
                            Wt_Corr = np.divide(histogram1d(SM_Sat[:,-1], SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM

                        Corr = np.divide(np.multiply(WeightList_SubOnly[0], Wt_Corr), SatBin)
                        Accretion_History[z_bin,j] = Accretion_History[z_bin,j] + Corr #N dex-1 per halo
                        Accretion_History_Halo[z_bin,j,k] = Accretion_History_Halo[z_bin,j,k] + WeightList_SubOnly[0]/AnalyticHaloBin#N dex-1
                    #================================================================

                    #Calculate pair fraction ========================================

                    if z_bin != i:
                        VR = (M_to_R(10**AvaHaloMass[i][j], z[i], 'vir')/h) #kpc

                        #Guo 2011 linear
                        Radius = VR*(1 - (np.abs(Time_To_0[z_bin:i] - Time_To_0[i]))/Tdyf)
                        #print(Radius)
                        #Binney & Tremaine 1987(8) ^1/2
                        #Radius = VR*np.sqrt(1-((np.abs(Time_To_0[z_bin:i] - Time_To_0[i]))/Tdyf))
                        #print(Radius)
                        #input("\n")

                        PF_bin_u = len(Radius[Radius < 30])
                        PF_bin_l = len(Radius[Radius < 5])
                        if len(np.shape(SM_Sat)) == 1:
                            Wt_Corr = np.divide(histogram1d(SM_Sat, SatM_len, (SatM_min, SatM_max)), N) #Weight per bin from scatter in SM-HM
                            Wt_Corr = np.full((len(Time_To_0[z_bin+PF_bin_l:z_bin+PF_bin_u]), len(Wt_Corr)), Wt_Corr) #matching array sizes

                            Corr = np.divide(np.multiply(WeightList_SubOnly[PF_bin_l:PF_bin_u], Wt_Corr.T).T, SatBin)#N dex-1 per halo
                            Pair_Frac[z_bin+PF_bin_l:z_bin+PF_bin_u,j] = Pair_Frac[z_bin+PF_bin_l:z_bin+PF_bin_u,j] + Corr#N dex-1 per halo
                            Pair_Frac_Halo[z_bin+PF_bin_l:z_bin+PF_bin_u,j,k] = Pair_Frac_Halo[z_bin+PF_bin_l:z_bin+PF_bin_u,j,k] + WeightList_SubOnly[PF_bin_l:PF_bin_u]/AnalyticHaloBin #N dex-1 per halo

                    #===============================================================

    #integrate the mass weighting in each satilite bin for diffrent SM cuts
    AnalyticalModel_Cuts_Frac = []
    AnalyticalModel_Cuts_NoFrac = []
    for i, Cut in enumerate(SM_Cuts):
        SM_Bin = np.digitize(Cut, Surviving_Sat_SMF_MassRange)
        Integrals_ = np.array([np.sum(Sat_List)*SatBin for Sat_List in Surviving_Sat_SMF_Weighting[:, SM_Bin:]])
        #Integrals
        AnalyticalModel_Cuts_Frac.append(np.divide(Integrals_, np.sum(Integrals_)))
        AnalyticalModel_Cuts_NoFrac.append(Integrals_)

    #integrate the mass weighting in each satilite bin for diffrent SM cuts high z
    AnalyticalModel_Cuts_Frac_highz = []
    AnalyticalModel_Cuts_NoFrac_highz = []
    for i, Cut in enumerate(SM_Cuts):
        AnalyticalModel_Cuts_Frac_Temp = []
        AnalyticalModel_Cuts_NoFrac_Temp = []
        for j in Surviving_Sat_SMF_Weighting_highz:
            SM_Bin = np.digitize(Cut, Surviving_Sat_SMF_MassRange)
            Integrals_ = np.array([np.sum(Sat_List)*SatBin for Sat_List in j[:, SM_Bin:]])
            #Integrals
            AnalyticalModel_Cuts_Frac_Temp.append(np.divide(Integrals_, np.sum(Integrals_)))
            AnalyticalModel_Cuts_NoFrac_Temp.append(Integrals_)
        AnalyticalModel_Cuts_Frac_highz.append(AnalyticalModel_Cuts_Frac_Temp)
        AnalyticalModel_Cuts_NoFrac_highz.append(AnalyticalModel_Cuts_NoFrac_Temp)
    #=====================Output The Surviving Subhalos========================
    #Unstripped
    if ((Stripping_DM == False) and (Stripping or SF) == False):
        OutHead = np.insert(SatHaloMass, 0, -np.inf)
        Data = np.column_stack((z, SurvivingSubhalos))
        Out = np.vstack((OutHead, Data))
        np.savetxt("./Data/Model/Output/Other/SubHaloes/Surviving_Subhalos{}.dat".format(Factor), Out)

        for i, Halos in enumerate(SurvivingSubhalos):
            if i%20 == 0:
                plt.plot(SatHaloMass, np.log10(Halos), label=z[i])
        plt.ylim(-6,0)
        plt.legend()
        plt.savefig("./Data/Model/Output/Other/SubHaloes/Figures/Surviving_Subhalos{}.png".format(Factor))
        plt.clf()
        np.save("./Data/Model/Output/Other/SubHaloes/Surviving_Subhalos_ByParent{}".format(Factor), SurvivingSubhalos_ByParent)
    #Stripped
    if Stripping_DM == True:
        OutHead = np.insert(SatHaloMass[:-1], 0, -np.inf)
        Data = np.column_stack((z, SurvivingSubhalos_Stripped))
        Out = np.vstack((OutHead, Data))
        np.savetxt("./Data/Model/Output/Other/SubHaloes/Surviving_Subhalos_Stripped{}.dat".format(Factor), Out)
        for i, Halos in enumerate(SurvivingSubhalos_Stripped):
            if i%20 == 0:
                plt.plot(SatHaloMass[:-1], np.log10(Halos), label=z[i])
        plt.ylim(-6,0)
        plt.legend()
        plt.savefig("./Data/Model/Output/Other/SubHaloes/Figures/Surviving_Subhalos_Stripped{}.png".format(Factor))
        plt.clf()
        np.save("./Data/Model/Output/Other/SubHaloes/Surviving_Subhalos_Stripped_ByParent{}".format(Factor), SurvivingSubhalos_Stripped_ByParent)
    if Stripping or SF:
        #Calculate the total Starformation means
        Total_StarFormation_Means = [[[ np.mean(Total_StarFormation[i][j][k]) for k in range(0, len(Surviving_Sat_SMF_MassRange[:-1]))] for j in range(0, b)] for i in range(0, a)]
        Total_StarFormation_Std = [[[ np.std(Total_StarFormation[i][j][k]) for k in range(0, len(Surviving_Sat_SMF_MassRange[:-1]))] for j in range(0, b)] for i in range(0,a)]

    #Check if we are running to make fits, else save the results
    if ParamOverRide:
        return Surviving_Sat_SMF_Weighting_Totals_highz, Surviving_Sat_SMF_MassRange[:-1], z
    else:
        #========================Save Data For Figures==============================
        F.SaveData_3(AvaHaloMass, Surviving_Sat_SMF_Weighting_Totals, Surviving_Sat_SMF_MassRange[:-1], Factor_Stripping_SF)
        F.SaveData_4_6(AvaHaloMass, AnalyticalModel_Cuts_Frac, AnalyticalModel_Cuts_NoFrac, SM_Cuts, Factor_Stripping_SF)
        F.SaveData_10(AvaHaloMass, Surviving_Sat_SMF_Weighting, Surviving_Sat_SMF_MassRange, Factor_Stripping_SF)
        F.SaveData_SMFhz(AvaHaloMass, Surviving_Sat_SMF_Weighting_Totals_highz, Surviving_Sat_SMF_MassRange[:-1], Factor_Stripping_SF)
        F.SaveData_z_infall(Surviving_Sat_SMF_MassRange[:-1], z, z_infall, Factor_Stripping_SF)
        F.SaveData_sSFR(Surviving_Sat_SMF_MassRange[:-1], sSFR_Range[:-1], Satilite_sSFR, Factor_Stripping_SF)
        F.SaveData_Sat_SMHM(z, SatHaloMass, AvaHaloMass, Surviving_Sat_SMF_MassRange[:-1], Sat_SMHM, Sat_SMHM_Host, Factor_Stripping_SF)
        F.SaveData_Mergers(Accretion_History, z, AvaHaloMass, Surviving_Sat_SMF_MassRange[:-1], Factor_Stripping_SF)
        F.SaveData_Pair_Frac(Pair_Frac, z, AvaHaloMass, Surviving_Sat_SMF_MassRange[:-1], Factor_Stripping_SF)
        F.SaveData_Sat_Env_Highz(AvaHaloMass, z, AnalyticalModel_Cuts_Frac_highz, AnalyticalModel_Cuts_NoFrac_highz, SM_Cuts, Factor_Stripping_SF)
        F.SaveData_Raw_Richness(AvaHaloMass, z, Surviving_Sat_SMF_MassRange, Surviving_Sat_SMF_Weighting_highz, Factor_Stripping_SF)
        F.SaveData_MultiEpoch_SubHalos(z, SatHaloMass, SurvivingSubhalos_z_z, Factor_Stripping_SF)
        F.SaveData_Pair_Frac_Halo(Pair_Frac_Halo, Accretion_History_Halo, z, AvaHaloMass, SatHaloMass, Factor_Stripping_SF)
        if (Stripping or SF):
            F.SaveData_Total_Starformation(AvaHaloMass, z, Surviving_Sat_SMF_MassRange[:-1], Total_StarFormation_Means, Total_StarFormation_Std, Factor_Stripping_SF)
        print(Factor_Stripping_SF, time.time() - T1)
        return (Factor_Stripping_SF, time.time() - T1)

#============================Running Loop=======================================
if __name__ == "__main__":
    #Pick the Running paramters for the model each tuple is one run
    #Tuple is (Tdyn_Factor (str), Stripping (bool), Star Fomation (bool), z_evo (Bool), Starformation ('str'), AbnMtch (Str))
    Tdyn_Factors = []
    #Tdyn_Factors += [('1.0', True, True, True, 'S16CE', 'G19_SE'), ('1.0_Alt', True, True, True, 'S16CE', 'G19_SE')]
    #Tdyn_Factors += [('1.0', False, False, True, 'CE', 'G19_SE')]
    #Tdyn_Factors += [('1.0', True, False, True, 'CE', 'G19_SE')]
    #Tdyn_Factors += [('1.0', False, True, True, 'CE', 'G19_SE')]
    #Tdyn_Factors += [('1.0', True, True, True, 'CE', 'G19_SE')]
    #Tdyn_Factors += [('1.0', False, True, True, 'G19_DPL', 'G19_SE')]
    #Tdyn_Factors += [('1.0', True, True, True, 'G19_DPL', 'G19_SE')]
    #Tdyn_Factors += [('1.0', True, True, True, 'G19_DPL_PP', 'G19_SE')]
    #Tdyn_Factors += [('1.2', True, True, True, 'G19_DPL_PP', 'G19_SE')]
    #Tdyn_Factors += [('0.8', True, True, True, 'G19_DPL_PP', 'G19_SE')]
    #Tdyn_Factors += [('1.2', True, True, True, 'G19_DPL', 'G19_SE')]
    #Tdyn_Factors += [('0.8', True, True, True, 'G19_DPL', 'G19_SE')]
    #Tdyn_Factors += [('1.0', False, False, True, 'CE', 'Override_z')]
    Tdyn_Factors += [('1.0', False, False, True, 'G19_DPL', 'Moster')]
    #Tdyn_Factors += [('1.0', True, True, True, 'G19_DPL', 'G19_SE')]
    #Tdyn_Factors += [('1.0', False, True, True, 'CE_PP', 'G19_cMod')]
    #Tdyn_Factors += [('1.0', True, True, True, 'CE_PP', 'G19_cMod')]
    #Tdyn_Factors += [('1.0', True, True, True, 'CE_PP', 'G19_SE')]
    #Tdyn_Factors += [('1.0', True, True, True, 'CE', 'G19_SE')]
    #Tdyn_Factors += [('1.0', True, True, True, 'Illustris', 'Illustris')]
    #Tdyn_Factors += [('1.0', True, True, True, 'Illustris_PP', 'Illustris')]
    #Tdyn_Factors += [('1.0', True, False, True, 'Illustris', 'Illustris')]

    msg = 'About to run' + str(Tdyn_Factors)
    shall = input("%s (y/N) " % msg).lower() != 'y'
    if shall:
        print(shall)
        print("abort")
        quit()

    #Create the folders for saving the output from the model
    F.PrepareToSave(Tdyn_Factors)

    #For runnning single runs without multiprocessing bugs
    #OneRealization(Tdyn_Factors[0])

    #run ecah instance on a seperate core
    pool = multiprocessing.Pool(processes = len(Tdyn_Factors))
    PoolReturn = pool.map(OneRealization, Tdyn_Factors)
    pool.close()
    print(PoolReturn)
