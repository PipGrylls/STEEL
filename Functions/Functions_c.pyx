# cython: profile=True, boundscheck=False, wraparound=False, nonecheck=False, cdivision = True
cimport cython
import numpy as np
cimport numpy as np
from colossus.cosmology import cosmology
from libc.math cimport pow as c_pow
from libc.math cimport log10 as c_log10
from libc.math cimport log as c_log
from libc.math cimport abs as c_abs
from libc.math cimport exp as c_exp
from libc.time cimport time, time_t
from libcpp cimport bool
from cython_gsl cimport *

cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()
cdef:
    double h = Cosmo.h
    double Ol = 0
    double Om = Cosmo.Om(0)
    double Or = Cosmo.Or(0)
    double O0 = Ol+Om+Or
    gsl_rng* RNG_set = gsl_rng_alloc(gsl_rng_taus)

def HaloMassLoss_c(double m, double[:] M, double[:] z, double[:] delta_t):
    #accelerated loop for HaloMassLoss
    cdef:
        int N = M.shape[0]
        int i 
        double Zeta = 0.07
        double Zeta_Pwr = (-1/Zeta)
        double A = 0.81
        double[:] m_new = np.zeros(N)
        double m_M, Ez, x, Tau, Part1, Part2
    m_new[0] = m
    
    for i in range(N - 1):
        m_M = 10.**(m_new[i] - M[i])
        Ez = (Ol + (1-O0)*(1+z[i])**2 + Om*(1+z[i])**3 + Or*(1+z[i])**4)**0.5
        x = 1 - (Om*(1+z[i])**3)/(Ez**2)
        dVz = (18*c_pow(pi, 2) + 82*x - 39*c_pow(x, 2)) / (1-x)
        Tau = (1.628*c_pow(h,-1)*c_pow(dVz/178, -0.5)*c_pow(Ez, -1))/A
        Part1 = Zeta*c_pow(m_M,Zeta)
        Part2 = (delta_t[i]/Tau)
        m_new[i + 1] = c_log10(c_pow(10, m_new[i])*c_pow((1+(Part1*Part2)), Zeta_Pwr))
    return np.array(m_new)

def Starformation_c(double[:] M_infall, double[:] t, double[:] delta_t, double[:] z, double[:] MaxGas, double[:] T_quench, double[:] Tau_f, double[:] StripFactor, double z_infall = -1, str SFR_Model = "CE", int Stripping = 0, int Scatter_On = 1):
    #accelerated loop for baryonic evolution processes
    cdef:
        int N_gal = M_infall.shape[0]
        int N = delta_t.shape[0]
        int i, j, k
        double[:,:] M_out = np.zeros((N_gal, N))
        double[:,:] GMLR = np.zeros((N_gal, N))
        double[:,:] M_dot = np.zeros((N_gal, N))
        double[:,:] SFH = np.zeros((N_gal, N))
        double[:,:] SFH_Stripped = np.zeros((N_gal, N))
        double[:,:] GasMass = np.zeros((N_gal, N))
        double SFR, Residual, alpha, 
        double A0 = 2.8 #Msun yr-1
        double C0 = 0.05 #0.046
        double Lambda =  3*c_pow(10,5)#1.4*c_pow(10,6)
        double beta = -0.25
        double s0, logM0, Gamma, log10MperY_0, log10MperY_5, log10MperY, sSFR, SM_new, SFR_tquench, alpha_l, beta_l, Factor, Scatter
        double A, B
        double m,r,m0,a0,a1,m1,a2,Max
        int SFR_Model_int
    #keeping stringformat checks out of loop (minor python interaction)
    if SFR_Model == "T16": SFR_Model_int = 1  
    if SFR_Model == "CE": SFR_Model_int = 2  
    if SFR_Model == "S16": SFR_Model_int = 3    
    if SFR_Model == "S16CE": SFR_Model_int = 4
    if SFR_Model == "Illustris": SFR_Model_int = 5
    if SFR_Model == "Test": SFR_Model_int = 6
    
    #Loop over galaxies
    for k in range(N_gal):
        #if we wish to assume galaxy stars are formed in a burst at infall
        #SFH[k,0] = SFH[k,0] + c_pow(10, M_infall[k])  

        #Fill M_out/M_dot assuming GMLR = 0
        M_out[k,0] = M_infall[k]
        #loop over timesteps
        for i in range(N):
            #if the quenching time has not been met
            if T_quench[k] < t[i] or i == 0:
                #Tomzac SFR-All Galaxies
                if SFR_Model_int == 1:
                    s0 = 0.195 + 1.157*(z[i]) - 0.143*(z[i]**2)
                    logM0 = 9.244 + 0.753*(z[i]) - 0.09*(z[i]**2)
                    Gamma = -1.118 #including -ve here to avoid it later              
                    log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[k,i] - logM0) ), Gamma))               
                #use the tomczak fit with the CE parameters
                if SFR_Model_int == 2:
                    s0 = 0.6 + 1.22*(z[i]) - 0.2*(z[i]**2)
                    logM0 = 10.3 + 0.753*(z[i]) - 0.15*(z[i]**2)
                    Gamma = -(1.3 - 0.1*(z[i])) #including -ve here to avoid it later
                    log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[k,i] - logM0) ), Gamma))
                #Schreiber 2015
                if SFR_Model_int == 3:
                    m = M_out[k,i]-9
                    r = c_log10(1+z[i])
                    m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
                    Max = m-m1-a2*r
                    if Max > 0:
                        Max = 0
                    log10MperY = m-m0+a0*r-a1*c_pow(Max, 2)
                #Schreiber 2015
                if SFR_Model_int == 4:
                    m = M_out[k,i]-9
                    r = c_log10(1+z[i])
                    m0, a0, a1, m1, a2 = 0.75, 1.75, 0.3, 0.36, 1.75
                    Max = m-m1-a2*r
                    if Max > 0:
                        Max = 0
                    log10MperY = m-m0+a0*r-a1*c_pow(Max, 2)
                #Illustrius CE
                if SFR_Model_int == 5:
                    s0 = 0.6+ 1.22*(z[i]) - 0.2*(z[i]**2)
                    logM0 = 10.7 + 0.5*(z[i]) - 0.09*(z[i]**2)
                    Gamma = -(1.6 - 0.25*(z[i]) + 0.01*(z[i]**2))#including -ve here to avoid it later
                    log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[k,i] - logM0) ), Gamma))
                #Test
                if SFR_Model_int == 6:
                    s0 = 0.6+ 1.1*(z[i]) - 0.12*(z[i]**2)
                    logM0 = 10.3 + 0.753*(z[i]) - 0.11*(z[i]**2)
                    Gamma = -(1.3 - 0.12*(z[i]))# + 0.01*(z[i]**2))#including -ve here to avoid it later
                    log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[k,i] - logM0) ), Gamma))
                
                SFR = c_pow(10, log10MperY)            
                
               
                #Check Gas depletion
                if Stripping == 1:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0]+StripFactor[i])
                    GasMass[k,i] = MaxGas[k]*c_pow(10,StripFactor[i]) - SM_new
                else:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0])
                    GasMass[k,i] = MaxGas[k] - SM_new
                if SM_new > 0:                
                    if c_log10(SM_new) > MaxGas[k]:
                        SFR = c_pow(10,M_out[k,i]-12.0) 
                
                #check sSFR
                sSFR = SFR/c_pow(10, M_out[k,i])               
                if sSFR < c_pow(10.0, -12):
                    SFR = c_pow(10,M_out[k,i] -12.0)
                
                SFR_tquench = SFR
            else:
                #galaxy is now quenched
                #apply fastmode quenching 
                SFR = SFR_tquench*c_exp(-((T_quench[k]-t[i])/Tau_f[k]))
                
                #Check Gas depletion
                if Stripping == 1:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0]+StripFactor[i])
                    GasMass[k,i] = MaxGas[k]*c_pow(10,StripFactor[i]) - SM_new
                else:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0])
                    GasMass[k,i] = MaxGas[k] - SM_new
                if SM_new > 0:                
                    if c_log10(SM_new) > MaxGas[k]:
                        SFR = c_pow(10,M_out[k,i]-12.0)
                #check sSFR
                sSFR = SFR/c_pow(10, M_out[k,i])              
                if sSFR <= c_pow(10.0, -12):
                    SFR = c_pow(10,M_out[k,i] -12.0)
            
            #apply sactter to SFR
            if Scatter_On == 1:
                Scatter = gsl_ran_gaussian(RNG_set, 0.3) # dex
                SFR = c_pow(10,c_log10(SFR)+(Scatter))
     
            #Set the star formation history actual amount of stars made in d_t[i]
            SFH[k,i] = SFR*delta_t[i]*c_pow(10, 9) #Msun
                
            #Calculate the GMLR 
            if i > 0 and i < N-1:
                #(and strip the SFH for the next loop saving additional loop)
                if Stripping == 1:
                    for j in range(i):
                        f_mr_1 = (1 - C0*c_log(((c_abs(t[j]-t[i])*c_pow(10, 9))/Lambda)+1))
                        f_mr_2 = (1 - C0*c_log(((c_abs(t[j]-t[i+1])*c_pow(10, 9))/Lambda)+1))
                        GMLR[k,i] = GMLR[k,i] + (c_abs(SFH[k,j]*(f_mr_1 - f_mr_2))/(c_abs(t[i] - t[i+1])*c_pow(10, 9))) #Msun yr-1    
                        SFH[k,i] = SFH[k,i]+(StripFactor[i+1]-StripFactor[i])
                else:
                    for j in range(i):
                        f_mr_1 = (1 - C0*c_log(((c_abs(t[j]-t[i])*c_pow(10, 9))/Lambda)+1))
                        f_mr_2 = (1 - C0*c_log(((c_abs(t[j]-t[i+1])*c_pow(10, 9))/Lambda)+1))
                        GMLR[k,i] = GMLR[k,i] + (c_abs(SFH[k,j]*(f_mr_1 - f_mr_2))/(c_abs(t[i] - t[i+1])*c_pow(10, 9))) #Msun yr-1
            #Set Mdot (rate of change of mass) at time t[i]
            M_dot[k,i] = SFR - GMLR[k,i] #Mun yr-1
            if i < N-1:
                if Stripping == 1:
                    M_out[k,i+1] = c_log10(c_pow(10, M_out[k,i]+(StripFactor[i+1]-StripFactor[i])) + M_dot[k,i]*(delta_t[i]*c_pow(10, 9))) #log10 Msun
                else:
                    M_out[k,i+1] = c_log10(c_pow(10, M_out[k,i]) + M_dot[k,i]*(delta_t[i]*c_pow(10, 9))) #log10 Msun
                
            
    return M_out, M_dot, SFH, GMLR




def Starformation_Centrals(double M_infall, double[:] t, double[:] delta_t, double[:] z, double[:] M_acc, double MaxGas, double T_quench, double Tau_f, str SFR_Model = "CE", int Scatter_On = 1):
    #accelerated loop for baryonic evolution processes
    cdef:
        int N = delta_t.shape[0]
        int i, j
        double[:] M_out = np.zeros((N))
        double[:] GMLR = np.zeros((N))
        double[:] M_dot = np.zeros((N))
        double[:] M_dot_noacc = np.zeros((N))
        double[:] SFH = np.zeros((N))
        double[:] SFH_Stripped = np.zeros((N))
        double[:] GasMass = np.zeros((N))
        double SFR, Residual, alpha, 
        double A0 = 2.8 #Msun yr-1
        double C0 = 0.05 #0.046
        double Lambda =  3*c_pow(10,5)#1.4*c_pow(10,6)
        double beta = -0.25
        double s0, logM0, Gamma, log10MperY_0, log10MperY_5, log10MperY, sSFR, SM_new, SFR_tquench, alpha_l, beta_l, Factor, Scatter
        double A, B
        double m,r,m0,a0,a1,m1,a2,Max
        int SFR_Model_int
        double M_n, Norm, Alpha, Beta, MperY
    #keeping stringformat checks out of loop (minor python interaction)
    if SFR_Model == "T16": SFR_Model_int = 1  
    if SFR_Model == "CE": SFR_Model_int = 2  
    if SFR_Model == "S16": SFR_Model_int = 3    
    if SFR_Model == "S16CE": SFR_Model_int = 4
    if SFR_Model == "Illustris": SFR_Model_int = 5
    if SFR_Model == "Test": SFR_Model_int = 6
    M_out[0] = M_infall
    for i in range(N):
        #if the quenching time has not been met
        if T_quench < t[i] or i == 0:
            #Tomzac SFR-All Galaxies
            if SFR_Model_int == 1:
                s0 = 0.195 + 1.157*(z[i]) - 0.143*(z[i]**2)
                logM0 = 9.244 + 0.753*(z[i]) - 0.09*(z[i]**2)
                Gamma = -1.118 #including -ve here to avoid it later              
                log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[i] - logM0) ), Gamma))               
            #use the tomczak fit with the CE parameters
            if SFR_Model_int == 2:
                s0 = 0.6 + 1.22*(z[i]) - 0.2*(z[i]**2)
                logM0 = 10.3 + 0.753*(z[i]) - 0.15*(z[i]**2)
                Gamma = -(1.3 - 0.1*(z[i])) #including -ve here to avoid it later
                log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[i] - logM0) ), Gamma))
            #Schreiber 2015
            if SFR_Model_int == 3:
                m = M_out[i]-9
                r = c_log10(1+z[i])
                m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
                Max = m-m1-a2*r
                if Max > 0:
                    Max = 0
                log10MperY = m-m0+a0*r-a1*c_pow(Max, 2)
            #Schreiber 2015
            if SFR_Model_int == 4:
                m = M_out[i]-9
                r = c_log10(1+z[i])
                m0, a0, a1, m1, a2 = 0.75, 1.75, 0.3, 0.36, 1.75
                Max = m-m1-a2*r
                if Max > 0:
                    Max = 0
                log10MperY = m-m0+a0*r-a1*c_pow(Max, 2)
            #Illustrius CE
            if SFR_Model_int == 5:
                s0 = 0.6+ 1.22*(z[i]) - 0.2*(z[i]**2)
                logM0 = 10.7 + 0.5*(z[i]) - 0.09*(z[i]**2)
                Gamma = -(1.6 - 0.25*(z[i]) + 0.01*(z[i]**2))#including -ve here to avoid it later
                log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[i] - logM0) ), Gamma))
            #Test
            if SFR_Model_int == 6:
                M_n = 10.6+ 0.4*z[i] - 0.075*(z[i]**2) #logMsun
                Norm = c_pow(10, 0.7 + 0.74*z[i] - 0.085*(z[i]**2)) #SFR peak
                Alpha = 1.05 #low mass slope
                Beta = 1.2 - 0.15*z[i] #high mass slope
                MperY = 2*Norm*c_pow( c_pow(10, -Alpha*(M_out[i]-M_n)) + c_pow(10, Beta*(M_out[i]-M_n)),-1) #SFR
                log10MperY = c_log10(MperY) #logSFR
            SFR = c_pow(10, log10MperY)            
                
               
            #Check Gas depletion
            SM_new = c_pow(10,M_out[i]) - c_pow(10,M_out[0])
            GasMass[i] = MaxGas - SM_new
            if SM_new > 0:                
                if c_log10(SM_new) > MaxGas:
                    SFR = c_pow(10,M_out[i]-12.0) 
            
            #check sSFR
            sSFR = SFR/c_pow(10, M_out[i])               
            if sSFR < c_pow(10.0, -12):
                SFR = c_pow(10,M_out[i] -12.0)
            
            SFR_tquench = SFR
        else:
            #galaxy is now quenched
            #apply fastmode quenching 
            SFR = SFR_tquench*c_exp(-((T_quench-t[i])/Tau_f))
            
            #Check Gas depletion
            SM_new = c_pow(10,M_out[i]) - c_pow(10,M_out[0])
            GasMass[i] = MaxGas - SM_new
            if SM_new > 0:                
                if c_log10(SM_new) > MaxGas:
                    SFR = c_pow(10,M_out[i]-12.0)
            #check sSFR
            sSFR = SFR/c_pow(10, M_out[i])              
            if sSFR <= c_pow(10.0, -12):
                SFR = c_pow(10,M_out[i] -12.0)
            
        #apply sactter to SFR
        if Scatter_On == 1:
            Scatter = gsl_ran_gaussian(RNG_set, 0.3) # dex
            SFR = c_pow(10,c_log10(SFR)+(Scatter))
    
        #Set the star formation history actual amount of stars made in d_t[i]
        SFH[i] = SFR*delta_t[i]*c_pow(10, 9) #Msun
            
        #Calculate the GMLR 
        if i > 0 and i < N-1:
            for j in range(i):
                f_mr_1 = (1 - C0*c_log(((c_abs(t[j]-t[i])*c_pow(10, 9))/Lambda)+1))
                f_mr_2 = (1 - C0*c_log(((c_abs(t[j]-t[i+1])*c_pow(10, 9))/Lambda)+1))
                GMLR[i] = GMLR[i] + (c_abs(SFH[j]*(f_mr_1 - f_mr_2))/(c_abs(t[i] - t[i+1])*c_pow(10, 9))) #Msun yr-1
        #Set Mdot (rate of change of mass) at time t[i]
        M_dot[i] = M_acc[i] + SFR - GMLR[i] #Mun yr-1
        M_dot_noacc[i] = SFR - GMLR[i] #Mun yr-1
        if i < N-1:
            M_out[i+1] = c_log10(c_pow(10, M_out[i]) + M_dot[i]*(delta_t[i]*c_pow(10, 9))) #log10 Msun      
            
    return M_out, M_dot, M_dot_noacc, SFH, GMLR
