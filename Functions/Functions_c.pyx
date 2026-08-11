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

def SeedRandomState(unsigned long int seed):
    """Seed the star-formation-rate scatter generator.

    `gsl_rng_alloc` leaves the generator on its fixed built-in default
    seed, so every process -- including every worker of the
    multiprocessing pool -- drew the identical scatter sequence, and
    there was no way to vary or reproduce it deliberately. Called from
    `Functions.SeedRandomState`.
    """
    gsl_rng_set(RNG_set, seed)

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
        double C0 = 0.05 
        double Lambda = 1.4*c_pow(10,6)
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
    if SFR_Model == "G19_DPL": SFR_Model_int = 6
    if SFR_Model == "Test": SFR_Model_int = 7
    
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
                    #Schreiber+2015 Eq. 9 is max(0, m - m1 - a2 r): the
                    #quadratic term switches ON above the knee mass and
                    #bends the main sequence down at high mass. This used
                    #to read `if Max > 0: Max = 0`, the opposite, which
                    #deletes the high-mass bend and applies the penalty to
                    #low-mass galaxies instead. Functions.py's
                    #StarFormationRate has always had it the right way
                    #round; the two disagreed and this, the compiled hot
                    #loop, is the one that runs.
                    Max = m-m1-a2*r
                    if Max < 0:
                        Max = 0
                    log10MperY = m-m0+a0*r-a1*c_pow(Max, 2)
                #Schreiber 2015
                if SFR_Model_int == 4:
                    m = M_out[k,i]-9
                    r = c_log10(1+z[i])
                    m0, a0, a1, m1, a2 = 0.75, 1.75, 0.3, 0.36, 1.75
                    #Schreiber+2015 Eq. 9 is max(0, m - m1 - a2 r): the
                    #quadratic term switches ON above the knee mass and
                    #bends the main sequence down at high mass. This used
                    #to read `if Max > 0: Max = 0`, the opposite, which
                    #deletes the high-mass bend and applies the penalty to
                    #low-mass galaxies instead. Functions.py's
                    #StarFormationRate has always had it the right way
                    #round; the two disagreed and this, the compiled hot
                    #loop, is the one that runs.
                    Max = m-m1-a2*r
                    if Max < 0:
                        Max = 0
                    log10MperY = m-m0+a0*r-a1*c_pow(Max, 2)
                #Illustrius CE
                if SFR_Model_int == 5:
                    s0 = 0.6+ 1.22*(z[i]) - 0.2*(z[i]**2)
                    logM0 = 10.7 + 0.5*(z[i]) - 0.09*(z[i]**2)
                    Gamma = -(1.6 - 0.25*(z[i]) + 0.01*(z[i]**2))#including -ve here to avoid it later
                    log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[k,i] - logM0) ), Gamma))
                #G19_DPL
                if SFR_Model_int == 6:
                    M_n = 10.7+ 0.34*z[i] - 0.079*(z[i]**2) #logMsun
                    Norm = c_pow(10, 0.74+ 0.71*z[i] - 0.087*(z[i]**2)) #SFR peak
                    Alpha = 1.035 - 0.022*z[i] + 0.0077*(z[i]**2) #low mass slope 
                    Beta = 1.55 - 0.35*z[i] - 0.02*(z[i]**2)#high mass slope
                    MperY = 2*Norm*c_pow( c_pow(10, -Alpha*(M_out[k,i]-M_n)) + c_pow(10, Beta*(M_out[k,i]-M_n)),-1) #SFR
                    log10MperY = c_log10(MperY) #logSFR
                #Test
                if SFR_Model_int == 7:
                    s0 = 0.6+ 1.1*(z[i]) - 0.12*(z[i]**2)
                    logM0 = 10.3 + 0.753*(z[i]) - 0.11*(z[i]**2)
                    Gamma = -(1.3 - 0.12*(z[i]))# + 0.01*(z[i]**2))#including -ve here to avoid it later
                    log10MperY = s0 - c_log10(1 + c_pow(c_pow(10, (M_out[k,i] - logM0) ), Gamma))
                
                SFR = c_pow(10, log10MperY)            
                
               
                #Check Gas depletion
                if Stripping == 1:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0]+StripFactor[i])
                    #MaxGas is a log10 mass, so it has to be raised
                    #before being combined with the linear SM_new. This
                    #line used to treat it as linear while the cap test
                    #below treated it as a log -- the same variable used
                    #both ways inside one function.
                    GasMass[k,i] = c_pow(10, MaxGas[k]+StripFactor[i]) - SM_new
                else:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0])
                    GasMass[k,i] = c_pow(10, MaxGas[k]) - SM_new
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
                    #MaxGas is a log10 mass, so it has to be raised
                    #before being combined with the linear SM_new. This
                    #line used to treat it as linear while the cap test
                    #below treated it as a log -- the same variable used
                    #both ways inside one function.
                    GasMass[k,i] = c_pow(10, MaxGas[k]+StripFactor[i]) - SM_new
                else:
                    SM_new = c_pow(10,M_out[k,i]) - c_pow(10,M_out[k,0])
                    GasMass[k,i] = c_pow(10, MaxGas[k]) - SM_new
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
                #The stripping and no-stripping branches were identical
                #except that the stripped one also ran
                #    SFH[k,i] = SFH[k,i] + (StripFactor[i+1]-StripFactor[i])
                #inside this loop. SFH is a mass in Msun (SFR*dt*1e9);
                #StripFactor is a base-10 logarithm of a surviving
                #fraction, so their difference is a dimensionless
                #log-ratio of order -0.01 to -1. Adding it to a quantity
                #of order 1e8 Msun is a unit error, and being inside
                #`for j in range(i)` it was applied i times per timestep
                #rather than once. SFH feeds both the recycled mass-loss
                #rate below and the reported sSFR.
                #
                #Tidal stripping of already-formed stars is applied where
                #it belongs, to the stellar mass itself, in the M_out
                #update further down -- which already carries exactly this
                #(StripFactor[i+1]-StripFactor[i]) term, in the log domain
                #where it is dimensionally correct. So this line was not
                #just mis-scaled, it was a double count.
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
        double Lambda =  1.4*c_pow(10,6)
        double beta = -0.25
        double s0, logM0, Gamma, log10MperY_0, log10MperY, sSFR, SM_new, SFR_tquench, alpha_l, beta_l, Factor, Scatter
        double A, B
        double m,r,m0,a0,a1,m1,a2,Max
        int SFR_Model_int
        double M_n, Norm, Alpha, Beta, MperY
    #keeping stringformat checks out of loop (minor python interaction)
    if SFR_Model == "T16": SFR_Model_int = 1  
    if SFR_Model == "CE": SFR_Model_int = 2  
    if SFR_Model == "S15": SFR_Model_int = 3    
    if SFR_Model == "S16CE": SFR_Model_int = 4
    if SFR_Model == "Illustris": SFR_Model_int = 5
    if SFR_Model == "G19_DPL": SFR_Model_int = 6
    if SFR_Model == "Test": SFR_Model_int = 7
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
                m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
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
            #G19_DPL
            if SFR_Model_int == 6:
                M_n = 10.7+ 0.34*z[i] - 0.079*(z[i]**2) #logMsun
                Norm = c_pow(10, 0.74+ 0.71*z[i] - 0.087*(z[i]**2)) #SFR peak
                Alpha = 1.035 - 0.022*z[i] + 0.0077*(z[i]**2) #low mass slope 
                Beta = 1.55 - 0.35*z[i] - 0.02*(z[i]**2)#high mass slope
                MperY = 2*Norm*c_pow( c_pow(10, -Alpha*(M_out[i]-M_n)) + c_pow(10, Beta*(M_out[i]-M_n)),-1) #SFR
                log10MperY = c_log10(MperY) #logSFR
            #Test
            if SFR_Model_int == 7:
                M_n = 10.7+ 0.4*z[i] - 0.075*(z[i]**2) #logMsun
                Norm = c_pow(10, 0.7 + 0.74*z[i] - 0.085*(z[i]**2)) #SFR peak
                Alpha = 1.05 #low mass slope
                Beta = 1.2 - 0.15*z[i] #high mass slope
                MperY = 2*Norm*c_pow( c_pow(10, -Alpha*(M_out[i]-M_n)) + c_pow(10, Beta*(M_out[i]-M_n)),-1) #SFR
                log10MperY = c_log10(MperY) #logSFR
            SFR = c_pow(10, log10MperY)
            
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
