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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append(path_to_STEEL+'Functions')
import Functions as F
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lmfit import Model

plt.ion()

#Abundance Matching Parameters
Override =\
{\
'M10':11.95,\
'SHMnorm10':0.03,\
'beta10':1.5,\
'gamma10':0.7,\
'M11':0.5,\
'SHMnorm11':-0.01,\
'beta11':-0.6,\
'gamma11':0.1\
}

AbnMtch =\
{\
'Behroozi13': False,\
'Behroozi18': True,\
'B18c':True,\
'B18t':False,\
'G18':False,\
'G18_notSE':False,\
'G19_SE':True,\
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

z = np.loadtxt(path_to_STEEL + 'Scripts/Plots/Redshift_array.txt')

HM_vdB = np.loadtxt(path_to_STEEL + 'Scripts/Plots/HM_vdB.dat')


SM = []

for i, HM_Arr in enumerate(HM_vdB):
    SM.append(F.DarkMatterToStellarMass_Alt(HM_Arr,  z[i], Paramaters, ScatterOn=False,))

file_name = path_to_STEEL + 'Data/Observational/Behroozi_catalog/Berhoozi_SM_forFS.txt'
z_Beh = np.loadtxt(file_name, skiprows=3, usecols=0)
SM_Beh = np.loadtxt(file_name, skiprows=3, usecols=(1,2,3,4))

def build_z_Beh_(z_Beh):
    z_min=0.1; z_max=4.
    idx_min=-1; idx_max=-1
    for i in range(0,z_Beh.size):
        if z_Beh[i]>=z_min and idx_min<0:
            idx_min=i
        elif z_Beh[i]>=z_max and idx_max<0:
            idx_max=i
    z_Beh_ = np.zeros(idx_max-idx_min+1)
    for i in range(0,z_Beh_.size):
        z_Beh_[i]=z_Beh[idx_min+i]
    return z_Beh_
    
"""def build_SM_Beh_(z_Beh, SM_Beh):
    #print(SM_Beh.shape)
    z_min=0.1; z_max=4.
    idx_min=-1; idx_max=-1
    for i in range(0,z_Beh.size):
        if z_Beh[i]>=z_min and idx_min<0:
            idx_min=i
        elif z_Beh[i]>=z_max and idx_max<0:
            idx_max=i
    SM_Beh_ = np.zeros(idx_max-idx_min+1)
    for i in range(0,SM_Beh_.size):
        SM_Beh_[i]=SM_Beh[idx_min+i]
    return SM_Beh_"""

def build_HM_to_fit ():
    global z, z_Beh, HM_vdB
    z_Beh_ = build_z_Beh_(z_Beh)
    HM_to_fit = []
    for i in range(0, HM_vdB[0,:].size):
        HM_to_fit.append(np.zeros(z_Beh_.size))
        for j in range(0, z_Beh_.size):
            for k in range(0, z.size):
                if z_Beh_[j]>z[k]:
                    diff1=np.abs(z[k-1]-z_Beh_[j])
                    diff2=np.abs(z[k]-z_Beh_[j])
                    if diff1<diff2:
                        HM_to_fit[i][j]=HM_vdB[k-1,i]
                    else:
                        HM_to_fit[i][j]=HM_vdB[k,i]
                    break
    return np.transpose(np.array(HM_to_fit))

def DM_to_SM_Behr (x, M3, alpha3, beta2, gamma2, set=0):
    #==================================================
    """if set==1:
        e0=-1.435; e1=1.831; e2=1.368; e3=-0.217; M0=12.035; M1=4.556; M2=4.417; M3=-0.731; alpha0=1.963; alpha1=-2.316; alpha2=-1.732; alpha3=0.178; beta0=0.482; beta1=-0.841; beta2=-0.471; gamma0=-1.034; gamma1=-3.100; gamma2=-1.055; delta=0.411 #Obs. All All Excl.
    elif set==2:
        e0=-1.435; e1=1.813; e2=1.353; e3=-0.214; M0=12.081; M1=4.696; M2=4.485; M3=-0.740; alpha0=1.957; alpha1=-2.650; alpha2=-1.953; alpha3=0.204; beta0=0.474; beta1=-0.903; beta2=-0.492; gamma0=-1.065; gamma1=-3.243; gamma2=-1.107; delta=0.386 #Obs. All Cen. Excl.
    elif set==3:
        e0=-1.449; e1=-1.256; e2=-1.031; e3=0.108; M0=11.896; M1=3.284; M2=3.413; M3=-0.580; alpha0=1.949; alpha1=-4.096; alpha2=-3.226; alpha3=0.401; beta0=0.477; beta1=0.046; beta2=-0.214; gamma0=-0.755; gamma1=0.461; gamma2=0.025; delta=0.357 #Obs. All Sat. Excl.
    elif set==4:
        e0=-1.471; e1=-1.952; e2=-2.508; e3=0.499; M0=12.021; M1=3.368; M2=3.615; M3=-0.645; alpha0=1.851; alpha1=-4.244; alpha2=-4.402; alpha3=0.803; beta0=0.505; beta1=-0.125; beta2=-0.094; gamma0=-0.858; gamma1=-0.933; gamma2=-0.098; delta=0.461 #Obs. Q All. Excl.
    elif set==5:
        e0=-1.480; e1=-0.831; e2=-1.351; e3=0.321; M0=12.069; M1=2.646; M2=2.710; M3=-0.431; alpha0=1.899; alpha1=-2.901; alpha2=-2.413; alpha3=0.332; beta0=0.502; beta1=-0.315; beta2=-0.218; gamma0=-0.867; gamma1=-1.146; gamma2=-0.294; delta=0.397 #Obs. Q Cen. Excl.
    elif set==6:
        e0=-1.441; e1=1.697; e2=1.326; e3=-0.221; M0=12.054; M1=4.554; M2=4.484; M3=-0.750; alpha0=1.976; alpha1=-2.123; alpha2=-1.617; alpha3=0.162; beta0=0.465; beta1=-1.071; beta2=-0.659; gamma0=-1.016; gamma1=-2.862; gamma2=-0.941; delta=0.436 #Obs. SF All. Excl.
    elif set==7:
        e0=-1.426; e1=1.588; e2=1.237; e3=-0.210; M0=12.071; M1=4.633; M2=4.527; M3=-0.757; alpha0=1.985; alpha1=-2.492; alpha2=-1.860; alpha3=0.188; beta0=0.448; beta1=-1.121; beta2=-0.665; gamma0=-1.149; gamma1=-3.221; gamma2=-1.022; delta=0.407 #Obs. SF Cen. Excl.
    elif set==8:
        e0=-1.430; e1=1.796; e2=1.360; e3=-0.216; M0=12.040; M1=4.675; M2=4.513; M3=-0.744; alpha0=1.973; alpha1=-2.353; alpha2=-1.783; alpha3=0.186; beta0=0.473; beta1=-0.884; beta2=-0.486; gamma0=-1.088; gamma1=-3.241; gamma2=-1.079; delta=0.407 #True All All Excl.
    elif set==9:
        e0=-1.466; e1=1.852; e2=1.439; e3=-0.227; M0=12.013; M1=4.597; M2=4.470; M3=-0.737; alpha0=1.965; alpha1=-2.137; alpha2=-1.607; alpha3=0.161; beta0=0.564; beta1=-0.835; beta2=-0.478; gamma0=-0.937; gamma1=-2.810; gamma2=-0.983; delta=0.411 #True All All Incl.
    elif set==10:
        e0=-1.431; e1=1.757; e2=1.350; e3=-0.218; M0=12.074; M1=4.600; M2=4.423; M3=-0.732; alpha0=1.974; alpha1=-2.468; alpha2=-1.816; alpha3=0.182; beta0=0.470; beta1=-0.875; beta2=-0.487; gamma0=-1.160; gamma1=-3.634; gamma2=-1.219; delta=0.382 #True All Cen Excl.
    elif set==11:
        e0=-1.462; e1=1.882; e2=1.446; e3=-0.224; M0=12.055; M1=4.667; M2=4.471; M3=-0.735; alpha0=1.956; alpha1=-2.570; alpha2=-1.904; alpha3=0.198; beta0=0.558; beta1=-0.840; beta2=-0.472; gamma0=-1.004; gamma1=-2.983; gamma2=-0.996; delta=0.393 #True All Cen Incl.
    elif set==12:
        e0=-1.432; e1=-1.231; e2=-0.999; e3=0.100; M0=11.889; M1=3.236; M2=3.378; M3=-0.577; alpha0=1.959; alpha1=-4.033; alpha2=-3.175; alpha3=0.390; beta0=0.464; beta1=0.130; beta2=-0.153; gamma0=-0.812; gamma1=0.522; gamma2=0.064; delta=0.319 #True All Sat Excl.
    elif set==13:
        e0=-1.491; e1=-2.313; e2=-2.778; e3=0.492; M0=12.005; M1=3.294; M2=3.669; M3=-0.683; alpha0=1.852; alpha1=-3.922; alpha2=-4.052; alpha3=0.692; beta0=0.511; beta1=-0.028; beta2=-0.041; gamma0=-0.858; gamma1=-0.902; gamma2=-0.041; delta=0.506 #True Q All Excl.
    elif set==14:
        e0=-1.462; e1=-0.732; e2=-1.273; e3=0.302; M0=12.072; M1=3.581; M2=3.665; M3=-0.634; alpha0=1.928; alpha1=-3.472; alpha2=-3.119; alpha3=0.507; beta0=0.488; beta1=-0.419; beta2=-0.256; gamma0=-0.980; gamma1=-1.443; gamma2=-0.335; delta=0.406 #True Q Cen Excl.
    elif set==15:
        e0=-1.494; e1=1.569; e2=1.293; e3=-0.215; M0=12.059; M1=4.645; M2=4.544; M3=-0.757; alpha0=1.905; alpha1=-2.555; alpha2=-1.875; alpha3=0.197; beta0=0.509; beta1=-0.889; beta2=-0.538; gamma0=-0.807; gamma1=-1.859; gamma2=-0.637; delta=0.460 #True SF All Excl.
    elif set==16:
        e0=-1.459; e1=1.515; e2=1.249; e3=-0.214; M0=12.060; M1=4.609; M2=4.525; M3=-0.756; alpha0=1.972; alpha1=-2.523; alpha2=-1.868; alpha3=0.188; beta0=0.488; beta1=-0.965; beta2=-0.569; gamma0=-0.958; gamma1=-2.230; gamma2=-0.706; delta=0.391 #True SF Cen Excl."""
    #==================================================
    e0=-1.431; M0=12.074; alpha0=1.974; beta0=0.470; gamma0=-1.160; delta=0.382
    e1=1.757; M1=4.600; alpha1=-2.468; beta1=-0.875; gamma1=-3.634
    e2=1.350; M2=4.423; alpha2=-0.487
    #global z_Beh_
    #z=z_Beh_
    #global z
    global HM_to_fit_arr
    global k
    DM = HM_to_fit_arr
    e3_arr = np.loadtxt('fitted_params_1.txt', usecols=0)
    if k==0:
        e3 = e3_arr[0]
    elif k==1:
        e3 = e3_arr[1]
    elif k==2:
        e3 = e3_arr[2]
    elif k==3:
        e3 = e3_arr[3]
    print(e3)
    z=x
    a = 1/(1+z)
    afac = a-1
    log10_M = M0 + M1*afac - M2*np.log(a) + M3*z
    e_ = e0 + e1*afac - e2*np.log(a) + e3*z
    alpha_ = alpha0 + alpha1*afac - alpha2*np.log(a) + alpha3*z
    beta_ = beta0 + beta1*afac + beta2*z
    log10_g = gamma0 + gamma1*afac + gamma2*z
    X=DM-log10_M
    gamma_ = 10**log10_g
    Part1 = np.log10(np.power(10, -alpha_*X) + np.power(10, -beta_*X))
    Part2 = np.exp(-0.5*np.power(np.divide(X, delta), 2))
    return log10_M+(e_ - Part1 + gamma_*Part2)


z_Beh_ = build_z_Beh_(z_Beh)
HM_to_fit = build_HM_to_fit ()
HM_to_fit_arr = np.zeros(HM_to_fit[:,0].size)
#print('HM_to_fit successfully built!')

plt.figure()


#Fit with scipy.curve_fit
#p0 = np.array([1.757, 1.350, -0.218, 4.600, 4.423, -0.732, -2.468, -1.816, 0.182, -0.875, -0.487, -3.634, -1.219])
#p0 = np.array([1.350, -0.218, 4.423, -0.732, -1.816, 0.182, -0.487, -1.219])
p0 = np.array([-0.218, -0.732, 0.182, -0.487, -1.219])
idx = [10,14,21,30]
fit_params = []
k=0
for i, DM in enumerate(HM_to_fit):
#for i in idx:
    if k>3:
        break
    #SM_Beh_ = build_SM_Beh_(z_Beh, SM_Beh[:,k])
    func = interp1d(z_Beh, SM_Beh[:,k])
    SM_Beh_ = func(z_Beh_)
    HM_to_fit_arr = HM_to_fit[:,idx[k]]
    popt, pcov = curve_fit(DM_to_SM_Behr, xdata=z_Beh_, ydata=SM_Beh_)#, sigma=np.repeat(0.15,SM_Beh_.size), p0=p0)
    fit_params.append(popt)
    print(popt)
    if k==0:
        plt.semilogx(z_Beh_, DM_to_SM_Behr(z_Beh_, popt[0], popt[1], popt[2], popt[3]), '-.', color='green', label='curve_fit')
    else:
        plt.semilogx(z_Beh_, DM_to_SM_Behr(z_Beh_, popt[0], popt[1], popt[2], popt[3]), '-.', color='green')
    k+=1
np.savetxt('fitted_params.txt', np.array(fit_params), header='Columns: M3, alpha3, beta2, gamma2')


#Fit with lmfit.Model
"""#p0 = np.array([1.757, 1.350, -0.218, 4.600, 4.423, -0.732, -2.468, -1.816, 0.182, -0.875, -0.487, -3.634, -1.219])
p0 = np.array([1.350, -0.218, 4.423, -0.732, -1.816, 0.182, -0.487, -1.219])
model = Model(DM_to_SM_Behr)
params = model.make_params(e1=p0[0], e2=p0[1], e3=p0[2], M1=p0[3], M2=p0[4], M3=p0[5], alpha1=p0[6], alpha2=p0[7], alpha3=p0[8], beta1=p0[9], beta2=p0[10],  gamma1=p0[11], gamma2=p0[12])
idx = [10,13,21]
k=0
for i in idx:
    SM_Beh_ = build_SM_Beh_(z_Beh, SM_Beh[:,k])
    x_eval = HM_to_fit[:,i]
    y_eval = model.eval(x=x_eval, e1=p0[0], e2=p0[1], e3=p0[2], M1=p0[3], M2=p0[4], M3=p0[5], alpha1=p0[6], alpha2=p0[7], alpha3=p0[8], beta1=p0[9], beta2=p0[10],  gamma1=p0[11], gamma2=p0[12])
    result = model.fit(SM_Beh_, x=HM_to_fit[:,i], e1=p0[0], e2=p0[1], e3=p0[2], M1=p0[3], M2=p0[4], M3=p0[5], alpha1=p0[6], alpha2=p0[7], alpha3=p0[8], beta1=p0[9], beta2=p0[10],  gamma1=p0[11], gamma2=p0[12])
    plt.plot(HM_to_fit[:,i], result.best_fit, '-.', color='green')"""



#p0 = np.array([1.757, 1.350, -0.218, 4.600, 4.423, -0.732, -2.468, -1.816, 0.182, -0.875, -0.487, -3.634, -1.219])
#p0 = np.array([1.350, -0.218, 4.423, -0.732, -1.816, 0.182, -0.487, -1.219])
"""e = np.array([-1.431, 1.757, 1.350, -0.218])
M = np.array([12.074, 4.600, 4.423, -0.732])
alpha = np.array([1.974, -2.468, -1.816, 0.182])
beta = np.array([0.470, -0.875, -0.487])
gamma = ([-1.160, -3.634, -1.219])
delta = 0.382""" #Numbers from Behroozi et al. 2019, True Q/SF Cen Excl.

SM=np.array(SM)
k=0
"""for i in idx:
    HM_to_fit_arr = HM_to_fit[:,idx[k]]
    if i==idx[0]:
        #plt.semilogx(z, SM[:,i], '-', color='blue', label='Behroozi param from file')
        plt.semilogx(z_Beh_, DM_to_SM_Behr(z_Beh_, popt[0], popt[1], popt[2], popt[3], popt[4]), '-.', color='black', label='Behroozi param (last set)')
    else:
        #plt.semilogx(z, SM[:,i], '-', color='blue')
        plt.semilogx(z_Beh_, DM_to_SM_Behr(z_Beh_, popt[0], popt[1], popt[2], popt[3], popt[4]), '-.', color='black')
    plt.figure(i)
    for j in range(1,17):
        SM_Beh_ = build_SM_Beh_(z_Beh, SM_Beh[:,k])
        xxx = DM_to_SM_Behr(HM_to_fit[:,i], e1=p0[0], e2=p0[1], e3=p0[2], M1=p0[3], M2=p0[4], M3=p0[5], alpha1=p0[6], alpha2=p0[7], alpha3=p0[8], beta1=p0[9], beta2=p0[10],  gamma1=p0[11], gamma2=p0[12], set=j) - SM_Beh_
        plt.semilogx(z_Beh_, xxx, '-.', label='{}'.format(j))
    plt.legend(fontsize=6)
    plt.xlim(0.1,4)
    #plt.ylim(9.5, 11.8)
    plt.xlabel(r'$z$', fontsize=20)
    plt.ylabel(r'$log10 M_*$', fontsize=20)
    k+=1"""

for k in range(0,4):
    if k==0:
        plt.semilogx(z_Beh, SM_Beh[:,k], '--', color='red', label='Behroozi data')
    else:
        plt.semilogx(z_Beh, SM_Beh[:,k], '--', color='red')


plt.xlim(0.1,4)
plt.ylim(9.5)#, 11.8)
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$log10 M_*$', fontsize=20)
plt.legend(fontsize=14)
#plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)





"""plt.figure()
z=np.arange(0.1,4,0.1)
plt.plot(z, z/(1+z), label=r'$z / 1+z$')
plt.plot(z, np.log(1+z), label=r'$ln(1+z)$')
plt.plot(z, z, label=r'$z$')
plt.legend()"""
