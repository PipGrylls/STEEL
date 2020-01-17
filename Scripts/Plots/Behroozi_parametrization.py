import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ScatterOn = True

z = np.loadtxt('Redshift_array.txt')
HM_vdB = np.loadtxt('HM_vdB.dat')

e = np.array([-1.331, 1.757, 1.350, -0.218]) #np.array([-1.431, 1.757, 1.350, -0.218])
M = np.array([12.074, 4.600, 4.423, -0.732])
alpha = np.array([1.974, -2.468, -1.816, 0.182])
beta = np.array([0.470, -0.875, -0.487])
gamma = ([-1.160, -3.634, -1.219])
delta = 0.382

a      = 1/(1+z)
afac   = a-1

log10_M  = M[0]     + (M[1]*afac)     - (M[2]*np.log(a))     + (M[3]*z)
e_       = e[0]     + (e[1]*afac)     - (e[2]*np.log(a))     + (e[3]*z)
alpha_   = alpha[0] + (alpha[1]*afac) - (alpha[2]*np.log(a)) + (alpha[3]*z)
beta_    = beta[0]  + (beta[1]*afac)                         + (beta[2]*z)
log10_g  = gamma[0] + (gamma[1]*afac)                        + (gamma[2]*z)

plt.ion(); plt.figure()
idx = [9,12,20]
N = 10000
for i in idx:
    x = HM_vdB[:,i]-log10_M
    gamma_ = np.power(10, log10_g)

    Part1 = np.log10(np.power(10, -alpha_*x) + np.power(10, -beta_*x))
    Part2 = np.exp(-0.5*np.power(np.divide(x, delta), 2))
    M_Star = []
    for j in range (0,N):
        M_Star.append(log10_M+(e_ - Part1 + gamma_*Part2))

        Scatter=np.linspace(0.15,0.25,z.size)
        if ScatterOn:
            Scatter = np.random.normal(scale = Scatter, size = np.shape(M_Star[j]))
            M_Star[j] += Scatter
    M_Star = np.array(M_Star)
    M_Star_median = np.zeros(z.size)
    for j in range(0,z.size):
        M_Star_median[j] = np.median(M_Star[:,j])
    
    if i==idx[0]:
        plt.semilogx(z, M_Star_median, '-.', color='green', label='Behroozi Monte Carlo')
    else:
        plt.semilogx(z, M_Star_median, '-.', color='green')
        

"""SM_Behr = np.loadtxt('SM_track_G19.txt')
#idx_ = [8,11,15]
idx_ = [8,10,14]
for i in idx_:
    plt.semilogx(z, SM_Behr[:,i], '-.')"""


z_Beh_to_plot = np.loadtxt('Behroozi_z_to_plot.txt')
SM_Beh_to_plot = np.loadtxt('Behroozi_SM_to_plot.txt')
obs_SM_Beh_to_plot = np.loadtxt('Behroozi_obs_SM_to_plot.txt')
idx = [10,14,21]
for i in idx:
    if i==idx[0]:
        plt.semilogx(z_Beh_to_plot, SM_Beh_to_plot[:,i], '--', color='dodgerblue', label='Behroozi catalog (True) binning')
        plt.semilogx(z_Beh_to_plot, obs_SM_Beh_to_plot[:,i], '-.', color='violet', label='Behroozi catalog (Obs.) binning')
    else:
        plt.semilogx(z_Beh_to_plot, SM_Beh_to_plot[:,i], '--', color='dodgerblue')
        plt.semilogx(z_Beh_to_plot, obs_SM_Beh_to_plot[:,i], '-.', color='violet')


file_name = '../../Data/Observational/Behroozi_catalog/Berhoozi_SM_forFS.txt'
z_Beh = np.loadtxt(file_name, skiprows=3, usecols=0)
SM_Beh = np.loadtxt(file_name, skiprows=3, usecols=(1,2,3,4))
for k in range(0,3):
    if k==0:
        plt.semilogx(z_Beh, SM_Beh[:,k], '-', color='red', label='Behroozi data (Chris)')
    else:
        plt.semilogx(z_Beh, SM_Beh[:,k], '-', color='red')

plt.xlim(0.2,4)
plt.ylim(9.5)
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$log10 M_*$', fontsize=20)
plt.legend(fontsize=14)
#plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
