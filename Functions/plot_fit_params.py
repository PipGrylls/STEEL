import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.ion()

def logarit(x, a, b, c):
    return a*np.log10(b*x+c)
    
def straight(x, a, b):
    return a*x+b

e3 = np.loadtxt('fitted_params.txt', usecols=0)
M3 = np.loadtxt('fitted_params.txt', usecols=1)
alpha3 = np.loadtxt('fitted_params.txt', usecols=2)
beta2 = np.loadtxt('fitted_params.txt', usecols=3)
gamma2 = np.loadtxt('fitted_params.txt', usecols=4)

DM = np.array([11.8105, 12.2082, 12.903, 13.7928])
x = np.linspace(11.5, 14, 1000)


#================ e3 =================
popt, pcov = curve_fit(straight, DM, e3)
plt.figure()
plt.plot(DM,e3,'o')
plt.plot(x, straight(x, popt[0], popt[1]))
plt.xlabel(r'$log_{10} ( HM(z=0) / M_\odot )$', fontsize=15)
plt.ylabel(r'$\epsilon_z$', fontsize=15)
plt.title(r'Evolution of $\epsilon_z$ with $HM(z=0)$', fontsize=20)
plt.grid()
print('Evolution of e3: y = ax + b')
print('a = {}'.format(popt[0]))
print('b = {}'.format(popt[1]))

#================ M3 ================= not found yet
"""plt.figure()
plt.plot(DM,M3,'o')"""

#================ alpha3 ================= not found yet
"""plt.figure()
plt.plot(DM,alpha3,'o')"""

#================ beta2 ================= not found yet
"""plt.figure()
plt.plot(DM,beta2,'o')"""

#================ gamma2 ================= not found yet
"""plt.figure()
plt.plot(DM,gamma2,'o')"""
