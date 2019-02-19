import numpy as np
Leja_in = np.loadtxt("fig6_pars.txt")
z_l = Leja_in[:,0]
b_l = Leja_in[:,1]
a1_l = Leja_in[:,2]
a2_l = Leja_in[:,3]

x = z_l
y = a1_l
A1 = np.polyfit(x,y, 3)
print("A1")
print(A1)
A1_m = np.polyfit([0.4, 2.3],[-2.5,0.5], 1)
print("A1_m")
print(A1_m)
y = a2_l
A2 = np.polyfit(x, y, 1)
print("A2")
print(A2)
y= b_l
B = np.polyfit(x, y, 3)
print("B")
print(B)
def Alpha1(x):
    y = A1[3] +  (A1[2]*x) + (A1[1]*x**2) + (A1[0]*x**3)
    return y
def Alpha1_merger(x):
    y = A1_m[1] +  (A1_m[0]*x)
    return y
def Alpha2(x):
    y = A2[1]+A2[0]*x
    return y
def Beta(x):
    y = B[3] + (B[2]*x) + (B[1]*x**2) + (B[0]*x**3)
    return y 

def SFR_t_corr(M_out, z = 0.5):
    s0 = 0.195 + 1.157*(z) - 0.143*(z**2)
    if z < 0.5:
        logM0 = 9.244 + 0.753*(0.5) - 0.09*(0.5**2)
    else:
        logM0 = 9.244 + 0.753*(z) - 0.09*(z**2)
    Gamma = -1.118 #including -ve here to avoid it later
    log10MperY = s0 - np.log10(1 + np.power(np.power(10, (M_out - logM0) ), Gamma))
    return log10MperY

def SFR_t_ten(M_out, z = 0.5):
    s0 = 0.195 + 1.157*(z) - 0.143*(z**2)
    logM0 = 10.5
    Gamma = -1.118 #including -ve here to avoid it later
    log10MperY = s0 - np.log10(1 + np.power(np.power(10, (M_out - logM0) ), Gamma))
    return log10MperY


def SFR_t(M_out, z = 0.5):
    s0 = 0.195 + 1.157*(z) - 0.143*(z**2)
    logM0 = 9.244 + 0.753*(z) - 0.09*(z**2)
    Gamma = -1.118 #including -ve here to avoid it later
    log10MperY = s0 - np.log10(1 + np.power(np.power(10, (M_out - logM0) ), Gamma))
    return log10MperY

#Leija15
def Leija15(M,z):
    A = np.full_like(M, 0.0)#np.full_like(M, Alpha1(z))
    A[M<10.5] = np.full_like(M[M<10.5], Alpha2(z))
    B = Beta(z)
    log10MperY = A*(M-10.5) + B
    return log10MperY#np.power(10, log10MperY)  
def Leija15_mergers(M,z):
    A = np.full_like(M, Alpha1_merger(z))
    A[M<10.5] = np.full_like(M[M<10.5], Alpha2(z))
    B = Beta(z)
    log10MperY = A*(M-10.5) + B
    return log10MperY#np.power(10, log10MperY)  

def SFR_t_fit(M_out, z = 0.5):
    s0 = 0.5 + 1.157*(z) - 0.143*(z**2)
    logM0 = 10.5 + 0.753*(z) - 0.15*(z**2)
    Gamma = -(1.2 - 0.02*(z) - 0.02*(z**2))#including -ve here to avoid it later
    log10MperY = s0 - np.log10(1 + np.power(np.power(10, (M_out - logM0) ), Gamma))
    return log10MperY