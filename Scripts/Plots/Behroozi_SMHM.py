import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

plt.ion()


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
    
def identify_idx_HM(HM_desired, HM):
    if HM_desired <= np.min(HM):
        HM_idx=0
    elif HM_desired > np.min(HM) and HM_desired <= np.max(HM):
        for i in range(0, HM.size):
            if HM[i]>HM_desired:
                HM_idx=i
                break
        HM2=HM[HM_idx]; HM1=HM[HM_idx-1]
        diff1=np.abs(HM1-HM_desired)
        diff2=np.abs(HM2-HM_desired)
        if diff1<=diff2:
            HM_idx=HM_idx-1
    elif HM_desired > np.max(HM):
        Err_msg = 'The desired value {} in HM_desired is larger than the maximum value in HM!\nCannot identify HM index!'.format(HM_desired)
        sys.exit(Err_msg)
    return HM_idx

def update_progress(job_title, progress):
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}% ".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += "DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


z_vdB = np.loadtxt('Redshift_array.txt')
HM_vdB = np.loadtxt('HM_vdB.dat') #[log(Msun)]
HM_vdB_bin = np.zeros(z_vdB.size)
for i in range(0, z_vdB.size):
    HM_vdB_bin[i] = (HM_vdB[i,1] - HM_vdB[i,0]) #[log(Msun)]

with open ("Behroozi_file_list.txt", "r") as my_file:
    file_list = my_file.readlines()

file_list.reverse()
a = np.zeros(len(file_list)-1)
for i in range(0, len(file_list)-1):
    a[i] = float(file_list[i][18:26])
#a = a[::-1]
z = 1/a - 1

z_desired = np.arange(0.2, 4, 0.2)
idx_z = []
idx_z_vdB = []
for i in range(0, z_desired.size):
    update_progress("Calculating redshift index", i/z_desired.size)
    idx_z.append(identify_idx_z(z_desired[i], z))
    idx_z_vdB.append(identify_idx_z(z_desired[i], z_vdB))
update_progress("Calculating redshift index", 1)

"""plt.figure()
plt.semilogx(z_desired, np.repeat(1,z_desired.size), '.')
plt.semilogx(z[idx_z], np.repeat(2,z[idx_z].size), '.')
plt.semilogx(z_vdB[idx_z_vdB], np.repeat(3,z_vdB[idx_z_vdB].size), '.')
plt.ylim(-2,6)"""

"""plt.figure()
plt.plot(np.repeat(1,HM_vdB[0,:].size), HM_vdB[0,:], '.')"""

HM = []
SM = []

k=0
for i in idx_z:
    update_progress("Reading data from Behroozi catalog", (k)/len(idx_z))
    file_name = '../../Data/Observational/Behroozi_catalog/' + file_list[i][6:30]
    """if i==idx_z[1]:
        break"""
    #HM.append(np.log10(np.loadtxt(file_name, usecols=11))) #[log(Msun)]
    #SM.append(np.log10(np.loadtxt(file_name, usecols=20))) #[log(Msun)]
    x = pd.read_csv(file_name, skiprows=np.arange(0,30,1), sep=' ', usecols=['M','SM'])
    HM.append(np.log10(x.loc[:,'M'].values))
    SM.append(np.log10(x.loc[:,'SM'].values))
    k+=1
update_progress("Reading data from Behroozi catalog", 1)

SM_to_plot = np.zeros((z_desired.size, HM_vdB[0,:].size))
for i in range(0, len(idx_z)):
    #update_progress("Calculating Stellar mass median", i/len(idx_z))
    for j in range(0, HM_vdB[idx_z[i],:].size):
        update_progress("Calculating SM median N. {}/{}".format(i+1,len(idx_z)), j/HM_vdB[idx_z[i],:].size)
        HM_low = HM_vdB[idx_z[i],j] - HM_vdB_bin[idx_z_vdB[i]]/2. #[log(Msun)]
        HM_upp = HM_vdB[idx_z[i],j] + HM_vdB_bin[idx_z_vdB[i]]/2. #[log(Msun)]
        idx_list = []
        for k in range(0, HM[i].size):
            if HM[i][k] > HM_low and HM[i][k] < HM_upp:
                idx_list.append(k)
        if not idx_list:
            SM_to_plot[i,j] = 0.
        else:
            SM_to_plot[i,j] = np.median(SM[i][idx_list]) #[log(Msun)]
        """if j==48:
            break"""
    update_progress("Calculating SM median N. {}/{}".format(i+1,len(idx_z)), 1)
    """if i==0:
        break"""
#update_progress("Calculating Stellar mass median", 1)
np.savetxt('Behroozi_SM_to_plot.txt', SM_to_plot)
