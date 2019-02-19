import numpy as np
import numpy.ma as ma
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
from Functions import *
import SDSS_Plots
from colossus.cosmology import cosmology
from itertools import cycle
from statsmodels.stats.weightstats import DescrStatsW
from scipy import optimize
from scipy.interpolate import interp1d
cosmology.setCosmology("planck15")
Cosmo =cosmology.getCurrent()
h = Cosmo.h
h_3 = h*h*h
Add_SDSS = SDSS_Plots.SDSS_Plots(11.5,15,0.1) #pass this halomass:min, max, and binwidth for amting the SDSS plots


f_tau_str = "$f_{tdyn}$"

#set plot paramaters here
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 15})
mpl.rcParams.update({'lines.linewidth': 2})
mpl.rcParams.update({'lines.markersize': 5})
DPI = 200


#gets the HMF interpolation function
HMF_fun = Make_HMF_Interp()

#Abundance Matching Parameters
Override =\
{\
'M10':11.95,\
'SHMnorm10':0.03,\
'beta10':1.6,\
'gamma10':0.7,\
'M11':0.5,\
'SHMnorm11':-0.01,\
'beta11':-0.6,\
'gamma11':0.1\
}

AbnMtch =\
{\
'Behroozi13': False,\
'Behroozi18': False,\
'B18c':False,\
'B18t':False,\
'G18':True,\
'G18_notSE':False,\
'Lorenzo18':False,\
'Moster': False,\
'z_Evo':True,\
'Scatter': 0.11,\
'Override_0': False,\
'Override_z': False,\
'Override': Override,\
'PFT': True,\
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
'g_PFT3': False\
}

Paramaters = \
{\
'AbnMtch' : AbnMtch,\
'AltDynamicalTime': 1,\
'NormRnd': 0.5,\
'ContinuityEqn': True\
}

#macc/M0
Unevolved = {\
'gamma' : 0.22,\
'alpha' : -0.91,\
'beta' : 6,\
'omega' : 3,\
'a' : 1,\
}

HighRes = False

#HaloMass Limits and Bins
AnalyticHaloMass_min = 11.5; AnalyticHaloMass_max = 16.6
if HighRes:
    AnalyticHaloBin = 0.05
else:
    AnalyticHaloBin = 0.1
AHB_2 = AnalyticHaloBin*AnalyticHaloBin
AnalyticHaloMass = np.arange(AnalyticHaloMass_min + np.log10(h), AnalyticHaloMass_max + np.log10(h), AnalyticHaloBin)
#Units are Mvir h-1

usSHMF_Data = np.loadtxt("/data/pg1g15/Side_Projects/Analytic_DM_Model/Subhalos/Surviving_Subhalos1.0.dat")
HaloMassRange_us = usSHMF_Data[0,2:]


#Chi2 for fit
def RMS_Fun(Model_X, Model_Y, Data_X, Data_Y, Verbose = False):
    
    Data_X = Data_X[Data_Y >= -10]
    Data_Y = Data_Y[Data_Y >= -10]
    
    Model_Interp = interp1d(Model_X, Model_Y)
    Interp_Y = Model_Interp(Data_X[Data_X>np.min(Model_X)])
    if Verbose:
        print(Interp_Y)
        print(Data_Y)
        print("\n")
    RMS = np.sum(np.divide(np.power((Data_Y[Data_X>np.min(Model_X)]-Interp_Y),2.0), len(Interp_Y)))
    return RMS



#===============================FinalPlot_SMF ==================================
def SMF(PltName, Tdyn_Factors, Figure = None, Panel = None):
    
    if Figure == None:  
        Figure = plt
        Figure.figure(figsize=(8,6))
        """SDSS"""
        SDSS = Add_SDSS.SMF(Only_Sat = True)
        MultiPlot = False
        mpl.rcParams.update({'font.size': 15})
        mpl.rcParams.update({'lines.linewidth': 2})
        
    else:
        mpl.rcParams.update({'font.size': 15})
        mpl.rcParams.update({'lines.markersize': 5})
        mpl.rcParams.update({'lines.linewidth': 2})
        """SDSS"""
        if Panel == 1:
            SDSS = Add_SDSS.SMF(Figure = Figure, Return_Leg = True)
        else:
            SDSS = Add_SDSS.SMF(Figure = Figure)
        MultiPlot = True
    
    AvaHaloMass, AnalyticalModel_SMF, SMF_x = LoadData_3(Tdyn_Factors)
    lines = ["--","-.", "-", ":"]
    linecycler = cycle(lines)
    colours = ['darkred', 'maroon', 'red', 'crimson']
    colourcycler = cycle(colours)
    lines2 = ["--","-."]
    linecycler2 = cycle(lines2)

    
    """Analytic"""
    SMF_Bin = SMF_x[1] - SMF_x[0]
    lines_plotted = []

    RMS_OTP = []
    for j, SMF_ in enumerate(AnalyticalModel_SMF):
        SMF_y_Sat  = np.full(len(SMF_x)+1, -300.)

        ColorParam = len(AnalyticalModel_SMF) - 1

        if ColorParam == 0:
            ColorParam = 1

        line, colour = next(linecycler), next(colourcycler)
        if PltName[-4:] == "Fig5":
            if Tdyn_Factors[j][3]:
                Label = "{}: {} + Evolution".format(f_tau_str, Tdyn_Factors[j][0])
            else:
                Label = "{}: {}".format(f_tau_str, Tdyn_Factors[j][0])
        elif PltName[-4:] == "Fig7":
            Label = "{}: = {}".format(f_tau_str, Tdyn_Factors[j][0])
        elif PltName[-5:] == "Fig11":
            if Tdyn_Factors[j][2]:
                if Tdyn_Factors[j][1]:
                    Label = "$f_{tdyn}$ = " + "{}".format(Tdyn_Factors[j][0]) + " + Starformation + Stripping"
                else:
                    Label = "$f_{tdyn}$ = " + "{}".format(Tdyn_Factors[j][0]) + " + Starformation"
            else:
                Label = "$f_{tdyn}$ = " + "{}".format(Tdyn_Factors[j][0])
        elif Tdyn_Factors[j][2]:
            Label = "{}, W13+F16".format(Tdyn_Factors[j][4])

        else:
            Label = "$f_{tdyn}$ = " + "{}".format(Tdyn_Factors[j][0])
        
        
        lines_plotted.append(Figure.plot(SMF_x, np.log10(SMF_), line, color = colour ,label = Label)[0])
       
        if SDSS != None:     
            RMS_OTP.append((Tdyn_Factors[j], RMS_Fun(SMF_x, np.log10(SMF_), SDSS[4], SDSS[5])))
    if len(RMS_OTP) > 0:
        with open("{}_RMS_SMF.dat".format(PltName), "w") as f:
            f.write("Factor, RMS\n")
            for i in RMS_OTP:
                f.write(str(i))
                f.write("\n")
        os.system("rm {}_RMS_SMF.pkl".format(PltName))
        pickle.dump(RMS_OTP, open('{}_RMS_SMF.pkl'.format(PltName), 'wb'))    
    
    
    if MultiPlot == False:
        
        Figure.xlabel("$log_{10}$ $M_*$ [$M_\odot$]", fontproperties = mpl.font_manager.FontProperties(size = 20))
        Figure.ylabel("$log_{10}  \phi$ $[Mpc^{-3} dex^{-1}]$", fontproperties = mpl.font_manager.FontProperties(size = 20))
        Figure.ylim (-7, -1.8)
        Figure.xlim(9, 12.2)
        Figure.legend(handles=lines_plotted, loc = 3, frameon = False)
        Figure.savefig("{}.png".format(PltName), dpi = DPI)
        Figure.savefig("{}.pdf".format(PltName), dpi = DPI)
        Figure.savefig("{}.eps".format(PltName), format = 'eps', dpi = DPI, bbox_inches='tight')
        Figure.clf()
        mpl.rcParams.update({'font.size': 13})
    else:
        Figure.legend(handles=lines_plotted, loc = 3, frameon = False)
        mpl.rcParams.update({'font.size': 13})
        mpl.rcParams.update({'lines.markersize': 5})
        return lines_plotted


#==========================FinalPlot_Frac (Fig 4 + 6)===========================
#Tdyn Changing
   
def Frac_Plot(PltName, Tdyn_Factors):
    mpl.rcParams.update({'lines.markersize': 5})
    mpl.rcParams.update({'lines.linewidth': 2})
    AvaHaloMass, AnalyticalModelFrac_, AnalyticalModelNoFrac_, SM_Cuts = LoadData_4_6(Tdyn_Factors)    
    
    SM_Cuts_Plt = [10.0, 10.5, 11] # use this to pick SM_Cut to plot
    
    #To Make 6 panel
    f_B, SubPlots_B = plt.subplots(2, len(SM_Cuts_Plt), figsize = (14,6))   
    #DynamicalTime Frac=======================================================
    f_F, SubPlots_F = plt.subplots(1, len(SM_Cuts_Plt), figsize = (13,4), sharey = True)

    lines = ["--","-.", "-", ":"]
    linecycler = cycle(lines)
    RMS_OTP = []
    SDSS_Panels_X, SDSS_Panels_Y = [], []
    for i, Factor in enumerate(Tdyn_Factors):
        linestyle = next(linecycler)
        for j, Cut in enumerate(SM_Cuts_Plt):
            if i == 0:
                X_SDSS, Y_SDSS = Add_SDSS.FracPlot(SubPlots_F[j], SM_Cut = Cut)
                SDSS_Panels_Y.append(Y_SDSS)
                SDSS_Panels_X.append(X_SDSS)
                X_SDSS, Y_SDSS = Add_SDSS.FracPlot(SubPlots_B[1][j], SM_Cut = Cut)
                SubPlots_F[j].set_title("$M_{*, sat} > 10^{%s}$" %(Cut))
                SubPlots_B[0][j].set_title("$M_{*, sat} > 10^{%s}$" %(Cut))
            Data_ix = np.digitize(Cut, bins = SM_Cuts)-1
            X_Bin = np.digitize(X_SDSS[0], AvaHaloMass[0]- np.log10(h))            
            Y_Model = np.divide(AnalyticalModelFrac_[i][Data_ix],AnalyticHaloBin)
            #Add Scatter on DM
            X, Y = Gauss_Scatt(AvaHaloMass[0][X_Bin:]- np.log10(h), Y_Model[X_Bin:], Scatt = 0.1)
            
            RMS_OTP.append((Factor, Cut, RMS_Fun(X, Y, SDSS_Panels_X[j], SDSS_Panels_Y[j])))
                        
            if PltName[-4:] == "Fig6":
                if Factor[3]:
                    Label = "{}: {} + Evolution".format(f_tau_str, Factor[0])
                else:
                    Label = "{}: {}".format(f_tau_str, Factor[0])
            elif PltName[-4:] == "Fig8":
                Label = "{}: = {}".format(f_tau_str, Factor[0])
            elif PltName[-5:] == "Fig12":
                if Factor[2]:
                    if Factor[1]:
                        Label = "$f_{tdyn}$" + "{}".format(Factor[0]) + " + Starformation + Stripping"
                    else:
                        Label = "$f_{tdyn}$" + "{}".format(Factor[0]) + " + Starformation"
                else:
                    Label = "$f_{tdyn}" + "{}".format(Factor[0])
            else:
                Label = "$f_{tdyn}$ = " + "{}".format(Factor[0])
            
            SubPlots_F[j].plot(X-0.1, Y, linestyle, label = Label)
            SubPlots_B[1][j].plot(X-0.1, Y, linestyle, label = Label)

    lgd_F = SubPlots_F[len(SM_Cuts_Plt)-1].legend(bbox_to_anchor=(1.05, 1), loc = 2, frameon = False)
    SubPlots_F[1].set_xlabel("$log_{10}$ $M_h$ [$M_\odot$]", fontproperties = mpl.font_manager.FontProperties(size = 15))
    SubPlots_F[0].set_ylabel("$Fraction$ [$dex^{-1}$]", fontproperties = mpl.font_manager.FontProperties(size = 15))
    f_F.savefig("{}a.png".format(PltName), bbox_extra_artists=(lgd_F,), bbox_inches='tight')
    f_F.savefig("{}a.pdf".format(PltName), bbox_extra_artists=(lgd_F,), bbox_inches='tight')
    f_F.savefig("{}.eps".format(PltName), format = 'eps', dpi = DPI, bbox_inches='tight')
    f_F.clf()


    os.system("rm {}_RMS_Frac.dat".format(PltName))
    with open("{}_RMS_Frac.dat".format(PltName), "w") as f:
        f.write("Factor, RMS\n")
        for i in RMS_OTP:
            f.write(str(i))
            f.write("\n")
    os.system("rm {}_RMS_Frac.pkl".format(PltName))
    pickle.dump(RMS_OTP, open('{}_RMS_Frac.pkl'.format(PltName), 'wb'))
    #===========================================================================
            
    #DynamicalTime NoFrac=======================================================  
    mpl.rcParams.update({'lines.linewidth': 2})
    f_NF, SubPlots_NF = plt.subplots(1, len(SM_Cuts_Plt), figsize = (13,4), sharey = True)

    lines = ["--","-.", "-", ":"]
    linecycler = cycle(lines)
    KS_OTP = []
    RMS_OTP = []
    SDSS_Panels_X, SDSS_Panels_Y = [], []
    for i, Factor in enumerate(Tdyn_Factors):
        linestyle = next(linecycler)
        for j, Cut in enumerate(SM_Cuts_Plt):
            if i == 0:
                X_SDSS, Y_SDSS = Add_SDSS.NoFracPlot(SubPlots_NF[j], SM_Cut = Cut)
                SDSS_Panels_Y.append(Y_SDSS)
                SDSS_Panels_X.append(X_SDSS)
                X_SDSS, Y_SDSS = Add_SDSS.NoFracPlot(SubPlots_B[0][j], SM_Cut = Cut)
                SubPlots_NF[j].set_title("$M_{*, sat} > 10^{%s}$" %(Cut))
            
            Data_ix = np.digitize(Cut, bins = SM_Cuts)-1
            X_Bin = np.digitize(X_SDSS[0], AvaHaloMass[0]- np.log10(h))
            Y_Model = np.divide(AnalyticalModelNoFrac_[i][Data_ix], AnalyticHaloBin)
            
            X, Y = Gauss_Scatt(AvaHaloMass[0][X_Bin:]- np.log10(h), Y_Model[X_Bin:], Scatt = 0.1)
            
            if Cut == 11:
                Bin = np.digitize(12.5, X_SDSS)
            else:
                Bin = np.digitize(12.0, X_SDSS)
            RMS_OTP.append((Factor, Cut, RMS_Fun(X, np.log10(Y), SDSS_Panels_X[j], SDSS_Panels_Y[j])))
            
            if PltName[-4:] == "Fig6":
                if Factor[3]:
                    Label = "{}: {} + Evolution".format(f_tau_str, Factor[0])
                else:
                    Label = "{}: {}".format(f_tau_str, Factor[0])
            elif PltName[-4:] == "Fig8":
                Label = "{}: = {}".format(f_tau_str, Factor[0])
            elif PltName[-5:] == "Fig12":
                if Factor[2]:
                    if Factor[1]:
                        Label = "$f_{tdyn}$" + "{}".format(Factor[0]) + " + Starformation + Stripping"
                    else:
                        Label = "$f_{tdyn}$" + "{}".format(Factor[0]) + " + Starformation"
                else:
                    Label = "$f_{tdyn}$" + "{}".format(Factor[0])
            else:
                Label = "$f_{tdyn}$ = " + "{}".format(Factor[0])
            
                       
            SubPlots_NF[j].plot(X-0.1, np.log10(Y), linestyle, label  = Label)
            SubPlots_B[0][j].plot(X-0.1, np.log10(Y), linestyle, label  = Label)

    SubPlots_NF[0].set_ylim(-7, -1)
    SubPlots_NF[1].set_ylim(-7, -1)
    SubPlots_NF[2].set_ylim(-7, -1)
    lgd_NF = SubPlots_NF[len(SM_Cuts_Plt)-1].legend(bbox_to_anchor=(1.05, 1), loc = 2, frameon = False)
    SubPlots_NF[1].set_xlabel("$log_{10}$ $M_h$ [$M_\odot$]", fontproperties = mpl.font_manager.FontProperties(size = 15))
    SubPlots_NF[0].set_ylabel("$log_{10} \phi$ $[Mpc^{-3} dex^{-1}]$", fontproperties = mpl.font_manager.FontProperties(size = 15))
    f_NF.savefig("{}b.png".format(PltName), bbox_extra_artists=(lgd_NF,), bbox_inches='tight')
    f_NF.savefig("{}b.pdf".format(PltName), bbox_extra_artists=(lgd_NF,), bbox_inches='tight')
    plt.savefig("{}.eps".format(PltName), format = 'eps', dpi = DPI, bbox_extra_artists=(lgd_NF,), bbox_inches='tight')
    f_NF.clf()
    

    os.system("rm {}_RMS_NoFrac.dat".format(PltName))
    with open("{}_RMS_NoFrac.dat".format(PltName), "w") as f:
        f.write("Factor, RMS\n")
        for i in RMS_OTP:
            f.write(str(i))
            f.write("\n")
    os.system("rm {}_RMS_NoFrac.pkl".format(PltName))
    pickle.dump(RMS_OTP, open('{}_RMS_NoFrac.pkl'.format(PltName), 'wb'))
    #===========================================================================
    SubPlots_B[1][0].set_ylabel("$Fraction$ [$dex^{-1}$]")
    SubPlots_B[0][0].set_ylim(-6, -2)
    SubPlots_B[0][1].set_ylim(-6, -2)
    SubPlots_B[0][2].set_ylim(-6, -2)
    SubPlots_B[0][0].set_xlim(11.5, 15.5)
    SubPlots_B[0][1].set_xlim(11.5, 15.5)
    SubPlots_B[0][2].set_xlim(11.5, 15.5)
    SubPlots_B[1][0].set_xlim(11.5, 15.5)
    SubPlots_B[1][1].set_xlim(11.5, 15.5)
    SubPlots_B[1][2].set_xlim(11.5, 15.5)
    lgd_B = SubPlots_B[0][len(SM_Cuts_Plt)-1].legend(loc = 2, fontsize = 10, frameon = False)
    SubPlots_B[1][1].set_xlabel("$log_{10}$ $M_{h, cent}$ [$M_\odot$]", fontproperties = mpl.font_manager.FontProperties(size = 15))
    SubPlots_B[0][0].set_ylabel("$log_{10} \phi$ $[Mpc^{-3} dex^{-1}]$", fontproperties = mpl.font_manager.FontProperties(size = 15))
    f_B.savefig("{}.png".format(PltName), bbox_extra_artists=(lgd_B,), bbox_inches='tight')
    f_B.savefig("{}.pdf".format(PltName), bbox_extra_artists=(lgd_B,), bbox_inches='tight')
    f_B.savefig("{}.eps".format(PltName), format = 'eps', dpi = DPI,bbox_extra_artists=(lgd_B,), bbox_inches='tight')
    f_B.clf()
    plt.clf()
    #6 Panel End=================================================================
    mpl.rcParams.update({'font.size': 13})                                      


#========================Sat_SMF_In HMF_Bin (Fig 9)=============================
def Sat_SMF_In_HMF(PltName, Tdyn_Factors):
    mpl.rcParams.update({'lines.linewidth': 2})
    AvaHaloMass, AnalyticalModel_SMF, SMF_x = LoadData_10(Tdyn_Factors)
    Min, Max, Bin = 12., 15., 1.0
    f, SubPlots = plt.subplots(len(Tdyn_Factors), 2, figsize = (12,4*len(Tdyn_Factors)), sharey = True, sharex = True)

    for i, Factor in enumerate(Tdyn_Factors):
        Add_SDSS.SMF_Fixed_HM(SubPlots[i,0], SubPlots[i,1], Min, Max, Bin)
        SMF_Arr = AnalyticalModel_SMF[i]
        for j, HM_Bin in enumerate(np.arange(Min, Max, Bin)):
            HM_Mask = ma.masked_inside(AvaHaloMass[0], HM_Bin, Max).mask#HM_Bin+0.5).mask
            Tot_In_Bin = np.sum(SMF_Arr[HM_Mask], axis = 0)
            SubPlots[i][0].plot(SMF_x[:-1], np.log10(Tot_In_Bin), '--C{}'.format(j), label = "Analytic: {}<HM<{}".format(HM_Bin, HM_Bin+0.5))
            HM_Mask = ma.masked_inside(AvaHaloMass[0], HM_Bin, HM_Bin+0.5).mask
            Tot_In_Bin = np.sum(SMF_Arr[HM_Mask], axis = 0)
            SubPlots[i][1].plot(SMF_x[:-1], np.log10(Tot_In_Bin), '--C{}'.format(j), label = "Analytic: {}<HM<{}".format(HM_Bin, HM_Bin+0.5))
            SubPlots[i][0].set_title("{}".format(Factor))

    SubPlots[0,0].set_ylim(-6, -2)
    SubPlots[0,0].set_xlim(10, 12)
    SubPlots[0, 1].legend(bbox_to_anchor=(1.05, 1), loc = 2)


    plt.savefig("{}.png".format(PltName), bbox_inches='tight', dpi = DPI)
    plt.savefig("{}.pdf".format(PltName), bbox_inches='tight', dpi = DPI)
    plt.savefig("{}.eps".format(PltName), format = 'eps', dpi = DPI, bbox_inches='tight')
    plt.clf()

        
#========================sSFR========================================    
def sSFR_Plot(PltName, Tdyn_Factors):
    mpl.rcParams.update({'lines.linewidth': 2})
    f, SubPlots = plt.subplots(1, 3, figsize = (10,3), sharey = True)
    FirstPass = True
    No_Leg = False
    lines = ["--", "-.", "-","-", "--","-."]
    colours = ["C3", "C0", "C1", "C2", "C5", "k"]
    linecycler = cycle(lines)
    colourcycler = cycle(colours)
    x,y=0,0
    for i, Factor in enumerate(Tdyn_Factors): 
        Line = next(linecycler)
        Colour = next(colourcycler)
        Surviving_Sat_SMF_MassRange, sSFR_Range, Satellite_sSFR = LoadData_sSFR(Factor)
        bin_w = Surviving_Sat_SMF_MassRange[1]-Surviving_Sat_SMF_MassRange[0]
        x = 0
        for l,u in [(10,10.5),(10.5,11),(11,12)]:
            Weights = np.sum(Satellite_sSFR[np.digitize(l, bins = Surviving_Sat_SMF_MassRange):np.digitize(u, bins = Surviving_Sat_SMF_MassRange)], axis = 0)
            N_Ntot = Weights/(np.sum(Weights)*bin_w)
            SubPlots[x].set_title("{}-{}".format(l,u) + "$M_{*, sat}$")
            if FirstPass == True:    
                if i == 0 and x == 2:
                    No_Leg = False
                else:
                    No_Leg = True
                A = Add_SDSS.sSFR_Plot(l, u, SubPlots[x], No_Leg = No_Leg)
                B = Add_SDSS.sSFR_Plot_Cen(l, u, SubPlots[x], No_Leg = No_Leg)
            if x==2:
                if Tdyn_Factors[i][2]:
                    Label = "{}, W13+F16".format(Tdyn_Factors[i][4])
                    
                else:
                    Label = "{}".format(Tdyn_Factors[i])
                SubPlots[x].plot(sSFR_Range, N_Ntot, Line, color = Colour,label = Label, alpha = 0.75)
            else:
                SubPlots[x].plot(sSFR_Range, N_Ntot, Line, color = Colour,alpha = 0.75)
            x +=1
        FirstPass = False
        
    SubPlots[0].set_xlim(-13, -9.0)
    SubPlots[1].set_xlim(SubPlots[0].get_xlim())
    SubPlots[2].set_xlim(SubPlots[0].get_xlim())
    mpl.rcParams.update({'font.size': 11})
    SubPlots[2].legend(loc = 1, frameon = False)
    mpl.rcParams.update({'font.size': 11})
    
    SubPlots[1].set_xlabel("log10(sSFR) [$yr^{-1}$]", fontproperties = mpl.font_manager.FontProperties(size = 15))
    #print(ax.get_xticks())  
    SubPlots[0].set_xticklabels(["-13", "-12", "-11", "-10", ""])
    SubPlots[1].set_xticklabels(["-13", "-12", "-11", "-10", ""])
    SubPlots[2].set_xticklabels(["-13", "-12", "-11", "-10", "-9"])
    
    #when you come looking for how to do this
    #ticks = plt.xticks(); plt.xticks(ticks[:-1])
    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    #tik.TickHelper
    #plt.tight_layout()
    plt.savefig("{}.png".format(PltName), bbox_inches='tight', dpi = DPI)
    plt.savefig("{}.pdf".format(PltName), bbox_inches='tight', dpi = DPI)   
    plt.savefig("{}.eps".format(PltName), format = 'eps', dpi = DPI, bbox_inches='tight')
        
#============================Run Plots===============================
if __name__ == "__main__":
    if False:
        #Custom run
        #Create list of dynamicaltimes to do Fit with
        tdyn_in = 1.0
        Tdyn_Factors = []
        tdyn = 1.0
        Tdyn_Factors = [(1.0, False, False, True, True, 'G18')]
        #tdyn = 1.2
        #Tdyn_Factors += [(tdyn, False, False, True), (tdyn, False, True, True, 6), (tdyn, True, True, True, 6)]
        #plots
        SMF("./Figures/Test/Test_SMF", Tdyn_Factors)
        Frac_Plot("./Figures/Test/Test_Frac", Tdyn_Factors)
        if len(Tdyn_Factors) == 1:
            Tdyn_Factors.append(Tdyn_Factors[0])
        Sat_SMF_In_HMF("./Figures/Test/SMF_Sat_ByHM_Test", Tdyn_Factors)
        sSFR_Plot("./Figures/Test/sSFR_Test", Tdyn_Factors)
        print("Test Done")
    
    PaperPlots = False
    
    if PaperPlots:
        #Zevo true/false and tdyn 1/inf
        Tdyn_Factors = [(1.0, False, False, True, True, 'G18'), (np.inf, False, False, True, True, 'G18'), (1.0, False, False, False, True, 'G18'), (np.inf, False, False, False, True, 'G18')]
        SMF("./Figures/Evo_Inf/Fig5", Tdyn_Factors)
        Frac_Plot("./Figures/Evo_Inf/Fig6", Tdyn_Factors)
        Sat_SMF_In_HMF("./Figures/Evo_Inf/SMF_Sat_ByHM_Evo_Inf", Tdyn_Factors)
        print("EvoInf Done")
    
    if PaperPlots:
        #Full Run of DynamicalTimes
        Tdyn_Factors = [(0.5, False, False, True, True, 'G18'), (1.0, False, False, True, True, 'G18'), (2.5, False, False, True, True, 'G18')]
        SMF("./Figures/Tdyn/Fig7", Tdyn_Factors)
        Frac_Plot("./Figures/Tdyn/Fig8", Tdyn_Factors)
        Sat_SMF_In_HMF("./Figures/Tdyn/SMF_Sat_ByHM_Tdyn", Tdyn_Factors)  
        print("Tdyn Done")
            
    if True:
        #ivestigating the effects of starformation and quenching
        tdyn = 1.0     
        #Tdyn_Factors = [(tdyn_in , False, True, True, False, 'G18'), (tdyn_in , False, True, True, True, 'G18')]
        Tdyn_Factors = [(1.0, True, True, True, 'T16', 'G18'), (1.0, True, True, True, 'S16', 'G18'), (1.0, True, True, True, 'CE', 'G18')]
        SMF("./Figures/SF_Q/Fig9", Tdyn_Factors)
        sSFR_Plot("./Figures/SF_Q/Fig10", Tdyn_Factors)
        print("SF_Q done")
        
    if True:
        #tdyn set with the stripping and SF
        tdyn = 1.0
        #Tdyn_Factors = [(tdyn, False, False, True, True, 'G18'), (tdyn, False, True, True, True, 'G18'), (tdyn, True, True, True, True, 'G18')]
        Tdyn_Factors = [(1.0, True, True, True, 'T16', 'G18'), (1.0, True, True, True, 'S16', 'G18'), (1.0, True, True, True, 'CE', 'G18')]
        
        #tdyn = 1.2
        #Tdyn_Factors += [(tdyn, False, False, True), (tdyn, False, True, True, 6), (tdyn, True, True, True, 6)]
        SMF("./Figures/StripSF/Fig11", Tdyn_Factors)
        Frac_Plot("./Figures/StripSF/Fig12", Tdyn_Factors)
        Sat_SMF_In_HMF("./Figures/StripSF/SMF_Sat_ByHM_StripSF_{}".format(tdyn), Tdyn_Factors)
        print("Strip SF Done")
    