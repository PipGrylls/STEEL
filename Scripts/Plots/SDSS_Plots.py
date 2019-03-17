import os
import sys
AbsPath = str(__file__)[:-len("/SDSS_Plots.py")]+"/../.."
sys.path.append(AbsPath)
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
import colossus.halo.mass_adv as massdefs
from colossus.cosmology import cosmology
cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()
h = Cosmo.h
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'lines.linewidth': 2})
mpl.rcParams.update({'lines.markersize': 5})

class SDSS_Plots:

    def __init__(self, HM_min, HM_max, HM_bin, Photomotry = "SerExp"):
        self.AnalyticHaloMass_min = HM_min
        self.AnalyticHaloMass_max = HM_max
        self.AnalyticHaloBin = HM_bin
        self.AnalyticHaloMass = np.arange(self.AnalyticHaloMass_min, self.AnalyticHaloMass_max, self.AnalyticHaloBin)

        """Cutting SDSS Data"""
        
        #Bernardi File
        #Header = ["galcount","z","Vmaxwt","Msflagserexp","MsMendSerExp","MsMendSerExptrunc","btSE","logRkpcSEcirc","logRkpcBcirc","nSE","logRkpcD","flagserexp","logSapSE", "newLcentsat", "newMhaloL","newgroupL","newMhaloLSer", "newgroupLSer", "newLcentsatSer", "probe","probell","probs0","probsab","probscd" ]
        #Loads SDSS
        #df = pd.read_csv("./Bernardi_SDSS/sdds_Bellas_great.csv", header = None, names = Header)
        #fracper=0.98
        
        #Lorenzo File
        #Header = ["galcount", "z", "Vmaxwt", "MsMendSerExp", "AbsMag", "logReSerExp", "BT", "n_bulge", "newLcentsat", "NewMCentSat", "newMhaloL", "probaE", "probaEll", "probaS0", "probaSab", "probaScd", "TType", "AbsMagCent", "MsCent", "veldisp", "veldisperr"]
        #Loads SDSS
        #df = pd.read_csv("./Bernardi_SDSS/new_catalog_Lorenzo.dat", header = None, names = Header, delim_whitespace = True, skiprows = 1)
        #fracper=0.724
        #Lorenzo = True
        
        #Lorenzo File sSFR inc
        Header = ["galcount", "z", "Vmaxwt", "MsMendSerExp", "AbsMag", "logReSerExp", "BT", "n_bulge", "newLcentsat", "NewMCentSat", "newMhaloL", "probaE", "probaEll", "probaS0", "probaSab", "probaScd", "TType", "AbsMagCent", "MsCent", "veldisp", "veldisperr", "AbsModel_newKcorr", "LCentSat", "raSDSS7", "decSDSS7", "Z", "sSFR", "FLAGsSFR", "MEDIANsSFR", "P16sSFR", "P84sSRF", "SFR", "FLAGSFR", "MEDIANSFR", "P16SFR", "P84SRF", "RA_SDSS", "DEC_SDSS", "Z_2", "Seperation"]
        df = pd.read_csv(AbsPath +"/Data/Observational/Bernardi_SDSS/new_catalog_SFRs.dat", header = None, names = Header, delim_whitespace = True, skiprows = 1)
        df['SerExpsSFR'] = df.apply(lambda row: row.SFR - row.MsMendSerExp, axis = 1) # Make a sSFR colum using the sersic exp photomotry
        fracper=0.724
        Lorenzo = True
        
        #Adds the cModel column
        Header = ["galcount", "finalflag", "z", "Vmaxwt", "MsMendSerExp", "MsMendCmodel", "AbsMag", "logReSerExp", "BT", "n_bulge", "newLCentSat", "NewMCentSat", "newMhaloL", "probaE", "probaEll", "probaS0", "probaSab", "probaScd", "TType", "P_S0", "veldisp", "veldisperr", "raSDSS7", "decSDSS7"]
        df_new = pd.read_csv(AbsPath +"/Data/Observational/Lorenzo_SDSS/new_catalog_cModel.dat", header = None, names = Header, delim_whitespace = True, skiprows = 1)
        df_cut = df_new[["galcount","MsMendCmodel"]]
        df = pd.merge(df, df_cut, on="galcount", how="inner")
        
        
        #Clears NAN/ unsuable data        
        df_noNAN = df.dropna()
        if Lorenzo:
            pass
        else:
            df_noNAN = df_noNAN[df_noNAN.Msflagserexp == 0]
        df_noNAN = df_noNAN[df_noNAN.Vmaxwt > 0]
        if Photomotry == "SerExptrunc":
            self.Photomotry = "MsMendSerExptrunc"
            df_noNAN = df_noNAN[df_noNAN.MsMendSerExptrunc > 0]
        elif Photomotry == "MsMendCmodel":
            self.Photomotry = "MsMendCmodel"
            df_noNAN = df_noNAN[df_noNAN.MsMendCmodel > 0]
        else:
            #leave this last so we default to SerExp
            self.Photomotry = "MsMendSerExp"
            df_noNAN = df_noNAN[df_noNAN.MsMendSerExp > 0]
        
        #Chage Mass Def of Catlouge to vir
        #df_noNAN = df_noNAN[df_noNAN.newMhaloL > 0]
        #df_noNAN.newMhaloL = np.log10(massdefs.changeMassDefinitionCModel(M=np.power(10, df_noNAN.newMhaloL.values + np.log10(h)), z=0.1, mdef_in='180c', mdef_out='vir')[0]) - np.log10(h) +0.025 #+0.025 is correction to planc15 cosmology

        #Redshift Cut, making Cent and Sat DB
        self.df_z = df_noNAN[df_noNAN.z < 0.25]
        self.df_cent = self.df_z[self.df_z.newLcentsat == 1.0]
        if Lorenzo:            
            self.df_sat = self.df_z[self.df_z.newLcentsat == 0.0]
        else:
            self.df_sat = self.df_z[self.df_z.newLcentsat == 2.0]
        #just update massdef for satilites
        self.df_sat = self.df_sat[self.df_sat.newMhaloL > 0]
        if "sdssVir.npy" in os.listdir(path=AbsPath +"/Data/Observational/Bernardi_SDSS"):
            self.df_sat.insert(loc= len(Header), column = "mhVir", value = np.load(AbsPath +"/Data/Observational/Bernardi_SDSS/sdssVir.npy")+0.025) #+0.025 is correction to planc15 cosmology 
        else:
            self.df_sat.insert(loc= len(Header), column = "mhVir", value = np.log10(massdefs.changeMassDefinitionCModel(M=np.power(10, self.df_sat.newMhaloL.values + np.log10(h)), z=0.1, mdef_in='178c', mdef_out='vir')[0]) - np.log10(h)) 
            np.save(AbsPath +"/Data/Observational/Bernardi_SDSS/sdssVir", self.df_sat.mhVir)
        
        self.df_cent.loc[self.Photomotry] = self.df_cent[self.Photomotry] + 0.025
        self.df_sat.loc[self.Photomotry] = self.df_sat[self.Photomotry] + 0.025
        
        Data_mh_orig = np.array(self.df_sat.newMhaloL) 
        Data_vmax_orig = np.array(self.df_sat.Vmaxwt)
        Data_ms = np.array(self.df_sat[self.Photomotry])


        #From bernardi
        skycov=8000.
        self.fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)
        print("FRACKSKY=", self.fracsky)
   
    def SMF(self, SMF_BinWidth = 0.1, Figure = None, Return_Leg = False, Only_Sat = False, OverridePhoto = None):
        mpl.rcParams.update({'font.size': 10})
        mpl.rcParams.update({'lines.markersize': 5})
        mpl.rcParams.update({'lines.linewidth': 2})
        if Figure == None:
            Figure = plt
        if OverridePhoto != None:
            self.Photomotry = OverridePhoto
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)
        
        
        if not Only_Sat:
        
            SM = np.array(self.df_cent[self.Photomotry][(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
            Vmax = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])

            Weights = Vmax
            Weightsum = np.sum(Vmax)
            totVmax = Weightsum/self.fracsky

            hist_cent, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
            hist_cent_raw, edges = np.histogram(SM, bins = SMF_Bins, density  = False)
            Poss_Err_cen = np.vstack(((hist_cent)*((np.sqrt(hist_cent_raw)-1)/np.sqrt(hist_cent_raw)),(hist_cent)*((np.sqrt(hist_cent_raw)+1)/np.sqrt(hist_cent_raw))))
            Y = np.log10(np.divide(hist_cent, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
            Y_e = np.log10(np.divide(Poss_Err_cen, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
            Y_e[0], Y_e[1] = Y-Y_e[0], Y_e[1]-Y
            #Cent_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_cent, self.fracsky*SMF_BinWidth)), 'b-', label = "SDSS: Central")[0]
            Cent_Plot = Figure.errorbar(SMF_Bins[1:], Y, yerr = Y_e, fmt ='b^', label = "SDSS: Central", fillstyle = "none")

        SM = np.array(self.df_sat[self.Photomotry][(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_sat, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        hist_sat_raw, edges = np.histogram(SM, bins = SMF_Bins, density  = False)
        Poss_Err_sat = np.vstack(((hist_sat)*((np.sqrt(hist_sat_raw)-1)/np.sqrt(hist_sat_raw)),(hist_sat)*((np.sqrt(hist_sat_raw)+1)/np.sqrt(hist_sat_raw))))
        Y = np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_sat = np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_e = np.log10(np.divide(Poss_Err_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_e[0], Y_e[1] = Y-Y_e[0], Y_e[1]-Y
        #Sat_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)), 'r-', label = "SDSS: Satellite")[0]
        Sat_Plot = Figure.errorbar(SMF_Bins[1:], Y, yerr = Y_e, fmt ='rs', label = "SDSS: Satellite", fillstyle = "none")
        
        if not Only_Sat:
            SM = np.array(self.df_z[self.Photomotry][(SMF_LB < self.df_z[self.Photomotry]) & (self.df_z[self.Photomotry] < SMF_UB)])
            Vmax = np.array(self.df_z.Vmaxwt[(SMF_LB < self.df_z[self.Photomotry]) & (self.df_z[self.Photomotry] < SMF_UB)])

            Weights = Vmax
            Weightsum = np.sum(Vmax)
            totVmax = Weightsum/self.fracsky

            hist_tot, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
            hist_tot_raw, edges = np.histogram(SM, bins = SMF_Bins, density  = False)
            Poss_Err_tot = np.vstack(((hist_tot)*((np.sqrt(hist_tot_raw)-1)/np.sqrt(hist_tot_raw)),(hist_tot)*((np.sqrt(hist_tot_raw)+1)/np.sqrt(hist_tot_raw))))
            Y = np.log10(np.divide(hist_tot, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
            Y_e = np.log10(np.divide(Poss_Err_tot, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
            Y_e[0], Y_e[1] = Y-Y_e[0], Y_e[1]-Y
            #Tot_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_tot, self.fracsky*SMF_BinWidth)), 'g-', label = "SDSS: Total")[0]
            Tot_Plot = Figure.errorbar(SMF_Bins[1:], Y, yerr = Y_e, fmt ='kh', label = "SDSS: Total", fillstyle = "none")
        
        
        #SDSS_Legend = Figure.legend(handles = [Tot_Plot, Sat_Plot, Cent_Plot], loc = 1, frameon = False)
        if Figure == plt:
            if not Only_Sat:
                SDSS_Legend = Figure.legend(handles = [Tot_Plot, Sat_Plot, Cent_Plot], loc = 1, frameon = False)
                ax = plt.gca().add_artist(SDSS_Legend) 
                return Tot_Plot, Sat_Plot, Cent_Plot, ax, SMF_Bins[1:], Y_sat
            else:
                SDSS_Legend = Figure.legend(handles = [Sat_Plot], loc = 1, frameon = False)
                ax = plt.gca().add_artist(SDSS_Legend) 
                return None, Sat_Plot, None, ax, SMF_Bins[1:], Y_sat

        else:
            if Return_Leg == False:
                return None
            else:
                if not Only_Sat:
                    SDSS_Legend = Figure.legend(handles = [Tot_Plot, Sat_Plot, Cent_Plot], loc = 1, frameon = False)
                    ax = Figure.add_artist(SDSS_Legend)
                    return Tot_Plot, Sat_Plot, Cent_Plot, ax, SMF_Bins[1:], Y_sat
                else:
                    SDSS_Legend = Figure.legend(handles = [Sat_Plot], loc = 1, frameon = False)
                    ax = Figure.add_artist(SDSS_Legend)
                    return None, Sat_Plot, None, ax, SMF_Bins[1:], Y_sat                
    
    def FracPlot(self, fig, SM_Cut = 10):
        #ActualCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat[self.Photomotry] > SM_Cut)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y, X = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = True)

        #UpperCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat[self.Photomotry] > SM_Cut + 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_U, X_U = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = True)

        #Lowercut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat[self.Photomotry] > SM_Cut - 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_L, X_L = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = True)

        Yplot_U = np.maximum(Y_U, Y_L) - Y
        Yplot_L = Y - np.minimum(Y_U, Y_L)
        Y_Err = np.vstack((Yplot_L, Yplot_U))

        fig.fill_between(X[1:], Y - Y_Err[0], Y + Y_Err[1], alpha = 0.5, color = 'tab:gray')
        return X[1:], Y
    
    def NoFracPlot(self, fig, SM_Cut = 10):
        #ActualCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat[self.Photomotry] > SM_Cut)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y, X = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = False)
        
        
        #UpperCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat[self.Photomotry] > SM_Cut + 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_U, X_U = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = False)
        
        #Lowercut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat[self.Photomotry] > SM_Cut - 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_L, X_L = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = False)
        
        #Combining to form range
        Yplot_U = (np.maximum(Y_U, Y_L) - Y)*0.9195 #0.9195 correction of volume to Planck15
        Yplot_L = (Y - np.minimum(Y_U, Y_L))*0.9195 #0.9195 correction of volume to Planck15
        Y_Err = np.vstack((Yplot_L, Yplot_U))

        fig.fill_between(X[1:], np.log10(np.divide(Y - Y_Err[0], self.fracsky*self.AnalyticHaloBin)), np.log10(np.divide(Y + Y_Err[1], self.fracsky*self.AnalyticHaloBin)), alpha = 0.5, color = 'tab:gray')
        return X[1:], np.log10(np.divide(Y,self.fracsky*self.AnalyticHaloBin)*0.9195) #0.9195 correction of volume to Planck15

    
    
    def EllipticalPlot(self):
        Cent_SM = np.array(self.df_cent[self.Photomotry][self.df_cent[self.Photomotry] > 10])
        #Cent_PE = np.array(self.df_cent.probell[self.df_cent[self.Photomotry] > 10])
        Cent_PE = np.array(self.df_cent.probe[self.df_cent[self.Photomotry] > 10] )#+ self.df_cent.probell[self.df_cent[self.Photomotry] > 10])
        Cent_Vmax = np.array(self.df_cent.Vmaxwt[self.df_cent[self.Photomotry] > 10])
        print(np.max(Cent_PE), np.min(Cent_PE))
        Cent_PE[Cent_PE < 0] = 0

        X_Plot = np.arange(10, 12.5, 0.1)
        Bins = np.digitize(Cent_SM, X_Plot)

        Y_Plot = []
        for i in range(len(X_Plot)):
            Cent_Vmax[Bins == i] = np.divide(Cent_Vmax[Bins == i ],np.sum(Cent_Vmax[Bins == i]))
            Y_Plot.append(np.sum(Cent_PE[Bins==i]*Cent_Vmax[Bins == i]))

        #Y_Plot = np.array([ np.sum(Cent_PE[Bins == i] * (Cent_Vmax[Bins == i]/np.sum(Cent_Vmax[Bins == i])))/np.size(Cent_PE[Bins == i]) for i in range(len(X_Plot))])
        #Y_Plot = np.array([ (np.sum(Cent_PE[Bins == i] * Cent_Vmax[Bins == i])/(np.sum(Cent_Vmax[Bins == i]))) for i in range(len(X_Plot))])

        #Y_Plot = np.array([ (np.sum(Cent_PE[Bins == i])/(np.size(Cent_PE[Bins == i]))) for i in range(len(X_Plot))])
        """
        Y, X = np.histogram(Cent_SM, bins = X_Plot, weights = Cent_PE*Cent_Vmax)
        Bins = np.digitize(Cent_SM, X)
        Div = np.array([(np.sum(Cent_Vmax[Bins == i])) for i in range(1,len(X))])
        Y = np.divide(Y,Div)
        """

        plot = plt.plot(X_Plot[1:], Y_Plot[1:], "x", label = "P(E) (SDSS)")
        return plot
        #plot = plt.plot(X[1:], Y)

    def SMF_Elliptical(self):
        SMF_BinWidth = 0.1
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)

        SM = np.array(self.df_cent[self.Photomotry][(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        Prob_E = np.array(self.df_cent.probe[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        Prob_E[Prob_E < 0.5] = 0
        Prob_E[Prob_E > 0.5] = 1

        Weights = Vmax*Prob_E
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky


        hist, edges = np.histogram(SM, bins = SMF_Bins, weights = Weights)
        plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'rx', label = "SDSS:Elliptical")

        hist, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS:Total")

        return plot

    def SMF_Fixed_HM(self, fig1, fig2, Min, Max, Bin):
        SMF_BinWidth = 0.1
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)


        SM = np.array(self.df_sat[self.Photomotry][(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        HM = np.array(self.df_sat.newMhaloL[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])

        for i, HM_Bin in enumerate(np.arange(Min, Max, Bin)):
            HM_Mask = ma.masked_inside(HM, HM_Bin, Max).mask#HM_Bin+0.5).mask
            Weightsum = np.sum(Vmax[HM_Mask])
            totVmax = Weightsum/self.fracsky
            hist, edges = np.histogram(SM[HM_Mask], bins = SMF_Bins, weights = Vmax[HM_Mask])
            fig1.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'xC{}'.format(i), label = "SDSS: {}<HM<{}".format(HM_Bin, HM_Bin+0.5))

        for i, HM_Bin in enumerate(np.arange(Min, Max, Bin)):
            HM_Mask = ma.masked_inside(HM, HM_Bin, HM_Bin+0.5).mask
            Weightsum = np.sum(Vmax[HM_Mask])
            totVmax = Weightsum/self.fracsky
            hist, edges = np.histogram(SM[HM_Mask], bins = SMF_Bins, weights = Vmax[HM_Mask])
            fig2.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'xC{}'.format(i), label = "SDSS: {}<HM<{}".format(HM_Bin, HM_Bin+0.5))

        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky
        hist, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        #fig1.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS:Total")
        fig2.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS:Total")

    def SMHM_Sat_Cent(self, fig):
        HM = np.array(self.df_sat.newMhaloL)
        Bins = np.digitize(HM, self.AnalyticHaloMass)
        Y_Plot = []
        Yplot_Err = []
        for i in range(0,len(self.AnalyticHaloMass)):
            SM = np.array(self.df_sat[self.Photomotry])[Bins == i]
            Vmax = np.array(self.df_sat.Vmaxwt)[Bins == i]
            VmaxSum = np.sum(Vmax)
            weighted_stats = DescrStatsW(SM, weights=Vmax, ddof=0)
            Y_Plot.append(weighted_stats.mean)
            Yplot_Err.append(weighted_stats.std)
            """Y_Plot.append(np.mean(SM))
            Yplot_Err.append(np.std(SM))"""
        Y_Plot = np.array(Y_Plot)
        Yplot_Err = np.array(Yplot_Err)
        fig.fill_between(np.array(self.AnalyticHaloMass), Y_Plot + Yplot_Err, Y_Plot - Yplot_Err, color = "tab:gray", alpha = 0.5, label = "SDSS CentHalo")

    
    def Old_SMF(self, SMF_BinWidth = 0.1):
        SMF_LB = 9; SMF_UB = 13
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)



        logMstar_cent = np.array( self.df_cent[self.Photomotry][(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        Vmax_cent = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        VmaxSum = np.sum(Vmax_cent)
        totVmax=VmaxSum/self.fracsky

        #Bins and calculates VmaxWt per bin and number per bin
        Binned = np.digitize(logMstar_cent, SMF_Bins)
        Totals_Cent = (np.array([ np.divide(np.sum(Vmax_cent[Binned == i]), VmaxSum*SMF_BinWidth) for i in range (0, len(SMF_Bins)) ])[1:])*totVmax
        Totals_Cent_num = np.array([ np.size(Vmax_cent[Binned == i]) for i in range (0, len(SMF_Bins)) ])[1:]


        #possion err
        Totals_Cent_err = (np.divide(Totals_Cent,np.sqrt(Totals_Cent_num)))

        TotalsPlot_Cent = np.log10(np.vstack((Totals_Cent, Totals_Cent-Totals_Cent_err, Totals_Cent+Totals_Cent_err)))
        TotalsPlot_Cent[1:] = np.abs(TotalsPlot_Cent[1:]-TotalsPlot_Cent[0])
        Cent_fig = plt.errorbar(SMF_Bins[1:], TotalsPlot_Cent[0], yerr = TotalsPlot_Cent[1:], fmt ='bx', label = "Central")



        logMstar_Sat = np.array( self.df_sat[self.Photomotry][(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        Vmax_Sat = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        VmaxSum = np.sum(Vmax_Sat)
        totVmax=VmaxSum/self.fracsky

        #Bins and calculates VmaxWt per bin and number per bin
        Binned = np.digitize(logMstar_Sat, SMF_Bins)
        Totals_Sat = (np.array([ np.divide(np.sum(Vmax_Sat[Binned == i]), VmaxSum*SMF_BinWidth) for i in range (0, len(SMF_Bins)) ])[1:])*totVmax
        Totals_Sat_num = np.array([ np.size(Vmax_Sat[Binned == i]) for i in range (0, len(SMF_Bins)) ])[1:]

        #possion err
        Totals_Sat_err = (np.divide(Totals_Sat,np.sqrt(Totals_Sat_num)))

        TotalsPlot_Sat = np.log10(np.vstack((Totals_Sat, Totals_Sat-Totals_Sat_err, Totals_Sat+Totals_Sat_err)))
        TotalsPlot_Sat[1:] = np.abs(TotalsPlot_Sat[1:]-TotalsPlot_Sat[0])
        Sat_fig = plt.errorbar(SMF_Bins[1:], TotalsPlot_Sat[0], yerr = TotalsPlot_Sat[1:],fmt ='rx', label = "Satilite")



        logMstar_Tot = np.append(np.array( self.df_cent[self.Photomotry][(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)]), np.array( self.df_sat[self.Photomotry][(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)]))
        Vmax_Tot = np.append(np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)]), np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)]))
        VmaxSum = np.sum(Vmax_Tot)
        totVmax=VmaxSum/self.fracsky

        #Bins and calculates VmaxWt per bin and number per bin
        Binned = np.digitize(logMstar_Tot, SMF_Bins)
        Totals_Tot = (np.array([ np.divide(np.sum(Vmax_Tot[Binned == i]), VmaxSum*SMF_BinWidth) for i in range (0, len(SMF_Bins)) ])[1:])*totVmax
        Totals_Tot_num = np.array([ np.size(Vmax_Tot[Binned == i]) for i in range (0, len(SMF_Bins)) ])[1:]

        #possion err
        Totals_Tot_err = (np.divide(Totals_Tot,np.sqrt(Totals_Tot_num)))

        TotalsPlot_Tot = np.log10(np.vstack((Totals_Tot, Totals_Tot-Totals_Tot_err, Totals_Tot+Totals_Tot_err)))
        TotalsPlot_Tot[1:] = np.abs(TotalsPlot_Tot[1:]-TotalsPlot_Tot[0])
        Tot_fig = plt.errorbar(SMF_Bins[1:], TotalsPlot_Tot[0], yerr = TotalsPlot_Tot[1:],fmt ='kx', label = "Total")
        SDSS_Legend = plt.legend(handles = [Tot_fig, Sat_fig, Cent_fig], loc = 1)
        ax = plt.gca().add_artist(SDSS_Legend)

        np.savetxt("./Bernardi_SDSS/SMF.dat", np.vstack((SMF_Bins[1:], TotalsPlot_Tot[0],  TotalsPlot_Tot[1:], TotalsPlot_Sat[0], TotalsPlot_Sat[1:], TotalsPlot_Cent[0], TotalsPlot_Cent[1:])))


        return Cent_fig, Sat_fig, Tot_fig, ax
    
    def sSFR_Plot(self, l, u, fig, No_Leg = False):
        #Make DF cuts
        df_sSFR_masscut = self.df_sat[ (l < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < u)]
        hist, bins = np.histogram(df_sSFR_masscut.SerExpsSFR, bins = np.arange(-13, -9, 0.1), weights = df_sSFR_masscut.Vmaxwt)
        Bi_Mod_X = bins[:-1]
        Bi_Mod_Y = hist/(np.sum(hist)*0.1)
        if No_Leg == False:
            print("1")
            fig.bar(Bi_Mod_X, Bi_Mod_Y, 0.1, color = "k", alpha = 0.4, label = "SDSS Satellites")
        else:
            print("2")
            fig.bar(Bi_Mod_X, Bi_Mod_Y, 0.1, color = "k", alpha = 0.4)
        return Bi_Mod_X, Bi_Mod_Y
    
    def sSFR_Plot_Cen(self, l, u, fig, No_Leg = False):
            #Make DF cuts
            df_sSFR_masscut = self.df_cent[ (l < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < u)]
            hist, bins = np.histogram(df_sSFR_masscut.SerExpsSFR, bins = np.arange(-13, -9, 0.1), weights = df_sSFR_masscut.Vmaxwt)
            Bi_Mod_X = bins[:-1]
            Bi_Mod_Y = hist/(np.sum(hist)*0.1)    
            if No_Leg == False:
                fig.bar(Bi_Mod_X, Bi_Mod_Y, 0.1, fill = False, alpha = 0.4 , label = "SDSS Centrals")
            else:
                fig.bar(Bi_Mod_X, Bi_Mod_Y, 0.1, fill = False, alpha = 0.4)
            return Bi_Mod_X, Bi_Mod_Y

    def sSFR_Scatter(self, l, u, fig, No_Leg = False):
        #Make DF cuts
        df_sSFR_masscut = self.df_sat[ (l < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < u)]
        fig.plot( df_sSFR_masscut[self.Photomotry] ,df_sSFR_masscut.SerExpsSFR,"x", label = "Sat")
        df_sSFR_masscut = self.df_cent[ (l < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < u)]
        fig.plot( df_sSFR_masscut[self.Photomotry] ,df_sSFR_masscut.SerExpsSFR,"+", label = "Cen")
        
        
    """def SMF_Data(self, SMF_BinWidth = 0.1):
        
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)
        
        
        #Cent
        SM = np.array(self.df_cent[self.Photomotry][(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        
        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky
        
        hist_cent, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        hist_cent_raw, edges = np.histogram(SM, bins = SMF_Bins, density  = False)
        Poss_Err_cen = np.vstack(((hist_cent)*((np.sqrt(hist_cent_raw)-1)/np.sqrt(hist_cent_raw)),(hist_cent)*((np.sqrt(hist_cent_raw)+1)/np.sqrt(hist_cent_raw))))
        Y = np.log10(np.divide(hist_cent, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_e = np.log10(np.divide(Poss_Err_cen, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_e[0], Y_e[1] = Y-Y_e[0], Y_e[1]-Y
        Cent_X, Cent_Y = SMF_Bins[1:], Y
        
        #Sat
        SM = np.array(self.df_sat[self.Photomotry][(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky
        
        hist_sat, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        hist_sat_raw, edges = np.histogram(SM, bins = SMF_Bins, density  = False)
        Poss_Err_sat = np.vstack(((hist_sat)*((np.sqrt(hist_sat_raw)-1)/np.sqrt(hist_sat_raw)),(hist_sat)*((np.sqrt(hist_sat_raw)+1)/np.sqrt(hist_sat_raw))))
        Y = np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_sat = np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_e = np.log10(np.divide(Poss_Err_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_e[0], Y_e[1] = Y-Y_e[0], Y_e[1]-Y
        Sat_X, Sat_Y = SMF_Bins[1:], Y
        
        #Tot
        SM = np.array(self.df_z[self.Photomotry][(SMF_LB < self.df_z[self.Photomotry]) & (self.df_z[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_z.Vmaxwt[(SMF_LB < self.df_z[self.Photomotry]) & (self.df_z[self.Photomotry] < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_tot, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        hist_tot_raw, edges = np.histogram(SM, bins = SMF_Bins, density  = False)
        Poss_Err_tot = np.vstack(((hist_tot)*((np.sqrt(hist_tot_raw)-1)/np.sqrt(hist_tot_raw)),(hist_tot)*((np.sqrt(hist_tot_raw)+1)/np.sqrt(hist_tot_raw))))
        Y = np.log10(np.divide(hist_tot, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_e = np.log10(np.divide(Poss_Err_tot, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_e[0], Y_e[1] = Y-Y_e[0], Y_e[1]-Y
        Tot_X, Tot_Y = SMF_Bins[1:], Y
        return Cent_X, Cent_Y, Sat_X, Sat_Y, Tot_X, Tot_Y"""
    
    def SMF_Data(self, SMF_BinWidth = 0.1, SMF_LB = 9, SMF_UB = 12.5, SMF_Bins = None, OverridePhoto = None): 
        if np.any(SMF_Bins == None):
            SMF_LB = 9; SMF_UB = 12.5
            SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)
        else:
            SMF_LB = np.min(SMF_Bins); SMF_UB = np.max(SMF_Bins)
            SMF_BinWidth = SMF_Bins[1] - SMF_Bins[0]
        if OverridePhoto != None:
            self.Photomotry = OverridePhoto
        #Centrals
        SM = np.array(self.df_cent[self.Photomotry][(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent[self.Photomotry]) & (self.df_cent[self.Photomotry] < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_cent, edges = np.histogram(SM, bins = np.append(SMF_Bins, np.max(SMF_Bins)+SMF_BinWidth)-(SMF_BinWidth/2), weights = Vmax)
        hist_cent_raw, edges = np.histogram(SM, bins = np.append(SMF_Bins, np.max(SMF_Bins)+SMF_BinWidth)-(SMF_BinWidth/2), density  = False)
        Poss_Err_cen = np.vstack(((hist_cent)*((np.sqrt(hist_cent_raw)-1)/np.sqrt(hist_cent_raw)),(hist_cent)*((np.sqrt(hist_cent_raw)+1)/np.sqrt(hist_cent_raw))))
        Y_cen = np.log10(np.divide(hist_cent, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_cen_e = np.log10(np.divide(Poss_Err_cen, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_cen_e[0], Y_cen_e[1] = Y_cen-Y_cen_e[0], Y_cen_e[1]-Y_cen

        #satellites
        SM = np.array(self.df_sat[self.Photomotry][(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat[self.Photomotry]) & (self.df_sat[self.Photomotry] < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_sat, edges = np.histogram(SM, bins = np.append(SMF_Bins, np.max(SMF_Bins)+SMF_BinWidth)-(SMF_BinWidth/2), weights = Vmax)
        hist_sat_raw, edges = np.histogram(SM, bins = np.append(SMF_Bins, np.max(SMF_Bins)+SMF_BinWidth)-(SMF_BinWidth/2), density  = False)
        Poss_Err_sat = np.vstack(((hist_sat)*((np.sqrt(hist_sat_raw)-1)/np.sqrt(hist_sat_raw)),(hist_sat)*((np.sqrt(hist_sat_raw)+1)/np.sqrt(hist_sat_raw))))
        Y_sat = np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_sat_e = np.log10(np.divide(Poss_Err_sat, self.fracsky*SMF_BinWidth)*0.9195)#0.9195 correction of volume to Planck15
        Y_sat_e[0], Y_sat_e[1] = Y_sat-Y_sat_e[0], Y_sat_e[1]-Y_sat

        
        #Total
        SM = np.array(self.df_z[self.Photomotry][(SMF_LB < self.df_z[self.Photomotry]) & (self.df_z[self.Photomotry] < SMF_UB)])
        Vmax = np.array(self.df_z.Vmaxwt[(SMF_LB < self.df_z[self.Photomotry]) & (self.df_z[self.Photomotry] < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_tot, edges = np.histogram(SM, bins = np.append(SMF_Bins, np.max(SMF_Bins)+SMF_BinWidth)-(SMF_BinWidth/2), weights = Vmax)
        hist_tot_raw, edges = np.histogram(SM, bins = np.append(SMF_Bins, np.max(SMF_Bins)+SMF_BinWidth)-(SMF_BinWidth/2), density  = False)
        Poss_Err_tot = np.vstack(((hist_tot)*((np.sqrt(hist_tot_raw)-1)/np.sqrt(hist_tot_raw)),(hist_tot)*((np.sqrt(hist_tot_raw)+1)/np.sqrt(hist_tot_raw))))
        Y_t = np.log10(np.divide(hist_tot, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_t_e = np.log10(np.divide(Poss_Err_tot, self.fracsky*SMF_BinWidth)*0.9195) #0.9195 correction of volume to Planck15
        Y_t_e[0], Y_t_e[1] = Y_t-Y_t_e[0], Y_t_e[1]-Y_t
        if OverridePhoto != None:
            self.Photomotry = "MsMendSerExp"
        return SMF_Bins, Y_t, Y_t_e, Y_sat, Y_sat_e, Y_cen, Y_cen_e