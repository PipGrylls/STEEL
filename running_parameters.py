#File For storing the essential running parameters to be inhereted by every script


#Set the cosmology for STEEL to run in.=============================
from colossus.cosmology import cosmology

#Named cosmologies can be found in the colossus documentation: https://bdiemer.bitbucket.io/colossus/cosmology_cosmology.html
# eg. 'planck15', 'millennium' ...
#Alterntivly set your own:
params = {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
cosmocosmology.addCosmology('myCosmo', params)

cosmo = cosmology.setCosmology('planck15')
#===================================================================

#Set the parameters for the abundance matching and modelling=======

#This parameter set is used to pass custom parameter to the SMHM relationship it should be set in script when required.
override =\
{\
'M10':0,\
'SHMnorm10':0,\
'beta10':0,\
'gamma10':0,\
'M11':0,\
'SHMnorm11':0,\
'beta11':0,\
'gamma11':0\
}

#Named Abundace matching models, scatter magnitude, redshift evolution toggle. 
abn_mtch =\
{\
'Behroozi13': False,\
'Behroozi18': False,\
'Grylls18':False,\
'Grylls19_PyMorph':False,\
'Grylls19_cModel':False,\
'Moster10': False,\
'Moster13': False,\
'Illustris': False,\
'Override_on': False,\
'Override_params':override,\
'z_Evo':True,\
'Scatter': 0.15
}

#Set the main dictonary
paramaters = \
{\
'AbnMtch' : abn_mtch,\
'AltDynamicalTime': 1,\
'NormRnd': 0.5,\
'SFR_Model': 'CE',\
'PreProcessing': False,\
}
#===================================================================

#Set functional model running parameters===========================

#Setting this to true reduces the binning for the HMF and SMF outputs and running and lowers the minimum satellite halo mass
high_res = True

#When using abundance matching do n realisations to capture upscatter effects, this can be as low as 5 and still provide decent testing output for smooth gaussians and goos statistics make this higher at the sacrifice of runtime. (high n and high_res do NOT play nicely)
n = 5

#Set the upper and lower central halo mass in the cosmology units are Mvir
analytic_hm_min = 11.0; analytic_hm_max = 16.6 

#These are the satellite mass cuts used to create output plots (These are all > so more is simply more flexibility not a decrease in bin)
sm_cuts = [9, 9.5, 10, 10.5, 11, 11.45]
