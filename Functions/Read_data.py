import numpy as np

def LoadData_3(RunParam_List):
    """Figure 3"""
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Figure3_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Figure3_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModel_SMF = []
    for RunParam in RunParam_List:
        AnalyticalModel_SMF.append(np.load(OutputFolder +"RunParam_{}/Figure3_AnalyticalModel_SMF.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModel_SMF), Surviving_Sat_SMF_MassRange
def LoadData_4_6(RunParam_List):
    """Figure 4 + 6"""
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Figure4_6_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    SM_Cuts = np.load(OutputFolder +"RunParam_{}/Figure4_6_SM_Cuts.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModelFrac_ = []
    for RunParam in RunParam_List:
        AnalyticalModelFrac_.append(np.load(OutputFolder +"RunParam_{}/Figure4_6_AnalyticalModelFrac_.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    AnalyticalModelNoFrac_ = []
    for RunParam in RunParam_List:
        AnalyticalModelNoFrac_.append(np.load(OutputFolder +"RunParam_{}/Figure4_6_AnalyticalModelNoFrac_.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModelFrac_), np.array(AnalyticalModelNoFrac_), SM_Cuts
def LoadData_5(RunParam_List):
    """Figure 5"""
    Sat_SMHM = []
    Sat_Parent_SMHM = []
    for RunParam in RunParam_List:
        Sat_SMHM.append(np.load(OutputFolder +"RunParam_{}/Figure5_Sat.npy".format("".join(("{}_".format(i) for i in RunParam)))))
        Sat_Parent_SMHM.append(np.load(OutputFolder +"RunParam_{}/Figure5_Sat_Parent_SMHM.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return Sat_SMHM, Sat_Parent_SMHM
def LoadData_7(RunParam):
    """Figure 7"""
    Mergers = np.load(OutputFolder +"RunParam_{}/Figure7_Mergers.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Minor_Mergers = np.load(OutputFolder +"RunParam_{}/Figure7_Minor_Mergers.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Figure7_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Mergers, Minor_Mergers, z
def LoadData_8(RunParam):
    """Figure 8"""
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Figure8_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    P_Elliptical = np.load(OutputFolder +"RunParam_{}/Figure8_P_Elliptical.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, P_Elliptical
def LoadData_9(RunParam):
    """Figure 9"""
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Figure9_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Figure9_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Analyticalmodel_SI = np.load(OutputFolder +"RunParam_{}/Figure9_Analyticalmodel_SI.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, Analyticalmodel_SI
def LoadData_10(RunParam_List):
    """Figure 10"""
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Figure10_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Figure10_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModel_SMF = []
    for RunParam in RunParam_List:
        AnalyticalModel_SMF.append(np.load(OutputFolder +"RunParam_{}/Figure10_AnalyticalModel_SMF.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    return AvaHaloMass, np.array(AnalyticalModel_SMF), Surviving_Sat_SMF_MassRange
def LoadData_SMFhz(RunParam_List):
    """Figure 3"""
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/SMFhz_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/SMFhz_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    AnalyticalModel_SMF = []
    for RunParam in RunParam_List:
        AnalyticalModel_SMF.append(np.load(OutputFolder +"RunParam_{}/SMFhz_AnalyticalModel_SMF_Highz.npy".format("".join(("{}_".format(i) for i in RunParam)))))
    z = np.load(OutputFolder +"RunParam_{}/z_infall_z.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    return AvaHaloMass, np.array(AnalyticalModel_SMF), Surviving_Sat_SMF_MassRange, z
def LoadData_z_infall(RunParam):
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/z_infall_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/z_infall_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z_infall = np.load(OutputFolder +"RunParam_{}/z_infall.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Surviving_Sat_SMF_MassRange, z, z_infall
def LoadData_sSFR(RunParam_List):
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/sSFR_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    sSFR_Range = np.load(OutputFolder +"RunParam_{}/sSFR_Range.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    Satellite_sSFR = np.load(OutputFolder +"RunParam_{}/Satellite_sSFR.npy".format("".join(("{}_".format(i) for i in RunParam_List[0]))))
    return Surviving_Sat_SMF_MassRange, sSFR_Range, Satellite_sSFR
def LoadData_Sat_SMHM(RunParam):
    z = np.load(OutputFolder +"RunParam_{}/Sat_SMHM_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SatHaloMass = np.load(OutputFolder +"RunParam_{}/Sat_SMHM_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Sat_SMHM_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Sat_SMHM_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Sat_SMHM = np.load(OutputFolder +"RunParam_{}/Sat_SMHM_Sat_SMHM.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Sat_SMHM_Host = np.load(OutputFolder +"RunParam_{}/Sat_SMHM_Sat_SMHM_Host.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return z, SatHaloMass, AvaHaloMass, Surviving_Sat_SMF_MassRange, Sat_SMHM, Sat_SMHM_Host
def LoadData_Mergers(RunParam):
    Accretion_History = np.load(OutputFolder +"RunParam_{}/Mergers_Accretion_History.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Mergers_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Mergers_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Mergers_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Accretion_History, z, AvaHaloMass, Surviving_Sat_SMF_MassRange
def LoadData_Pair_Frac(RunParam):
    Pair_Frac = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Pair_Frac.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Pair_Frac_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Pair_Frac_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Pair_Frac, z, AvaHaloMass, Surviving_Sat_SMF_MassRange
def LoadData_Sat_Env_Highz(RunParam):
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Sat_Env_Highz_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Sat_Env_Highz_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AnalyticalModelFrac_ = np.load(OutputFolder +"RunParam_{}/Sat_Env_Highz_AnalyticalModelFracHighz.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AnalyticalModelNoFrac_ = np.load(OutputFolder +"RunParam_{}/Sat_Env_Highz_AnalyticalModelNoFracHighz.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SM_Cuts = np.load(OutputFolder +"RunParam_{}/Sat_Env_Highz_SM_Cuts.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, AnalyticalModelFrac_, AnalyticalModelNoFrac_, SM_Cuts
def LoadData_Raw_Richness(RunParam):
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Raw_Richness_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Raw_Richness_Highz_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Raw_Richness_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_Weighting_highz = np.load(OutputFolder +"RunParam_{}/Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, Surviving_Sat_SMF_MassRange, Surviving_Sat_SMF_Weighting_highz
def LoadData_MultiEpoch_SubHalos(RunParam):
    z = np.load(OutputFolder +"RunParam_{}/MultiEpoch_SubHalos_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SatHaloMass = np.load(OutputFolder +"RunParam_{}/MultiEpoch_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SurvivingSubhalos_z_z = np.load(OutputFolder +"RunParam_{}/MultiEpoch_SurvivingSubhalos_z_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return z, SatHaloMass, SurvivingSubhalos_z_z
def LoadData_Pair_Frac_Halo(RunParam):
    z = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Halo_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Pair_Frac_Halo = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Halo_Pair_Frac_Halo.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Accretion_History_Halo = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Halo_Accretion_History_Halo.npy".format("".join(("{}_".format(i) for i in RunParam))))
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Halo_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    SatHaloMass = np.load(OutputFolder +"RunParam_{}/Pair_Frac_Halo_SatHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return Pair_Frac_Halo, Accretion_History_Halo, z, AvaHaloMass, SatHaloMass
def LoadData_Total_Starformation(RunParam):
    AvaHaloMass = np.load(OutputFolder +"RunParam_{}/Total_Starformation_AvaHaloMass.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Surviving_Sat_SMF_MassRange = np.load(OutputFolder +"RunParam_{}/Total_Starformation_Surviving_Sat_SMF_MassRange.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Total_StarFormation_Means = np.load(OutputFolder +"RunParam_{}/Total_Starformation_Total_StarFormation_Means.npy".format("".join(("{}_".format(i) for i in RunParam))))
    Total_StarFormation_Std = np.load(OutputFolder +"RunParam_{}/Total_Starformation_Total_StarFormation_Std.npy".format("".join(("{}_".format(i) for i in RunParam))))
    z = np.load(OutputFolder +"RunParam_{}/Total_Starformation_z.npy".format("".join(("{}_".format(i) for i in RunParam))))
    return AvaHaloMass, z, Surviving_Sat_SMF_MassRange, Total_StarFormation_Means, Total_StarFormation_Std
#==========================Loading Output=======================================

