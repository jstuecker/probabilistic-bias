import numpy as np

bias_estimators = {2:{}, 4:{}}

#---------- Spatial Order 2 -----------

def deriv1_J2_so2(t, sigma):
    return t["J2"]/sigma[0]**2
bias_estimators[2]["J2"] = deriv1_J2_so2

def deriv2_J22_so2(t, sigma):
    return (-sigma[0]**2 + t["J2"]**2)/sigma[0]**4
bias_estimators[2]["J22"] = deriv2_J22_so2

def deriv2_J2__2_so2(t, sigma):
    return (15/4)*(-2*sigma[0]**2 + 3*t["J2=2"])/sigma[0]**4
bias_estimators[2]["J2=2"] = deriv2_J2__2_so2

#---------- Spatial Order 4 -----------
def deriv1_J2_so4(t, sigma):
    return (sigma[1]**2*t["J4"] + sigma[2]**2*t["J2"])/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)
bias_estimators[4]["J2"] = deriv1_J2_so4

def deriv2_J22_so4(t, sigma):
    return (sigma[1]**4*t["J4"]**2 + 2*sigma[1]**2*sigma[2]**2*t["J2"]*t["J4"] + sigma[2]**4*t["J2"]**2 - sigma[2]**2*(sigma[0]**2*sigma[2]**2 - sigma[1]**4))/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)**2
bias_estimators[4]["J22"] = deriv2_J22_so4

def deriv2_J2__2_so4(t, sigma):
    return (15/4)*(3*sigma[1]**4*t["J4=4"] + 6*sigma[1]**2*sigma[2]**2*t["J2=4"] + 3*sigma[2]**4*t["J2=2"] - 2*sigma[2]**2*(sigma[0]**2*sigma[2]**2 - sigma[1]**4))/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)**2
bias_estimators[4]["J2=2"] = deriv2_J2__2_so4

def deriv1_J4_so4(t, sigma):
    return (sigma[0]**2*t["J4"] + sigma[1]**2*t["J2"])/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)
bias_estimators[4]["J4"] = deriv1_J4_so4

def deriv2_J24_so4(t, sigma):
    return (sigma[0]**2*sigma[1]**2*t["J4"]**2 + sigma[0]**2*sigma[2]**2*t["J2"]*t["J4"] + sigma[1]**4*t["J2"]*t["J4"] + sigma[1]**2*sigma[2]**2*t["J2"]**2 - sigma[1]**2*(sigma[0]**2*sigma[2]**2 - sigma[1]**4))/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)**2
bias_estimators[4]["J24"] = deriv2_J24_so4

def deriv2_J2__4_so4(t, sigma):
    return (15/4)*(3*sigma[0]**2*sigma[1]**2*t["J4=4"] + 3*sigma[0]**2*sigma[2]**2*t["J2=4"] + 3*sigma[1]**4*t["J2=4"] + 3*sigma[1]**2*sigma[2]**2*t["J2=2"] - 2*sigma[1]**2*(sigma[0]**2*sigma[2]**2 - sigma[1]**4))/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)**2
bias_estimators[4]["J2=4"] = deriv2_J2__4_so4

def deriv2_J3_3_so4(t, sigma):
    return 3*(-sigma[1]**2 + t["J3-3"])/sigma[1]**4
bias_estimators[4]["J3-3"] = deriv2_J3_3_so4

def deriv2_J3___3_so4(t, sigma):
    return (35/4)*(5*t["J3---3"] - 2*sigma[1]**2)/sigma[1]**4
bias_estimators[4]["J3---3"] = deriv2_J3___3_so4

def deriv2_J44_so4(t, sigma):
    return (sigma[0]**4*t["J4"]**2 + 2*sigma[0]**2*sigma[1]**2*t["J2"]*t["J4"] - sigma[0]**2*(sigma[0]**2*sigma[2]**2 - sigma[1]**4) + sigma[1]**4*t["J2"]**2)/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)**2
bias_estimators[4]["J44"] = deriv2_J44_so4

def deriv2_J4__4_so4(t, sigma):
    return (15/4)*(3*sigma[0]**4*t["J4=4"] + 6*sigma[0]**2*sigma[1]**2*t["J2=4"] - 2*sigma[0]**2*(sigma[0]**2*sigma[2]**2 - sigma[1]**4) + 3*sigma[1]**4*t["J2=2"])/(sigma[0]**2*sigma[2]**2 - sigma[1]**4)**2
bias_estimators[4]["J4=4"] = deriv2_J4__4_so4

def deriv2_J4____4_so4(t, sigma):
    return (315/64)*(-8*sigma[2]**2 + 35*t["J4==4"])/sigma[2]**4
bias_estimators[4]["J4==4"] = deriv2_J4____4_so4

