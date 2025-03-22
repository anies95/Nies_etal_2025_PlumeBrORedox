# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:47:39 2025

@author: Alexander Nies

Example script to reproduce the model results shown in Nies et al 2025

doi: 
 

"""

import cantera as ct
import numpy as np
import sys
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

###############################################################################
#################### load HT and ATM stage plume models #######################
###############################################################################

folder_HT_stage_path = os.path.abspath('HT_stage')
folder_ATM_stage_path = os.path.abspath('ATM_stage')

sys.path.append(folder_HT_stage_path)
sys.path.append(folder_ATM_stage_path)

from _Nies_etal_2025_HT_stage_plume_model import*
from _Nies_etal_2025_ATM_stage_plume_model import*

###############################################################################
################## initialisation of variables HT stage #######################
###############################################################################

inifile_HT_stage = 'HT_stage/Nies_etal_2025_HT_stage_inifile.txt'

mech_HT_stage, T_0_atm, P, T_0_plume_HT_stage, C, r0 = parse_initial_mix(inifile_HT_stage)

# source time (mixing parameter)
ts = source_time(C,r0)

#simulation time in the HT stage
t_range_HT_stage = (0,10) #s

###############################################################################
##################### gas composition (plume + atmosphere) ####################
###############################################################################

#gas data from Martin et al 2011 (based on Bagnato et al. 2007)
#HCl/HBr = 1000 (Aiuppa 2005a)
#SO2/H2S = 20 (Aiuppa 2005b)
#equilibrium H2/H2O at 1373 K log(fO2) = NNO + 0.03 (e.g. Allard 2005)
#CO calculated as equilibrium at 1373 K
#HF, HI are not in the model => AR

X_HT_stage = {'H2O':0.86, 'SO2':0.029, 'CO2':0.096, 'HCL':0.014, 'HBR':1.4e-5, 'H2S':0.0015, 'H2':0.005, 'AR':0.00300039, 'CO':0.0037}

X_0_atm = {'N2':0.775, 'O2':0.205, 'H2O':0.02, 'CO2':400e-6, 'CO': 148e-9, 'H2':0.55e-6, 'CH4':1.785e-6, 'O3': 80e-9, 
           'NO': 0.05e-9, 'NO2': 0.11e-9, 'NO3': 5e-14, 'OH': 0.0007e-9, 'HO2': 0.03e-9, 'H2O2': 2e-9, 
           'CH2O':0.55e-9, 'HONO':2.1e-12, 'H2S':0.00011e-6}

atm = initialize_gas(mech_HT_stage, T_0_atm, P, X_0_atm) #atmosphere gas object

###############################################################################
######################### HT stage model run ##################################
###############################################################################

sol = calc_sol_threshold(mech_HT_stage, X_HT_stage, atm, C, r0, T_0_plume_HT_stage, P, t_range_HT_stage,1e-12)

###############################################################################
################# intitalisation of variables ATM stage #######################
############################################################################### 
#%%

inifile_ATM_stage = 'ATM_stage/Nies_etal_2025_ATM_stage_inifile.txt'
mech_ATM_stage, T_0_atm, P, T_0_plume_ATM_stage, C, R0_ATM_stage = parse_initial_mix(inifile_ATM_stage)

species_list_ATM_stage = ['N2', 'O2', 'O3', 'NO', 'NO2', 'NO3', 'N2O5', 'HNO3', 'HONO',
                          'CH4', 'CH3O2', 'HCHO', 'H2O', 'CO2', 'SO2','OH', 'HO2', 'H2O2', 'H2', 'CO', 'H2S',
                          'HBR', 'BR', 'BROH', 'BR2', 'BRCL', 'BRO','HCL', 'CL', 'CL2', 'CLOH', 'CLOO', 'OCLO', 'CLO']

X_0_atm = {'N2':0.775, 'O2':0.205, 'H2O':0.02, 'CO2':400e-6, 'CO': 148e-9, 'H2':0.55e-6, 'CH4':1.785e-6, 'O3': 80e-9, 
           'NO': 0.05e-9, 'NO2': 0.11e-9, 'NO3': 5e-14, 'OH': 0.0007e-9, 'HO2': 0.03e-9, 'H2O2': 2e-9, 
           'HCHO':0.55e-9, 'HONO':2.1e-12, 'H2S':0.00011e-6}

#identify index where T_plume <= 400 K
indices = np.where(sol['T'] <= 400)[0]
first_index = indices[0] if indices.size > 0 else None

#simulation time in the ATM stage
t_range_ATM_stage = (0,3600) #s

T_plume_ATM_stage = sol['T'][first_index] # set correct T_plume

#initialize gas composition for ATM stage
X_ATM_stage = {key: sol[key][first_index] for key in species_list_ATM_stage if key in sol}
atm_ATM_stage = initialize_gas(mech_ATM_stage, T_0_atm, P, X_0_atm)


#mixing scenario

v_wind = 10 #windspeed in m/s
F_SO2 = 12 #SO2 flux in kg/s
R0_ATM_stage = calc_R0(X_ATM_stage['SO2'], F_SO2, v_wind, T_plume_ATM_stage, P)

#Pasquill-Gifford mixing coefficients
ay = 0.230
az = 0.076
by = 0.855
bz = 0.879

###############################################################################
###################### ATM stage model run ####################################
###############################################################################
sol_atm = calc_sol(mech_ATM_stage, X_ATM_stage, atm_ATM_stage, R0_ATM_stage, T_plume_ATM_stage, P,
                          t_range_ATM_stage, v_wind, ay, az, by, bz, chem_on=True, aerosol_on=True)

###############################################################################
######################## plots ################################################
###############################################################################
#%%

fig, ax = plt.subplots(figsize=(7.25,3.75))

ax.plot(sol['t'][:first_index], sol['T'][:first_index], linewidth=3, color='black', label=r'HT stage')
ax.plot(sol_atm['t']+sol['t'][first_index], sol_atm['T'], linewidth=3, linestyle='--', color='black', label=r'ATM stage')

ax.grid()
ax.legend(fontsize=16)
ax.set_xscale('log')
ax.set_xlim(1e-3,3600)
ax.set_xlabel(r'plume age [s]',fontsize=16)
ax.set_ylabel(r'plume temperature [K]',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

fig, ax = plt.subplots(figsize=(7.25,3.75))

ax.plot(sol['t'][:first_index], sol['OH'][:first_index], linewidth=3, color='black', label=r'OH')
ax.plot(sol['t'][:first_index], sol['HO2'][:first_index], linewidth=3, linestyle='--', color='black', label=r'HO$_2$')

ax.grid()
ax.legend(fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-3,sol['t'][first_index])
ax.set_ylim(1e-12,1e-3)
ax.set_xlabel(r'plume age [s]',fontsize=16)
ax.set_ylabel(r'mixing ratio [mol/mol]',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

fig, ax = plt.subplots(figsize=(7.25,3.75))

ax.plot(sol_atm['t'], sol_atm['BRO']/sol_atm['SO2'], linewidth=3, color='black', label=r'BrO/SO$_2$')

ax.grid()
ax.legend(fontsize=16)
ax.set_ylim(0,3e-4)
ax.set_xlabel(r'plume age [s]',fontsize=16)
ax.set_ylabel(r'mixing ratio [mol/mol]',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()












 

