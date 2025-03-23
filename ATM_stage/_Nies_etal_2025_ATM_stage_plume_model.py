'''
ATM stage model

'''

import cantera as ct
import numpy as np

###############################################################################
# geometry
###############################################################################

def calc_R0(X_SO2, F_SO2, v_wind, T, P):
    '''
    Parameters
    ----------
    X_SO2 : float
        SO2 mixing ratio [mol/mol] (output from HT stage)
    F_SO2 : float
        SO2 Flux [kg/s] 
    v_wind : float
        wind speed [m/s]
    T : float
        plume temperature [K]
    P : float
        pressure [Pa]

    Returns
    -------
    r_eff : float
        plume radius R0 at the beginning of the ATM stage

    '''
    NA = 6.022e23 # Avogadro number
    kb = 1.38e-23 # Boltzmann constant
    M_SO2 = 64.066*1e-3 #molecular weight of SO2 in kg/mol
    
    F_SO2_molec = (F_SO2*NA)/M_SO2
    N_SO2 = (X_SO2*P)/(kb*T)
    
    A = F_SO2_molec/(v_wind*N_SO2)
    r_eff = np.sqrt(A/(2*np.pi))
    
    return np.array([r_eff])

def effective_radius(t, v_wind, t_source, A, B, C):
    '''
    Parameters
    ----------
    t : array-like
        simulation time
    v_wind : float
             windspeed in [m/s]
    t_source : array-like
               source time for different mixing scenarios in [s]
    A : array-like
        factor A for general radius description
    B : array-like
        exponent of windspeed for general radius description
    C : array-like
        exponent of the time (defines mixing scenario)

    Returns
    -------
    r_eff : array-like
            effective plume radius under the assumption of elliptical geometry

    '''
    
    r_eff = A*v_wind**B*(t+t_source)**C
    
    return r_eff

def calculate_mixing_factors(t, v_wind, damp, r0, ts_low_T, A_low_T, B_low_T, C_low_T):
    '''
    Parameters
    ----------
    t : array-like
        simulation time in [s]
    v_wind : float
             windspeed in [m/s]
    damp : float
           dampening constant of transformation between low-T and high-T mixing (0.01 works well)
    A_low_T : float
              low-T mixing factor A (see effective radius calculation)
    B_low_T : float
              low-T mixing factor B (see effective radius calculation)
    C_low_T : float
              low-T mixing factor C (see effective radius calculation)

    Returns
    -------
    A : array-like
        general mixing factor A for the complete simulation (high to low-T)
    B : array-like
        general mixing factor B for the complete simulation (combined mixinng)
    C : array-like
        general mixing factor C for the complete simulation (combined mixing)

    '''
    
    #Roberts et al 2014 PG coefficients
    ay = 0.216
    az = 0.110
    by = 0.894
    bz = 0.911

    ts_is = (r0/(np.sqrt(np.pi*az*ay)*v_wind**((bz+by)/2)))**(2/(bz+by))
    A_is = np.sqrt(np.pi*az*ay)
    B_is = (bz+by)/2
    C_is = (bz+by)/2
    
    #calculate factors for distinct timestep (A-low, B_low and C_low depend on chosen low-T mixing scenario)
    ts = ts_low_T*(1-np.exp((-1)*damp*t)) + ts_is*np.exp((-1)*damp*t)
    
    A = A_low_T*(1-np.exp((-1)*damp*t)) + A_is*np.exp((-1)*damp*t)
    B = B_low_T*(1-np.exp((-1)*damp*t)) + B_is*np.exp((-1)*damp*t)
    C = C_low_T*(1-np.exp((-1)*damp*t)) + C_is*np.exp((-1)*damp*t)
    
    return ts, A, B, C

def calculate_mixing_factors_new(t, v_wind, R0, ay, az, by, bz):
    '''
    Parameters
    ----------
    t : array-like
        simulation time in [s]
    v_wind : float
        wind speed in [m/s]
    R0 : float
        initial plume radius in the ATM stage in [m]
    ay : float
        Pasquill-Gifford mixing coefficient
    az : float
        Pasquill-Gifford mixing coefficient
    by : float
        Pasquill-Gifford mixing exponent
    bz : float
        Pasquill-Gifford mixing exponent

    Returns
    -------
    ts_atm : float
        source time in the ATM stage in [s]
    A : float
        generalized mixing coefficient
    B : float
        generalized mixing coefficient
    C : float
        generalized mixing exponent

    '''
    
    ts_atm = (R0/(np.sqrt(np.pi*az*ay)*v_wind**((bz+by)/2)))**(2/(bz+by))
    A = np.sqrt(np.pi*az*ay)
    B = (bz+by)/2
    C = (bz+by)/2
    
    return ts_atm, A, B, C

def mixing_derivative(t, t_source, C):
    '''
    Parameters
    ----------
    t : array-like
        simulation time in [s]
    t_source : array-like
               source time for chosen mixing scenario in [s] (can be variable, see calculate_mixing_factors)
    C : array-like
        mixing factor C (exponent of t in the effective radius calculation)

    Returns
    -------
    md : array-like
         mixing derivative prefactor (md*(c_A - C_pl))

    '''
    md = 2*C/(t+t_source)
    
    return md


def source_time_high_T(gamma, r0_high_T):
    
    '''
    Parameters
    ----------
    gamma : float
            product of the Richardson Obukhov constant and the viscous energy
            dissipation in m^3s^-3
    r0_high_T : float
                radius of the source of the high T simulation in m

    Returns
    -------
    array-like
    time for a point source plume to reach source size with radius r0 or
    time transformation of point source case to approximate finite source of r0
    parameter for turbulent mixing scenario (r propto 3/2)

    '''
    
    return ((r0_high_T**2/gamma)**(1/3))

def pressure_altitude_scaling(altitude):

    '''

    Parameters
    ----------
    altitude : float
        altitude in m

    Returns
    -------
    scaling factor from normal pressure to represent pressure on altitude

    '''

    return np.exp(-altitude/8000)

###############################################################################
# read ini
###############################################################################

def parse_initial_mix(inifile):
    '''

    Parameters
    ----------
    inifile : string
        path to ini file

    Returns
    -------
    file_chemmech : string
        string with the path to the chemical mechanism (*.yaml)
    T_0_atm : float
        atmospheric temperature
    P : float
        atmospheric pressure
    T_0_plume : float
        initial plume temperature
    C : float
        turbulent mixing strength (TMS) in m^2 s^-3
    r0 : float array
        some of initial source radii

    '''

    import configparser

    fconf = configparser.ConfigParser()
    fconf.read(inifile)

    file_chemmech = fconf['mechanism']['file_chemmech']
    T_0_atm = fconf.getfloat('cond','T_atm')
    P = fconf.getfloat('cond','P_atm') * pressure_altitude_scaling(fconf.getfloat('cond','alt'))
    T_0_plume = fconf.getfloat('cond','T_pl_0')
    C = fconf.getfloat('cond','C')
    r0 = np.array([float(i) for i in fconf['cond']['r0'].split(',')])

    return file_chemmech, T_0_atm, P, T_0_plume, C, r0

###############################################################################
# aerosol
###############################################################################
def aerosol_forward_rate(T,M,gamma,A,n,F):
    '''
    Parameters
    ----------
    T : numpy float
        temperature of plume atmosphere mixture [K]
    M : numpy float
        molecular weight of gas phase species [kg/mol]
    gamma : numpy float
            uptake coefficient for gas aerosol uptake
    A : numpy float
        aerosol surface density [m^2/m^3]
    n : numpy float
        number concentration of gas phase species [1/m^3]
    F : numpy float
        branching coefficient for branching reactions

    Returns
    -------
    R_1 : numpy float
        reaction rate for branching reaction 1 [molec/cm^3 s]
    R_2 : numpy float
        reaction rate for branching reaction 2 [molec/cm^3 s]

    '''
    
    R =  8.314 # ideal gas constant
    nu = np.sqrt((3*R*T)/M) # mean molecular speed [m/s]
    
    R_mks_1 = 0.25*gamma*nu*A*n*F
    R_mks_2 = 0.25*gamma*nu*A*n*(1-F)
    
    R_1 = R_mks_1 #standard unit for net production rates kmol/m^3s
    R_2 = R_mks_2
    
    return R_1, R_2

def br2_brcl_partition_factor(P,K_2,H_HBR,HBR,K_3,HCL,H_HCL):
    '''
    Parameters
    ----------
    P : numpy float
        atmospheric pressure [Pa]
    K_2 : numpy float
        aq. phase equilibrium constant: BrCl+Br- <-> Br2Cl- [m^3/mol]
    H_HBR : numpy float
        Henry coefficient for dissolution of HBr on aerosol [mol^2 m^-6 Pa^-1]
    HBR : numpy float
        HBr gas phase mixing ratio
    K_3 : numpy float
        aq. phase equilibrium constant: Br2Cl <-> Br2 + Cl- [m^3/mol]
    HCL : numpy float
        HCl gas phase mixing ratio
    H_HCL : numpy float
        Henry coefficient for dissolution of HCl on aerosol [mol^2 m^-6 Pa^-1]

    Returns
    -------
    F_BR2 : numpy float
        branching factor for Br2
    F_BRCL : numpy float
        branching factor for BrCl (F_BR2 + F_BRCL = 1)
        
    '''
    if HBR>1e-18:
        # calculate partial pressure
        pp_HCL = HCL*P # [Pa]
        pp_HBR = HBR*P
        # aqeous phase concentration of Br- and Cl-
        br_aq = pp_HBR*H_HBR #[mol^2 m^-6]
        cl_aq = pp_HCL*H_HCL
        # calculate branching factor
        F_BR2 = 1/(1+(K_3*cl_aq)/(K_2*br_aq))
        F_BRCL = 1/((K_2*br_aq)/(K_3*cl_aq)+1)
        
    else:
        if HCL>1e-18:
            F_BR2 = 0
            F_BRCL = 1
        else:
            F_BR2 = 0
            F_BRCL = 0
                    
    return F_BR2, F_BRCL


def surface_density_dilution(A0,r_plume):
    '''
    Parameters
    ----------
    A0 : numpy float
        initial aerosol surface [m^2]
    r_plume : numpy float
             plume radius for dilution calculation [m]

    Returns
    -------
    A : numpy float
        aerosol surface density at time t [m^2/m^3]

    '''
    
    A = A0/r_plume**2     #2D for 3D multiply with distance to obain volume of a cylinder
    
    return A


def aero_func(plume_obj, aerosol_rates):
    '''
    Parameters
    ----------
    plume_obj : cantera solution object
        cantera plume object for species order
    aerosol_rates : python dictionary
        dictionary specifying the rates obtained from aerosol reactions 

    Returns
    -------
    aero_vector : numpy array
        aerosol vector containing modifications on reaction rates from aerosol
    '''
    aero_dict = {}
    
    for name in plume_obj.species_names:
        aero_dict[name] = 0.0
    
    for name in aero_dict.keys():
        for aero_species in aerosol_rates.keys():
            if name == aero_species:
                aero_dict[name] = aerosol_rates[aero_species]
            else:
                pass
    
    aero_vector = np.hstack(list(aero_dict.values()))
    
    return aero_vector

def aerosol_surface(SO2,P,T,r,v_wind,t):
    
    NA = 6.023e23 # Avogadro number [molec/mol]
    R = 8.314 # ideal gas constant [J/(K mol)]
    
    n = (NA*P*SO2)/(R*T) # SO2 number concentration molec/m^3
    N = n*r**2 
    
    A0_um = N*2e-11 # aerosol surface density in um^2
    A0 = A0_um*1e-12
    
    return A0

###############################################################################
# plume model
###############################################################################

def initialize_gas(mechanism, T, P, X):

    '''

    Parameters
    ----------
    mechanism : string
        path to reaction mechanism file (*.yaml)
    T : float
        gas temperature in K
    P : float
        pressure in Pa
    X : dict
        dict of species names with respective molar mixing ratio

    Returns
    -------
    gas : Cantera Solution object
        Object containing all relevant information on the gas mixture
        (see Cantera documentation)
    '''

    gas = ct.Solution(mechanism)
    gas.TPX = T, P, X
    return gas


def T_grad_atm(T_atm, T_plume, tauT, t):

    '''

    Parameters
    ----------
    T_atm : float
        ambient atmospheric temperature in K
    T_plume : float
        magmatic temperature in K
    tauT : float
        time constant of the exponential temperature decay T_plume --> T_atm
    t : array like
        time

    Returns
    -------
    array like
        progression of temperature along the t-axis following an exponential
        decay from T_plume to T_atm with time constant tauT
        approximation of the atmospheric temperature above the lava surface

    '''
    return (T_atm+ (T_plume-T_atm)*np.exp(-t/tauT))


# Plume Class representing a plume growing with r ~ t^3/2 according to
# early turbulent diffusion

class turbPlumeOde:
    '''
    Plume Class representing a plume growing with r~t^x (x determined by mixing scenario) according to
    turbulent relative dispersion in the inertial subrange
    the plume expansion determines the in mixed air and temperature
    on that mixing axis the chemistry is calculated simultaneously
    '''

    def __init__(self, plume, atm, A0, R0, v_wind, ay, az, by, bz, chem_on, aerosol_on):

        '''

        Parameters
        ----------
        plume : ct.Solution object
            gas object representing the (initial) plume state.
        atm : ct.Solution object
            gas object representing the surrounding atmosphere (reservoir).
        aerosol : vector
            object represesnting the changes in reaction rates according to aerosol phase
        ts : float
            source time, only mixing parameter
        chem_on : boolean, optional
            switch chemistry off for pure mixing. The default is True.

        Returns
        -------
        None.

        '''
        # Parameters of the ODE system and auxiliary data are stored in the
        # turbPlumeOde object.

        self.plume = plume
        self.atm = atm
        
        self.A0 = A0
        self.R0 = R0
        self.P = atm.P
        self.Tatm_0 = atm.T
        self.T_pl_init=plume.T
        
        self.chem_on = chem_on
        self.aerosol_on = aerosol_on
        
        self.v_wind = v_wind
        
        self.ay = ay
        self.az = az
        self.by = by
        self.bz = bz

        self.tauT = .1 # decay of atm T is set here

    def __call__(self, t, y):
        '''
        the ODE function, y' = f(t,y)
        # State vector is [T, c_1, c_2, ... c_K] of the plume gas object


        set new temperature, old pressure, and new concentrations as X
        (! error: through T progression in const V)
        T is scaled to be numerically more close to concentration values
        this needs to be accounted for when calling the function
        '''

        self.plume.TPX = y[0]*1e6, self.P, y[1:]/y[1:].sum()


        '''
        calculate and set the atmospheric temperature for the time step
        '''
        Tatm = T_grad_atm(self.Tatm_0, self.T_pl_init, self.tauT, t)
        self.atm.TP = Tatm, self.P

        '''
        mixing
        determine specific heat factor 'beta' for the time step
        calculate the mixing derivatives according to Kuhn et al., 2022
        '''
        beta = (self.atm.density_mole/self.plume.density_mole)*(self.atm.cp_mole/self.plume.cp_mole)
        deltaT = ((self.atm.T) - self.plume.T) # temperature in K
        deltac = ((self.atm.concentrations) - (self.plume.concentrations)) # concentrations in kmol/m^3

        delta = np.hstack((beta*deltaT, deltac))

        # mixing derivatives
        ts_atm, A, B, C = calculate_mixing_factors_new(t, self.v_wind, self.R0, self.ay, self.az, self.by, self.bz)
        md = mixing_derivative(t, ts_atm, C)
        m1 = md*delta
        
        '''
        read the chemical conversion rates from the plume gas objects
        (chemistry derivatives) and calculate the T derivative from chemistry
        im chem_on is True
        '''
        if self.aerosol_on:
            # aerosol surface area implementation
            aerosol_sd = surface_density_dilution(2*self.A0, effective_radius(t, self.v_wind, ts_atm, A, B, C)) # dilution according to turbulent diffusion with K = 10 #3*self.A0/(4*np.pi*1.188**3)#
            
            #BROH + HBR/HCL => BR2/BRCL
            n_BROH = self.plume.concentrations[np.where(np.array(self.plume.species_names)=='BROH')] #set correct entry for gas phase number concentration of HOBR, change unit to 1/m^3
            gamma_BROH = 0.6 #newest version 0.2 roberts et al 2014
            M_HOBR = 96.911*1e-3 #molecular weight in [kg/mol]
            
            H_HCL = 2.0e4*np.exp(9000*(1/self.plume.T-1/298)) # [mol^2 m^-6 Pa^-1] Henry coefficients Surl et al 2021
            H_HBR = 1.3e7*np.exp(10000*(1/self.plume.T-1/298))
            K_2 = 1.8e4*1e-3 #BRCL+BR- <-> BR2CL Equilibrium constants
            K_3 = 1.3*1e-3 # BR2CL <-> BR2 +CL-
            HBR = self.plume.X[np.where(np.array(self.plume.species_names)=='HBR')] #mixing ratio
            HCL = self.plume.X[np.where(np.array(self.plume.species_names)=='HCL')]
           
            F_BR2, F_BRCL = br2_brcl_partition_factor(self.P,K_2,H_HBR,HBR,K_3,HCL,H_HCL) #calculate partiton of BR2 and BRCL production according to fluid phase equilibrium
            R_HOBR_BR2, R_HOBR_BRCL = aerosol_forward_rate(self.plume.T, M_HOBR, gamma_BROH, aerosol_sd, n_BROH, F_BR2)
            
            #BRONO2 + H2O => BROH + HNO3
            n_BRONO2 = self.plume.concentrations[np.where(np.array(self.plume.species_names)=='BRONO2')] 
            gamma_BRONO2 = 0.8
            M_BRONO2 = 141.909*1e-3
            R_BRONO2, R_0 = aerosol_forward_rate(self.plume.T, M_BRONO2, gamma_BRONO2, aerosol_sd, n_BRONO2, 1)
            
            #CLONO2 + H2O => CLOH + HNO3
            n_CLONO2 = self.plume.concentrations[np.where(np.array(self.plume.species_names)=='CLONO2')] 
            gamma_CLONO2 = 0.11
            M_CLONO2 = 97.46*1e-3
            R_CLONO2, R_0 = aerosol_forward_rate(self.plume.T, M_CLONO2, gamma_CLONO2, aerosol_sd, n_CLONO2, 1)
            
            #N2O5 + H2O = 2 HNO3
            n_N2O5 = self.plume.concentrations[np.where(np.array(self.plume.species_names)=='N2O5')]
            gamma_N2O5 = 0.03
            M_N2O5 = 108.01*1e-3
            R_N2O5, R0 = aerosol_forward_rate(self.plume.T, M_N2O5, gamma_N2O5, aerosol_sd, n_N2O5, 1)
                       
            #implement aerosol contribution into production rate vector
            aerosol_rates = {'BROH':(-1)*R_HOBR_BR2+(-1)*R_HOBR_BRCL+R_BRONO2, 'HBR':(-1)*R_HOBR_BR2, 'BR2':R_HOBR_BR2, 'HCL':(-1)*R_HOBR_BRCL, 'BRCL':R_HOBR_BRCL,
                             'BRONO2':(-1)*R_BRONO2, 'HNO3':R_BRONO2+R_CLONO2+2*R_N2O5, 'H2O':(-1)*R_BRONO2+(-1)*R_CLONO2+(-1)*R_N2O5, 'CLONO2':(-1)*R_CLONO2, 'CLOH':R_CLONO2,
                             'N2O5':(-1)*R_N2O5}
            aero = aero_func(self.plume, aerosol_rates)
        
            dcdt = self.plume.net_production_rates + aero # reaction rates in m^3/kmol s + addjust reaction rates by aerosol contribution
            #print(R_HOBR_BR2, R_HOBR_BRCL, R_BRONO2)
        
        if self.aerosol_on == False:
            dcdt = self.plume.net_production_rates
            print('Aerosol switched off!',dcdt[22],dcdt[23])

            
        if self.chem_on == False:
            dcdt = np.zeros(self.plume.concentrations.shape)


        dTdt = - (np.dot(self.plume.partial_molar_enthalpies, dcdt)/(self.plume.density * self.plume.cp))*1e-6

        m1[0]=m1[0]*1e-6
        # dont forget temperature scaling!!!

        # return vector of total temporal derivatives
        
        return np.hstack((dTdt, dcdt)) + m1


################################################################################
############### funcions to apply model calculations ###########################
################################################################################

def calc_sol(mech, X_plume, atm, R0, T_0_plume, P, t_range, v_wind, ay, az, by, bz, chem_on, aerosol_on):
    from scipy.integrate import solve_ivp

    # initialize plume and atmosphere objects
    plume = initialize_gas(mech, T_0_plume, P, X_plume)

    c0 = plume.concentrations
    y0 = np.hstack((T_0_plume*1e-6,c0))

    #initialize lists for solution and plume ODE for several mixing scenarios
    sol = []
    plume_ode = []
    rates = []

    #initialize output dictionary with entries for time (t), temperature (T) and gas species
    sol_comp = {}
    sol_comp['t'] = []
    sol_comp['T'] = []
    sol_comp['A'] = []
    rates = {}
    for k in range(0,len(plume.species_names)):
        sol_comp[plume.species_names[k]] = []
        rates[plume.species_names[k]] = []


    #calculate solution

    plume.TP = T_0_plume, P
        
    SO2 = X_plume['SO2']  # SO2 initial mixing ratio
    ts_atm, A, B, C = calculate_mixing_factors_new(t_range[0], v_wind, R0, ay, az, by, bz)
    A0 = aerosol_surface(SO2, P, T_0_plume, effective_radius(t_range[0], v_wind, ts_atm, A, B, C),v_wind,t_range[0])  # check to divide by proper plume volume
    #print(A0, t_source, A, B, C)
    
    plume_ode = turbPlumeOde(plume, atm, A0, R0, v_wind, ay, az, by, bz, chem_on = chem_on, aerosol_on = aerosol_on)
        
    sol = solve_ivp(plume_ode,t_range,y0,method='BDF',atol=1e-20,rtol=1e-6)
    sol.y[0,:] = sol.y[0,:]*1e6

    sol_comp['T'] = sol.y[0,:].T
    sol_comp['t'] = sol.t.T
    sol_comp['A'] = surface_density_dilution(A0, effective_radius(sol.t.T, v_wind, ts_atm, A, B, C))
    for k in range(0,len(plume.species_names)):
        sol_comp[plume.species_names[k]] = sol.y[k+1,:].T/sol.y[1:,:].sum(axis=0)
    
    for j in range(0,len(sol.t)):
        tmp = plume_ode(sol.t[j].T,sol.y[:,j].T)
        for k in range(0,len(plume.species_names)):
            rates[plume.species_names[k]].append(tmp[k])
                
    return sol_comp


    

