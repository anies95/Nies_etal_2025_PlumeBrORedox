'''
HT stage model

'''

import cantera as ct
import numpy as np

###############################################################################
# geometry
###############################################################################

def source_time(C, r0):

    '''
    Parameters
    ----------
    C : float
        product of the Richardson Obukhov constant and the viscous energy
        dissipation in m^3s^-3
    r0 : float
        radius of the source in m

    Returns
    -------
    array like
    time for a point source plume to reach source size with radius r0 or
    time transformation of point source case to approximate finite source of r0
    parameter for turbulent mixing scenario

    '''
    return((r0**2/C)**(1/3))



def r_plume_turb(C,r0,t):

    '''
    Parameters
    ----------
    C : float
        product of the Richardson Obukhov constant and the viscous energy
        dissipationin m^3s^-3
    r0 : float
        radius of the source in m
    t : array like
        time

    Returns
    -------
    array like
    plume radius after a given time t, source radius r0 and turbulent mixing
    strength C for atmospheric turbulent mixing in the inertial subrange

    '''

    ts = source_time(C,r0) # source time
    return (C*(ts+t)**3)**.5



def V_plume_turb(C,r0,t):

    '''

    Parameters
    ----------
    C : float
        product of the Richardson Obukhov constant and the viscous energy
        dissipationin m^2s^-3
    r0 : float
        radius of the source in m
    t :array like
        time

    Returns
    -------
    array like
        plume volume after a given time t, source radius r0 and turbulent
        mixing strength C for atmospheric turbulent mixing in the inertial
        subrange

    '''

    return np.pi*(r_plume_turb(C,r0,t))**2



def dil_plume_turb(C,r0,t):

    '''

    Parameters
    ----------
    C : float
        product of the Richardson Obukhov constant and the viscous energy
        dissipationin m^3s^-3
    r0 : float
        radius of the source in m
    t :array like
        time

    Returns
    -------
    array like
        plume dilution after a given time t, source radius r0 and turbulent
        mixing strength C for atmospheric turbulent mixing in the inertial
        subrange

    '''

    return V_plume_turb(C,r0,0)/V_plume_turb(C,r0,t)

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


class turbPlumeOde:
    '''
    Plume Class representing a plume growing with r~t^3/2 according to
    turbulent relative dispersion in the inertial subrange
    the plume expansion determines the in mixed air and temperature
    on that mixing axis the chemistry is calculated simultaneously
    '''

    def __init__(self, plume, atm, ts, chem_on = True):

        '''

        Parameters
        ----------
        plume : ct.Solution object
            gas object representing the (initial) plume state.
        atm : ct.Solution object
            gas object representing the surrounding atmosphere (reservoir).
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
        self.P = atm.P
        self.Tatm_0 = atm.T
        self.T_pl_init=plume.T
        self.chem_on = chem_on
        self.t_s = ts

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
        m1 = delta * (3 / (self.t_s+t))

        '''
        read the chemical conversion rates from the plume gas objects
        (chemistry derivatives) and calculate the T derivative from chemistry
        im chem_on is True
        '''
        if self.chem_on:
            dcdt = self.plume.net_production_rates # reaction rates in kmol/m^3/s
        else:
            dcdt = np.zeros(self.plume.concentrations.shape)


        dTdt = - (np.dot(self.plume.partial_molar_enthalpies, dcdt)/(self.plume.density * self.plume.cp))*1e-6

        m1[0]=m1[0]*1e-6
        # dont forget temperature scaling!!!

        # return vector of total temporal derivatives
        return np.hstack((dTdt, dcdt)) + m1


################################################################################
# equilibrium calculation
################################################################################

def calc_equilibrium(mech, X_plume, atm, ts, T_0_plume,P, t_range, chem_on = False):

    '''
    function to calculate the equilibrium and pure mixing evolution of the plume

    Parameters
    ----------
    mech : string
        file containing the chemical mechanism (*.yaml)
    X_plume : dict
        dictionary with plume's initial composition'
    atm : ct Solution object
        atmosphere gas object
    ts : float
        mixing parameter ts
    T_0_plume : float
        initital plume temperature
    P : float
        pressure
    t_range : 2-sequence
        start and end point of simulation time range
    chem_on : boolean, optional
        turn kinetical chemistry on or off. The default is False.

    Returns
    -------
    sol_mix
        solution containing the pure mixing of plume and atmosphere (provides t and T for sol_eq)
    sol_eq : TYPE
        solution containing the equilibrium calculation between plume and atmosphere
    '''

    from scipy.integrate import solve_ivp

    # initialize plume and atmosphere objects
    plume = initialize_gas(mech, T_0_plume, P, X_plume)
    plume.equilibrate('TP')


    c0 = plume.concentrations
    y0 = np.hstack((T_0_plume*1e-6,c0))

    #thermodynamic equilibrium calculation for every timestep from mixing calculation
    sol_mix = []
    plume_ode = []

    for i,ele in enumerate(ts):
        plume.TP = T_0_plume, P
        plume_ode.append(turbPlumeOde(plume, atm, ele, chem_on = chem_on))
        sol_mix.append(solve_ivp(plume_ode[i],t_range,y0,method='BDF',atol=1e-20,rtol=1e-9))
        sol_mix[i].y[0,:]=sol_mix[i].y[0,:]*1e6

    plume_eq = initialize_gas(mech, T_0_plume, P, X_plume)

    sol_eq = {}
    for k in range(0,len(plume_eq.species_names)):
        sol_eq[plume_eq.species_names[k]] = []

    for i,ets in enumerate(ts):
        for j,ele in enumerate(sol_mix[i].y[0,:]):
            plume_eq.TPX = ele, P, sol_mix[i].y[1:,j]/sol_mix[i].y[1:,j].sum()
            plume_eq.equilibrate('TP')
            for k in range(0,len(plume_eq.species_names)):
                sol_eq[plume_eq.species_names[k]].append(plume_eq.X[k])

    return sol_mix, sol_eq

################################################################################
# H2S threshold calculation
#necessary because solution sometimes shows some instabilities based on large H2S gradients
################################################################################

def calc_sol_threshold(mech, X_plume, atm, C,r0, T_0_plume,P, t_range,threshhold, chem_on = True):

    '''
    function to dampen numerical issues through fast H2S oxidation

    Parameters
    ----------
    mech : string
        file containing the chemical mechanism (*.yaml)
    X_plume : dict
        dictionary with plume's initial composition'
    atm : ct Solution object
        atmosphere gas object
    r0 : float
        initial bubble radius to calculate mixing scenario through source time t_s
    T_0_plume : float
        initital plume temperature
    P : float
        pressure
    t_range : 2-sequence
        start and end point of simulation time range
    threshold : float
        value of H2S threshold (simulation termination and restart when triggerd)
    chem_on : boolean, optional
        turn kinetical chemistry on or off. The default is True.

    Returns
    -------
    sol_comp: dictionary
        dictionary containing the solution of the threshold calculation

    '''

    from scipy.integrate import solve_ivp

    #implement H2S threshold
    def H2S_threshold(t,y):
        return y[41] - threshhold

    H2S_threshold.terminal = True

    # initialize plume and atmosphere objects
    plume = initialize_gas(mech, T_0_plume, P, X_plume)

    c0 = plume.concentrations
    y0 = np.hstack((T_0_plume*1e-6,c0))

    #initialize lists for solution and plume ODE for several mixing scenarios
    sol = []
    plume_ode = []

    #initialize output dictionary with entries for time (t), temperature (T) and gas species
    sol_comp = {}
    sol_comp['t'] = []
    sol_comp['T'] = []
    for k in range(0,len(plume.species_names)):
        sol_comp[plume.species_names[k]] = []


    #loop over different mixing scenarois
    for i,ele in enumerate(r0):

        # simulation run (terminated when H2S threshold is triggered, see events =...)
        t_source = source_time(C,r0)
        plume.TP = T_0_plume, P
        plume_ode.append(turbPlumeOde(plume, atm, t_source, chem_on = chem_on))
        tmp = solve_ivp(plume_ode[i],t_range,y0,method='BDF',events=H2S_threshold,atol=1e-20,rtol=1e-5)
        sol.append(tmp)

        #status == 1 when H2S threshold triggered
        if tmp.status == 1:
            #create index j since solution object has two entries because of the newly started simulation
            j = i+1
            print('H2S threshhold active!')

            #calculate bubble radius at the terminated timestep to restart simulation with correct dilution
            r_turb = r_plume_turb(C,ele,sol[i].t[-1])
            t_s = source_time(C,r_turb)
            #initialize plume object with gas composition from the last timestep of the terminated solution
            X = sol[i].y[1:,-1]
            X[40] = threshhold - 0.1*threshhold #X[40] entry for H2S
            plume = initialize_gas(mech, sol[i].y[0,-1]*1e6, P, X)

            c0 = plume.concentrations
            y0 = np.hstack((sol[i].y[0,-1],c0))

            plume_ode.append(turbPlumeOde(plume, atm, t_s, chem_on = chem_on))
            sol.append(solve_ivp(plume_ode[i],(sol[i].t[-1],t_range[1]),y0,method='BDF',atol=1e-20,rtol=1e-5))

            #create output dictionary sol_comp
            sol[i].y[0,:]=sol[i].y[0,:]*1e6
            sol[j].y[0,:]=sol[j].y[0,:]*1e6

            sol_comp['T'] = np.append(sol[i].y[0,:].T,sol[j].y[0,:].T)
            sol_comp['t'] = np.append(sol[i].t.T,sol[j].t.T)
            for k in range(0,len(plume.species_names)):
                sol_comp[plume.species_names[k]] = np.append(sol[i].y[k+1,:].T/sol[i].y[1:,:].sum(axis=0),sol[j].y[k+1,:].T/sol[j].y[1:,:].sum(axis=0))

        #if H2S threshold is not triggered: create normal output dictionary sol_comp
        else:
            sol[i].y[0,:]=sol[i].y[0,:]*1e6

            sol_comp['T'] = sol[i].y[0,:].T
            sol_comp['t'] = sol[i].t.T
            for k in range(0,len(plume.species_names)):
                sol_comp[plume.species_names[k]] = sol[i].y[k+1,:].T/sol[i].y[1:,:].sum(axis=0)

    return sol_comp

def calc_sol_threshold_ini_eq(mech, X_plume, atm, C,r0, T_0_plume,P, t_range,threshhold, chem_on = True):
    '''

    same function as above, but plume object is equilibrated before start of the simulation

    Parameters
    ----------
    mech : string
        file containing the chemical mechanism (*.yaml)
    X_plume : dict
        dictionary with plume's initial composition'
    atm : ct Solution object
        atmosphere gas object
    r0 : float
        initial bubble radius to calculate mixing scenario through source time t_s
    T_0_plume : float
        initital plume temperature
    P : float
        pressure
    t_range : 2-sequence
        start and end point of simulation time range
    threshold : float
        value of H2S threshold (simulation termination and restart when triggerd)
    chem_on : boolean, optional
        turn kinetical chemistry on or off. The default is True.

    Returns
    -------
    sol_comp: dictionary
        dictionary containing the solution of the threshold calculation

    '''

    from scipy.integrate import solve_ivp

    #implement H2S threshold
    def H2S_threshold(t,y):
        return y[41] - threshhold

    H2S_threshold.terminal = True

    # initialize plume and atmosphere objects
    plume = initialize_gas(mech, T_0_plume, P, X_plume)
    plume.equilibrate('TP')

    c0 = plume.concentrations
    y0 = np.hstack((T_0_plume*1e-6,c0))

    #initialize lists for solution and plume ODE for several mixing scenarios
    sol = []
    plume_ode = []

    #initialize output dictionary with entries for time (t), temperature (T) and gas species
    sol_comp = {}
    sol_comp['t'] = []
    sol_comp['T'] = []
    for k in range(0,len(plume.species_names)):
        sol_comp[plume.species_names[k]] = []


    #loop over different mixing scenarois
    for i,ele in enumerate(r0):

        # simulation run (terminated when H2S threshold is triggered, see events =...)
        t_source = source_time(C,r0)
        plume.TP = T_0_plume, P
        plume_ode.append(turbPlumeOde(plume, atm, t_source, chem_on = chem_on))
        tmp = solve_ivp(plume_ode[i],t_range,y0,method='BDF',events=H2S_threshold,atol=1e-20,rtol=1e-5)
        sol.append(tmp)

        #status == 1 when H2S threshold triggered
        if tmp.status == 1:
            #create index j since solution object has two entries because of the newly started simulation
            j = i+1
            print('H2S threshhold active!')

            #calculate bubble radius at the terminated timestep to restart simulation with correct dilution
            r_turb = r_plume_turb(C,ele,sol[i].t[-1])
            t_s = source_time(C,r_turb)
            #initialize plume object with gas composition from the last timestep of the terminated solution
            X = sol[i].y[1:,-1]
            X[40] = threshhold - 0.1*threshhold
            plume = initialize_gas(mech, sol[i].y[0,-1]*1e6, P, X)

            c0 = plume.concentrations
            y0 = np.hstack((sol[i].y[0,-1],c0))

            plume_ode.append(turbPlumeOde(plume, atm, t_s, chem_on = chem_on))
            sol.append(solve_ivp(plume_ode[i],(sol[i].t[-1],t_range[1]),y0,method='BDF',atol=1e-20,rtol=1e-5))

            #create output dictionary sol_comp
            sol[i].y[0,:]=sol[i].y[0,:]*1e6
            sol[j].y[0,:]=sol[j].y[0,:]*1e6

            sol_comp['T'] = np.append(sol[i].y[0,:].T,sol[j].y[0,:].T)
            sol_comp['t'] = np.append(sol[i].t.T,sol[j].t.T)
            for k in range(0,len(plume.species_names)):
                sol_comp[plume.species_names[k]] = np.append(sol[i].y[k+1,:].T/sol[i].y[1:,:].sum(axis=0),sol[j].y[k+1,:].T/sol[j].y[1:,:].sum(axis=0))

        #if H2S threshold is not triggered: create normal output dictionary sol_comp
        else:
            sol[i].y[0,:]=sol[i].y[0,:]*1e6

            sol_comp['T'] = sol[i].y[0,:].T
            sol_comp['t'] = sol[i].t.T
            for k in range(0,len(plume.species_names)):
                sol_comp[plume.species_names[k]] = sol[i].y[k+1,:].T/sol[i].y[1:,:].sum(axis=0)

    return sol_comp

