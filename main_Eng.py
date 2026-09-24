"""
The following code aims to simulate the dynamic transients of a pressurized light-water nuclear reactor.
In particular, the data used refer to the RITM-200 SMR model, but the trends can be generalized
to other nuclear reactors of the same type.

Below is a list of the main variables used, with their respective meanings, to facilitate
understanding of the code:

- T_hot     : temperature of the reactor "hot leg", i.e. the temperature leaving the core and entering the steam generator
- T_cold    : temperature of the reactor "cold leg", i.e. the temperature leaving the steam generator and entering the core
- T_s_in    : temperature of the secondary circuit water entering the steam generator
- T_s_out   : temperature of the secondary circuit water (steam) leaving the steam generator
"""

"""
HEAT EXCHANGER MODELING
Geometric scheme (cross-section):
  ------------------------------
  │ Shell (primary side)       │
  │   -----------------------  │
  │   │  Outer tube (2nd)   │  │
  │   │  -----------------  │  │
  │   │  │  Inner tube   │  │  │
  │   │  │   (primary)   │  │  │
  │   │  -----------------  │  │
  │   │    annulus (2nd)    │  │
  │   -----------------------  │
  |                            |
  ------------------------------

Flows:
  Primary:     flows through the inner tube (1st) and the shell (1st), in counterflow with the secondary
  Secondary:   flows through the annulus between the inner and outer tubes
"""


from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from pyXSteam.XSteam import XSteam
import time

# Initialization of the steam tables (SI system: bar, °C, kJ, m^3, kg)
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)


#==========================================================================
# SECTION 0 — System parameters (geometry, operating conditions, etc.)
#==========================================================================

#NEUTRONICS VALUES

af = -2e-5
am = -10e-5                 # pcm/°C
ah = -70e-5                 # pcm/cm
h0 = 0                      # initial control rod height [cm]
h = float(input("Enter the variation h in cm (control rod height): "))
delta_fr = float(input("Enter the percentage increase in flowrate (e.g. +-10%): ")) / 100
t_start = time.perf_counter()
BETA = 650e-5               # delayed neutron fraction [pcm]
PROMPT_LIFETIME = 1e-4      # prompt neutron lifetime [s]
LAMBDAS = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])          # decay constants of the delayed neutron groups [1/s]
BETAS_I = BETA * np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])   # delayed fraction for each precursor group
n0 = 1.0                    # normalization factor
c0 = (BETAS_I / (LAMBDAS * PROMPT_LIFETIME)) * n0                       # normalization constants for the delayed-neutron equations

# PARAMETERS OF THE SECONDARY PRESSURE MODEL
P_p, P_s = 157, 34          # nominal pressures of primary and secondary [bar]
P_s0 = P_s                  # nominal secondary pressure [bar]
tau_p  = 5.0               # [s] pressure response time constant
Kp_p   = 52.36e-3             # [bar/(kg/s*s)] pressure sensitivity to secondary flow rate


#=====================================================================================
# SECTION 0 — Correlations for the convective coefficient and other auxiliary functions
#=====================================================================================
def h_conv_semplice (A_flow, Dh, m_dot_ch, mu, k_w, cp, rho):
    """
    Standard convective coefficient (Dittus-Boelter, without geometric factors).

    Used for the flow in the annulus (secondary) and in the inner tube (primary).

    Parameters
    ----------
    A_flow   : flow cross-section area [m^2]
    Dh       : hydraulic diameter [m]
    (other parameters as above)

    Returns
    -------
    h : convective coefficient [W/(m^2*K)]
    """

    Re = rho * (m_dot_ch / (rho * A_flow)) * Dh / mu
    Pr = mu * cp / k_w
    return 0.023 * Re ** 0.8 * Pr ** 0.4 * k_w / Dh


def h_conv_mant (Passo, D_ext, m_dot_channel, mu, k_w, cp, rho):
    """
    Convective coefficient for flow in the shell (shell-side)
    with a triangular tube arrangement.

    Uses the Dittus-Boelter correlation (Nu = 0.023 Re^0.8 Pr^0.4)
    corrected with a geometric factor accounting for the triangular pitch.

    Parameters
    ----------
    Passo    : triangular pitch between tube centers [m]
    D_ext    : outer tube diameter [m]
    m_dot_ch : mass flow rate in the channel [kg/s]
    mu       : dynamic viscosity [Pa*s]
    k_w      : fluid thermal conductivity [W/(m*K)]
    cp       : specific heat [J/(kg*K)]
    rho      : density [kg/m^3]

    Returns
    -------
    h : convective coefficient [W/(m²·K)]
    """

    # Flow cross-section area in the shell (triangular cell minus the tube circle)
    area_flow = (np.sqrt(3) / 4 * Passo**2) - (np.pi * D_ext**2 / 8)

    # Hydraulic diameter for triangular geometry
    Dh = D_ext * ((2 * np.sqrt(3) / np.pi) * (Passo / D_ext)**2 - 1)

    # Reynolds number: Re = rho·v·Dh/mu  (with v = m_dot / (rho·A_flow))
    v  = m_dot_channel / (rho * area_flow)
    Re = (rho * v * Dh) / mu

    # Prandtl number: Pr = mu*cp/k
    Pr = (mu * cp) / k_w

    # Dittus-Boelter with correction factor for triangular pitch
    C_weisman = 1.106 * (Passo / D_ext) - 0.013
    Nu = 0.023 * (Re**0.8) * (Pr**0.4) * C_weisman

    # Heat exchange coefficient h
    h  = (Nu * k_w) / Dh
    return h


# function to compute the total uranium mass, given the mass of uranium-235 (in kg) and the enrichment fraction

def Total_fuel_mass(um235, enrich):
    Na       = 6.0225e23
    MMol235  = 235.0439     # molar mass of U-235
    MMol238  = 238.05       # molar mass of U-238
    atomi235 = um235 * 1000 * Na / MMol235          # number of U-235 atoms
    atomi238 = (1 - enrich) * (atomi235 / enrich)   # number of U-238 atoms
    MMtotal  = (1 - enrich) * MMol238 + enrich * MMol235    # average molar mass of the fuel
    return (atomi238 + atomi235) / Na * MMtotal / 1000      # total fuel mass in kg


# CORE GEOMETRIC VALUES
Passo_core  = 9.69 / 1000           # [m]
H_core      = 1.2                   # [m]
D_f         = 0.0059 - 0.0002*2     # [m], fuel diameter
D_ci        = 0.0059                # [m], cladding inner diameter
D_ce        = 0.0069                # [m], cladding outer diameter
num_ass     = 199                   
num_rod_ass = 69

# Subdivision of the fuel for the calculation of the thermal conductivity coefficient
r1 = (D_f / 2) / np.sqrt(3)
r2 = (D_f / 2) * np.sqrt(2 / 3)
V1 = np.pi * H_core * r1**2
V2 = np.pi * H_core * (r2**2 - r1**2)
V3 = np.pi * H_core * ((D_f / 2)**2 - r2**2)

# FUEL AND CLADDING VALUES
Mf       = Total_fuel_mass(438, 0.1406)
Mf_rod   = Mf / (num_ass * num_rod_ass)
P0       = 175e6         # W
rho_fuel = 10600         # kg/m^3
rho_clad = 6500          # kg/m^3

cp_clad  = 0.4 * 1000    # J/(kg*K)
cp_f     = lambda T: 4.186e-2 * (710 + 0.6 * (T+273.15) - 146 * (T+273.15)**-2 + 4 * ((T+273.15) / 1000)**7) # J/(kg*K)
k_f      = lambda T: (100 / (11.8 + 0.0238 * (T+273.15))) + 8.775e-15 * (T+273.15)**3                        # W/(m*K)
k_clad   = 17            # W/(m*K)

M1 = rho_fuel * V1
M2 = rho_fuel * V2
M3 = rho_fuel * V3
V_clad = np.pi * H_core * ((D_ce/2)**2 - (D_ci/2)**2)
M_clad = rho_clad * V_clad

mu_p   = steamTable.my_pt(P_p, 295.78)
k_w_p  = steamTable.tc_pt(P_p, 295.78)
cp_p   = steamTable.Cp_pt(P_p, 295.78)*1000
rho_p  = steamTable.rho_pt(P_p, 295.78)


# COOLANT VALUES

flowrate   = 902.78     # Total primary flow rate [kg/s]
flowrateII = 77.5       # Total secondary flow rate [kg/s]
Tp0        = 295        # °C, initial average coolant temperature

m_dot_channel = flowrate / (num_ass * num_rod_ass)      # flow rate of a single channel in the primary
area_flow     = (np.sqrt(3) / 4 * Passo_core**2) - (np.pi * (D_ce)**2 / 8)
Mc            = 2 * area_flow * H_core * rho_p          # coolant mass

h_p_core = h_conv_mant(Passo_core, D_ce, m_dot_channel, mu_p, k_w_p, cp_p, rho_p)
h_gap    = 1593.77      # W/(m^2*K)


# THERMAL RESISTANCES
def R_cond_core(r_in, r_out, k):
    return np.log(r_out / r_in) / (2 * np.pi * H_core * k)


# Lumped model for the initial guess, assuming a fuel centerline temperature of 1000 °C and an average coolant temperature of 295 °C (Tp0)

R_fuel_in = 1 / (4 * np.pi * H_core * k_f(1000 + 273.15))
R_gap     = 1 / (2 * np.pi * (D_f/2) * H_core * h_gap)
R_clad    = np.log(D_ce / D_ci) / (2 * np.pi * H_core * k_clad)
R_conv    = 1 / (2 * np.pi * (D_ce/2) * H_core * h_p_core)
R_tot     = R_fuel_in + R_gap + R_clad + R_conv

P_rod             = P0 / (num_ass * num_rod_ass)    # power generated by a single fuel rod
T_clad_ext_guess  = Tp0 + P_rod * R_conv
T_clad_int_guess  = Tp0 + P_rod * (R_conv + R_clad)
T_fuel_surf_guess = Tp0 + P_rod * (R_conv + R_clad + R_gap)
Tf0_guess         = Tp0 + P_rod * R_tot

# STEAM GENERATOR VALUES

h_s_in           = steamTable.h_pt(P_s, 170.0) #assuming a secondary inlet temperature of 170 °C
T_s_in           = 170.0            # °C
L_sg             = 1.9              # m
D_2_int, D_2_ext = 10e-3, 13e-3     # m
D_1_int, D_1_ext =  5e-3,  8e-3     # m
Passo_sg         = D_2_ext * 1.09
k_cond           = 12               # W/m*K
N                = 30
dz               = L_sg / N
n_sub            = 118 * 7 * 12     # number of subchannels

A_mant  = (np.sqrt(3)/2)*Passo_sg**2 - np.pi*D_2_ext**2/4
A_int   = (D_1_int/2)**2 * np.pi
A_ann   = (D_2_int**2 - D_1_ext**2)*np.pi/4
A_tot_I = A_mant + A_int

m_dot_ann  = flowrateII / n_sub
m_dot_I0   = flowrate   / n_sub
m_dot_int0 = m_dot_I0 / (1 + A_mant/A_tot_I)
m_dot_ext0 = m_dot_I0 - m_dot_int0


def _get_sat_props(P):
    """Returns the saturation properties for a given pressure [bar]."""
    hv = steamTable.hV_p(P)
    hl = steamTable.hL_p(P)
    Ts = steamTable.tsat_p(P)
    rL = steamTable.rhoL_p(P)
    rV = steamTable.rhoV_p(P)
    return hv, hl, Ts, rL, rV

hv_sat, hl_sat, Tsat, rhoL_II, rhoV_II = _get_sat_props(P_s)

# thermal resistances of the heat exchanger walls
R_wall_2 = np.log((D_2_ext/2)/(D_2_int/2)) / (2*np.pi*dz*k_cond)
R_wall_1 = np.log((D_1_ext/2)/(D_1_int/2)) / (2*np.pi*dz*k_cond)

z_coord   = np.linspace(0, L_sg, N)                 # cell generation
z_centers = np.linspace(dz/2, L_sg - dz/2, N)       # cell center coordinates

_blend_band = 3.0

# Auxiliary functions

def controlla_fase(h, P_loc=None, hv=None, hl=None, Ts=None):
    """
    Checks the phase of the secondary fluid.
    If P_loc is provided, it uses the corresponding saturation properties;
    otherwise it uses the global values (for compatibility with SS and warmup).
    """
    if P_loc is None:
        _hv, _hl, _Ts = hv_sat, hl_sat, Tsat
    else:
        _hv = hv if hv is not None else steamTable.hV_p(P_loc)
        _hl = hl if hl is not None else steamTable.hL_p(P_loc)
        _Ts = Ts if Ts is not None else steamTable.tsat_p(P_loc)

    if h <= _hl - _blend_band:
        return 'liq', 0.0, steamTable.t_ph(P_loc if P_loc else P_s0, h)
    elif h < _hl:
        alpha = (h - (_hl - _blend_band)) / _blend_band
        T_liq = steamTable.t_ph(P_loc if P_loc else P_s0, h)
        T_eff = T_liq + alpha * (_Ts - T_liq)
        return 'liq', 0.0, T_eff
    elif h >= _hv:
        return 'vap', 1.0, steamTable.t_ph(P_loc if P_loc else P_s0, h)
    else:
        return 'bifase', (h - _hl)/(_hv - _hl), _Ts


def rho_s(fase, x, T_s_loc, P_s_loc = None):
    if P_s_loc is None:
        P_s_loc = P_s0 
        _rL, _rV = rhoL_II, rhoV_II
    else:
        _, _, _, _rL, _rV = _get_sat_props(P_s_loc)
   
    if fase == 'liq':   
        rho_II_l = steamTable.rho_pt(P_s_loc, T_s_loc)
        return rho_II_l
    elif fase == 'vap': 
        rho_II_g = steamTable.rho_pt(P_s_loc, T_s_loc)
        return rho_II_g
    else:              
        return _rV*x + _rL*(1-x)


def calcola_R_I(m_dot_int_loc, m_dot_ext_loc):
    H_int = h_conv_semplice(A_int, D_1_int, m_dot_int_loc, mu_p, k_w_p, cp_p, rho_p)
    H_ext = h_conv_mant(Passo_sg, D_2_ext, m_dot_ext_loc, mu_p, k_w_p, cp_p, rho_p)
    return (1/(2*np.pi*(D_2_ext/2)*dz*H_ext),
            1/(2*np.pi*(D_1_int/2)*dz*H_int))


def calcola_R_tot(fase, x, Rpe, Rpi, T_s_loc, P_s_loc=None):
    if P_s_loc is None: 
        P_s_loc = P_s0
        _hv, _hl, _Ts = hv_sat, hl_sat, Tsat
    else: 
        _hv, _hl, _Ts, _rL, _rV = _get_sat_props(P_s_loc)

    if fase == 'liq':
        mu_II_l  = steamTable.my_pt(P_s_loc, T_s_loc)
        k_w_II_l = steamTable.tc_pt(P_s_loc, T_s_loc)
        cp_II_l  = steamTable.Cp_pt(P_s_loc, T_s_loc)*1000
        rho_II_l = steamTable.rho_pt(P_s_loc, T_s_loc)

        H = h_conv_semplice(A_ann, D_2_int-D_1_ext, m_dot_ann,
                              mu_II_l, k_w_II_l, cp_II_l, rho_II_l)
    elif fase == 'vap':
        mu_II_g  = steamTable.my_pt(P_s_loc, T_s_loc)  
        k_w_II_g = steamTable.tc_pt(P_s_loc, T_s_loc)
        cp_II_g  = steamTable.Cp_pt(P_s_loc, T_s_loc)*1000
        rho_II_g = steamTable.rho_pt(P_s_loc, T_s_loc)

        H = h_conv_semplice(A_ann, D_2_int-D_1_ext, m_dot_ann,
                              mu_II_g, k_w_II_g, cp_II_g, rho_II_g)
    else:
        mu_II_g  = steamTable.my_pt(P_s_loc, (_Ts+0.5))  
        k_w_II_g = steamTable.tc_pt(P_s_loc, (_Ts+0.5))
        cp_II_g  = steamTable.Cp_pt(P_s_loc, (_Ts+0.5))*1000
        rho_II_g = steamTable.rho_pt(P_s_loc, (_Ts+0.5))

        mu_II_l  = steamTable.my_pt(P_s_loc, _Ts-0.5)
        k_w_II_l = steamTable.tc_pt(P_s_loc, _Ts-0.5)
        cp_II_l  = steamTable.Cp_pt(P_s_loc, _Ts-0.5)*1000
        rho_II_l = steamTable.rho_pt(P_s_loc, _Ts-0.5)

        H_l = h_conv_semplice(A_ann, D_2_int-D_1_ext, m_dot_ann,
                              mu_II_l, k_w_II_l, cp_II_l, rho_II_l)
        H_v = h_conv_semplice(A_ann, D_2_int-D_1_ext, m_dot_ann,
                              mu_II_g, k_w_II_g, cp_II_g, rho_II_g)
        if x < 0.8:
            H = H_l * (1.0 + 3.8 * ((x / (1.0 - x))**0.76 * (3.4 / 22.1e6)**0.38))
        else:
            H_08 = H_l * (1.0 + 3.8 * ((0.8 / 0.2)**0.76 * (3.4 / 22.1e6)**0.38))
            H = H_08 + (H_v - H_08) * (x - 0.8) / (1.0 - 0.8)

    return (Rpe + R_wall_2 + 1/(2*np.pi*(D_2_int/2)*dz*H),
            Rpi + R_wall_1 + 1/(2*np.pi*(D_1_ext/2)*dz*H))

Rpe0, Rpi0 = calcola_R_I(m_dot_int0, m_dot_ext0)    # heat exchanger convective resistances

# function to compute the initial conditions of the core

def core_ss(y, T_p_cold):

    T1, T2, T3, Tcl, Tp = y

    # distribution of the thermal power in the fuel
    q1 = P_rod * V1 / (V1 + V2 + V3)
    q2 = P_rod * V2 / (V1 + V2 + V3)
    q3 = P_rod * V3 / (V1 + V2 + V3)

    # fuel thermal resistances as a function of temperature
    R12 = 1 / (4 * np.pi * H_core * k_f(T1))
    R23 = R_cond_core(r1, r2, k_f(T2))
    R3s = R_cond_core(r2, D_f/2, k_f(T3))

    # calculation of the exchanged thermal powers
    Q12 = (T1  - T2) / R12
    Q23 = (T2  - T3) / R23
    Q3s = (T3  - Tcl) / (R3s + R_gap)
    Qcl = (Tcl - Tp) / (R_clad + R_conv)
    
    # residuals (generation + input - output)
    res1  = q1 - Q12
    res2  = q2 + Q12 - Q23
    res3  = q3 + Q23 - Q3s
    resCl = Q3s - Qcl
    resTc = Qcl - 2 * m_dot_channel * cp_p * (Tp - T_p_cold)

    return [res1, res2, res3, resCl, resTc]


def solve_system(T_cold_guess, Tp_hot):
    T_p, hs_rel = T_cold_guess, 0.0
    for _ in range(N):
        fase, x, T_s    = controlla_fase(hs_rel + h_s_in)
        Rpe, Rpi        = calcola_R_tot(fase, x, Rpe0, Rpi0, T_s)
        G               = 1/Rpe + 1/Rpi
        dQ              = max(0.0, T_p - T_s) * G
        hs_rel          += (dQ / m_dot_ann) / 1000
        T_p             += dQ / (m_dot_I0 * cp_p)
    return T_p - Tp_hot     # Residual: difference between the average coolant temperature and the assumed Tp0 of 295 K


# =====================================================
# SECTION 1 — Coupled core-SG steady-state solution
#
# The core and the SG are coupled through the
# primary outlet temperature (T_p_out),
# which is both the input for the core and the output for the SG.
# =====================================================

T_p_cold = 277.0        # °C, initial cold-leg guess

for _ in range(N):
    y_ss   = fsolve(core_ss, [Tf0_guess, (Tf0_guess+T_fuel_surf_guess)/2,
                               T_fuel_surf_guess, T_clad_ext_guess, Tp0], args=(T_p_cold,))
    Tp_ss  = y_ss[4]                    # steady-state coolant temperature, the last output of core_ss
    T_p_hot = 2*(Tp_ss) - T_p_cold      # °C, hot leg
    try:
        T_p_cold_new = root_scalar(lambda x: solve_system(x, T_p_hot),
                                  bracket=[T_s_in, T_p_hot], method='brentq').root
    except ValueError:
        T_p_cold_new = T_p_cold
    if abs(T_p_cold_new - T_p_cold) < 1e-4: break
    T_p_cold = 0.6*T_p_cold + 0.4*T_p_cold_new

T1_ss,T2_ss,T3_ss,Tcl_ss,Tp_ss = y_ss
T_hot_ss  = T_p_hot
T_cold_ss = T_p_cold
print("=== Coupled SS ===")
print(f"  T_hot={T_hot_ss:.2f}°C  T_cold={T_cold_ss:.2f}°C")
print(f"  T_fuel_center={T1_ss:.1f}°C  T_coolant={Tp_ss:.1f}°C")

Tp_list, hs_list = [], []
T_I, hs_rel = T_cold_ss, 0.0
for _ in range(N):
    fase, x, T_II = controlla_fase(hs_rel + h_s_in)
    Tp_list.append(T_I)
    hs_list.append(hs_rel + h_s_in)
    Rpe, Rpi  = calcola_R_tot(fase, x, Rpe0, Rpi0, T_II)
    G       = 1/Rpe + 1/Rpi                 # conductance
    dQ      = max(0.0, T_I - T_II) * G
    hs_rel += (dQ / m_dot_ann) / 1000
    T_I    += dQ / (m_dot_I0 * cp_p)

# In the discrete steady state:  h_impl[i] = h_impl[i-1] + dQ_i / (m_dot * 1000)
#                                   = hs_face[i+1]   (= value at the face of cell i+1)
# shift of one cell upward; the last cell is extrapolated linearly.
# The primary Tp profile does NOT require this adjustment because the shooting
# loop exactly reconstructs the implicit steady-state solution of the primary.
Tp_prof  = np.array(Tp_list)                # primary temperature profile
hs_faces = np.array(hs_list)
h_outlet = hs_rel + h_s_in
hs_prof  = np.append(hs_faces, h_outlet)    # cells 0..N-2: take the value of the next face


def sg_ode(Tp_prof, hs_prof, m_dot_I, m_dot_ann, Rpe, Rpi, dt_loc, T_p_hot,
           P_s_loc=None):
    """
    One implicit step of the SG. Optionally accepts P_s_loc to use the
    saturation properties updated during the transient. Returns Tp_new, hs_new and Q_step [W]

    Counter-current upwind check:
        Primary: from i = N-1 to i = 0, T_new_up = T_new[i+1] (already computed), boundary condition: T_I_in0 at i=N-1
        Secondary: from i = 0 to i = N-1, h_new_up = h_new[i-1] (already computed), boundary condition: h_II_in0 at i=0
    """ 
    # Saturation properties: updated if P_s_loc is provided
    
    if P_s_loc is not None:
        _hv, _hl, _Ts, _rL, _rV = _get_sat_props(P_s_loc)
    else:
        _hv, _hl, _Ts, _rL, _rV = hv_sat, hl_sat, Tsat, rhoL_II, rhoV_II

    Tp_new  = np.empty(N)
    hs_new  = np.empty(N+1)

    # Pre-compute quantities for each cell (from the previous timestep)
    T_II_arr = np.empty(N)
    G_arr    = np.empty(N)      # conductance [W/K]: G = 1/Re + 1/Ri
    rhos_arr = np.empty(N)

    for i in range(N):
        fase, x, T_II = controlla_fase(hs_prof[i], P_s_loc, _hv, _hl, _Ts)
        Re, Ri        = calcola_R_tot(fase, x, Rpe, Rpi, T_II, P_s_loc)
        T_II_arr[i]   = T_II
        G_arr[i]      = 1.0/Re + 1.0/Ri
        rhos_arr[i]   = rho_s(fase, x, T_II, P_s_loc)

    # Primary sweep: from i=N-1 to i=0
    alpha = rho_p * A_tot_I * dz / dt_loc   #  [kg/s]

    for i in range(N-1, -1, -1):
        # T_new[i+1] is already computed because the sweep goes from N-1 to 0
        T_new_up = Tp_new[i+1] if i < N-1 else T_p_hot

        # Case 1: active exchange (T_new[i] > T_II_old[i])
        Gcp   = G_arr[i] / cp_p
        T_new = (alpha * Tp_prof[i] + m_dot_I * T_new_up + Gcp * T_II_arr[i]) \
                / (alpha + m_dot_I + Gcp)
        
        # Case 2: if T_new < T_II there is no heat exchange
        if T_new < T_II_arr[i]:
            T_new = (alpha * Tp_prof[i] + m_dot_I * T_new_up) / (alpha + m_dot_I)
        Tp_new[i] = T_new

    # Secondary sweep: from i=0 to i=N-1
    h_s_in_loc = steamTable.h_pt(P_s_loc, T_s_in) if P_s_loc is not None else h_s_in
    hs_new[0] = h_s_in_loc

    for i in range(N):
        # h_new[i-1] is already computed because the sweep goes from 0 to N-1
        beta_i  = rhos_arr[i] * A_ann * dz / dt_loc     # [kg/s]

        # semi-implicit dQ: uses T_new[i] already computed + T_II_old[i]
        dQ_i    = max(0.0, Tp_new[i] - T_II_arr[i]) * G_arr[i]


        hs_new[i+1] = (beta_i * hs_prof[i+1] + m_dot_ann * hs_new[i] + dQ_i/1000) \
                      / (beta_i + m_dot_ann)

    # Q with the state consistent at t+dt
    Q_step = 0.0
    for i in range(N):
        fase, x, T_II_n = controlla_fase(hs_new[i], P_s_loc, _hv, _hl, _Ts)
        Rpe_n, Rpi_n    = calcola_R_tot(fase, x, Rpe, Rpi, T_II_n, P_s_loc)
        Q_step         += max(0.0, Tp_new[i] - T_II_n) * (1.0/Rpe_n + 1.0/Rpi_n)

    return Tp_new, hs_new, Q_step


# DEFINITION OF THE DIFFERENTIAL EQUATION FOR THE FUEL AND COOLANT TEMPERATURE EVOLUTION

def core_ode(t, y, T_p_cold, h=None):
    if h is None:
        h = h0
    T1  = y[0]; T2  = y[1]; T3  = y[2]; Tcl = y[3]; Tp  = y[4]
    n   = y[5]; c   = y[6:]

    P  = P0 / (num_rod_ass * num_ass)
    q1 = P * V1 / (V1 + V2 + V3) * n
    q2 = P * V2 / (V1 + V2 + V3) * n
    q3 = P * V3 / (V1 + V2 + V3) * n

    R12 = 1 / (4 * np.pi * H_core * k_f(T1))
    R23 = R_cond_core(r1, r2, k_f(T2))
    R3s = R_cond_core(r2, D_f/2, k_f(T3))

    Q12 = (T1 - T2) / R12
    Q23 = (T2 - T3) / R23
    Q3s = (T3 - Tcl) / (R3s + R_gap)
    Qcl = (Tcl - Tp) / (R_clad + R_conv)

    dndt   = ((reactivity(T2, Tp, h) - BETA) / PROMPT_LIFETIME) * n + np.dot(LAMBDAS, c)
    dcidt  = (BETAS_I / PROMPT_LIFETIME) * n - LAMBDAS * c
    dT1dt  = (q1 - Q12) / (M1 * cp_f(T1))
    dT2dt  = (q2 + Q12 - Q23) / (M2 * cp_f(T2))
    dT3dt  = (q3 + Q23 - Q3s) / (M3 * cp_f(T3))
    dTcldt = (Q3s - Qcl) / (M_clad * cp_clad)
    dTpdt  = (Qcl - 2 * m_dot_channel * cp_p * (Tp - T_p_cold)) / (Mc * cp_p)

    return [dT1dt, dT2dt, dT3dt, dTcldt, dTpdt, dndt, *dcidt]


reactivity = lambda TF, TM, H: (af*(float(TF)-T2_ss) +
                                am*(float(TM)-Tp_ss) +
                                ah*(float(H)-h0))

# ============================================================================================================
# SECTION 2 — Implicit warmup with convergence criterion
# To reach the discrete steady state, we run a warmup using the implicit scheme just defined.
#
# With the implicit method dt_wu = 1.0 s is used (since there is no CFL constraint).
# Primary tau_adv is about L / v_prim = 1-2 s  --> 50-100 steps are enough
# Secondary tau_adv is about L / v_ann = 5-10 s --> 200-300 steps are enough
# In practice it converges in <500 steps instead of >50,000 for the explicit scheme.
# ============================================================================================================

print("\nRunning Coupled Warmup...")
dt_wu    = 1       # [s] — large because the implicit scheme is unconditionally stable
passi_wu = 1000
y_core   = [T1_ss, T2_ss, T3_ss, Tcl_ss, Tp_ss, n0, *c0]

for wu in range(passi_wu):
    m_dot_ann_wu = flowrateII / n_sub
    m_dot_I_wu   = flowrate / n_sub
    m_dot_int_wu = m_dot_I_wu / (1 + A_mant/A_tot_I)
    m_dot_ext_wu = m_dot_I_wu - m_dot_int_wu

    Rpe_wu, Rpi_wu = calcola_R_I(m_dot_int_wu, m_dot_ext_wu)
    Tp_mean_wu = y_core[4]

    T_p_cold_pred = Tp_prof[0]
    T_p_hot_pred  = 2 * Tp_mean_wu - T_p_cold_pred
    Tp_pred, hs_pred, Q_pred = sg_ode(Tp_prof, hs_prof, m_dot_I_wu, m_dot_ann_wu,
                                       Rpe_wu, Rpi_wu, dt_wu, T_p_hot_pred, P_s0)

    T_p_cold_corr = Tp_pred[0]
    T_p_hot_wu    = 2 * Tp_mean_wu - T_p_cold_corr
    Tp_prof, hs_prof, Q_cell_ss = sg_ode(Tp_prof, hs_prof, m_dot_I_wu, m_dot_ann_wu,
                                          Rpe_wu, Rpi_wu, dt_wu, T_p_hot_wu, P_s0)

    T_p_cold_wu = Tp_prof[0]
    sol_wu = solve_ivp(core_ode, [0, dt_wu], y_core, args=(T_p_cold_wu,), method='LSODA')  # BDF and Radau can also be used
    y_core = sol_wu.y[:, -1]

print("-> Warmup completed.")

# Final conditions for the core steady state
T_cold_ss = Tp_prof[0]
T1_ss, T2_ss, T3_ss, Tcl_ss, Tp_ss, n0, *c0 = y_core
T_hot_ss  = Tp_ss*2 - T_cold_ss
Ts_prof   = [controlla_fase(h, P_s0)[2] for h in hs_prof]
Ts_out_ss = Ts_prof[-1]
hs_out_ss = hs_prof[-1]

Q_prim_ss = m_dot_I0   * cp_p  * (T_hot_ss - Tp_prof[0])
Q_sec_ss  = m_dot_ann  * 1000  * (hs_prof[N] - h_s_in)

Gtot_ss_prof = np.empty(N)
Re_ss_prof   = np.empty(N)
Ri_ss_prof   = np.empty(N)
x_ss_prof    = np.empty(N)
q_ss_prof    = np.empty(N)

for i in range(N):
    fase_i, x_i, T_II_i = controlla_fase(hs_prof[i], P_s0)
    Re_i, Ri_i = calcola_R_tot(fase_i, x_i, Rpe0, Rpi0, T_II_i, P_s0)
    G_i = 1.0/Re_i + 1.0/Ri_i

    Re_ss_prof[i] = Re_i
    Ri_ss_prof[i] = Ri_i
    Gtot_ss_prof[i] = G_i
    x_ss_prof[i] = x_i
    q_ss_prof[i] = max(0.0, Tp_prof[i] - T_II_i) * G_i

Gtot_ss = Gtot_ss_prof.sum()
Re_ss   = Re_ss_prof.sum()
Ri_ss   = Ri_ss_prof.sum()
x_mean_ss    = x_ss_prof.mean()

print(f"\nDiscrete SS Q check:")
print(f"  cell-sum      : {Q_cell_ss:.2f} W  ( Total = {Q_cell_ss*n_sub/1e6:.4f} MW)")
print(f"  primary bal.  : {Q_prim_ss:.2f} W  ( Total = {Q_prim_ss*n_sub/1e6:.4f} MW)")
print(f"  secondary bal.: {Q_sec_ss:.2f} W  ( Total = {Q_sec_ss*n_sub/1e6:.4f} MW)")
print(f"  primary-secondary imbalance: {abs(Q_prim_ss-Q_sec_ss)/Q_prim_ss*100:.5f}%")

# core steady-state conditions
print("\nSteady state")
print(f"T_fuel center      = {T1_ss:.2f} °C")
print(f"T_fuel intermediate= {T2_ss:.2f} °C")
print(f"T_fuel surface     = {T3_ss:.2f} °C")
print(f"T_clad             = {Tcl_ss:.2f} °C")
print(f"T_coolant          = {Tp_ss:.2f} °C")
print(f"T_cold_leg         = {T_cold_ss:.2f} °C")
print(f"T_hot_leg          = {T_hot_ss:.2f} °C")
print(f"T_secondary_outlet = {Ts_out_ss:.2f} °C")
print(f"P_s (nominal)      = {P_s0:.2f} bar")

# Plot of the heat exchanger profile at steady state
plt.figure(figsize=(10, 5))
z_faces = np.linspace(0, L_sg, N+1)

# primary, cell centers and boundary condition at z=L
plt.plot(np.concatenate([z_centers, [L_sg]]), list(Tp_prof) + [T_hot_ss],
         'r', lw=2, label='Primario')

# secondary, boundary condition at z=0 and cell centers
plt.plot(z_faces, Ts_prof, 'b', lw=2, label='Secondario')
plt.title("Profilo T — SS (punto di partenza warmup)")
plt.xlabel("z [m]"); plt.ylabel("T [°C]")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()


# =========================================================================
# SECTION 3 — Implicit transient with secondary flow-rate variation
# =========================================================================

y_core = [T1_ss, T2_ss, T3_ss, Tcl_ss, Tp_ss, n0, *c0]
dt, t_end, t_salto = 0.2, 50.0, 10.0
nt = int(t_end / dt)
tau_fr = 5.0

# Secondary pressure state
P_s_current = P_s0   # [bar]


def flowrate_t(t):
    """Total secondary flow rate [kg/s] as a function of time."""
    if t < t_salto:
        return flowrateII
    return flowrateII * (1.0 + delta_fr * (1.0 - np.exp(-(t - t_salto) / tau_fr)))


# History array initialization
T1_hist      = []
T2_hist      = []
T3_hist      = []
Tcl_hist     = []
Tp_hist      = []
n_hist       = []
T_cold_hist  = []
T_hot_hist   = []
T_s_out_hist = []
h_s_hist     = []
Q_cell_hist  = []
Q_prim_hist  = []
Q_sec_hist   = []
P_s_hist     = []  
t_hist       = []
frs_arr      = []
Gtot_hist    = []
Re_hist      = []
Ri_hist      = []
x_mean_hist  = []

print(f"\nTransient 0-{t_end} s ({nt} steps, dt={dt} s) — implicit scheme with dynamic P_s...")

for n_step in range(nt):
    t = n_step * dt

    # Secondary flow rate at time t
    frs_t = flowrate_t(t)
    frs_arr.append(frs_t)

    # Secondary pressure update
    # Model: dP_s/dt = -(1/tau_p)*(P_s - P_s0) - Kp_p*(frs_t - frs_nom)
    #
    # If the secondary flow rate increases, the pressure drops; the restoring
    # term (P_s - P_s0)/tau_p ensures that at the new steady state the
    # pressure settles at a finite value
    #
    # With delta_fr > 0 (flow-rate increase):
    #   - frs_t > flowrateII  :  Kp_p term negative, so P_s drops
    # With delta_fr < 0 (flow-rate decrease):
    #   - frs_t < flowrateII  :  Kp_p term positive, so P_s rises
    dPs_dt      = -(1.0 / tau_p) * (P_s_current - P_s0) - Kp_p * (frs_t - flowrateII)
    P_eq = P_s0 - tau_p*Kp_p*(frs_t - flowrateII)     
    P_s_current = P_eq + (P_s_current - P_eq)*np.exp(-dt/tau_p) 

    # Primary flow rate and geometry
    m_dot_ann     = frs_t / n_sub
    fr_t          = flowrate
    m_dot_I       = fr_t / n_sub
    m_dot_int     = m_dot_I / (1 + A_mant/A_tot_I)
    m_dot_ext     = m_dot_I - m_dot_int
    m_dot_channel = fr_t / (num_ass * num_rod_ass)
    Rce, Rci      = calcola_R_I(m_dot_int, m_dot_ext)

    # Implicit steam generator step with updated P_s
    Tp_mean   = y_core[4]
    T_p_cold   = Tp_prof[0]
    T_p_hot    = 2 * Tp_mean - T_p_cold

    Tp_prof, hs_prof, Q_step = sg_ode(Tp_prof, hs_prof, m_dot_I, m_dot_ann,
                                       Rce, Rci, dt, T_p_hot,
                                       P_s_loc=P_s_current)   #  dynamic P_s

    T_p_cold_new = Tp_prof[0]

    # Secondary phase and outlet temperature with updated P_s
    _hv_c, _hl_c, _Ts_c, _, _ = _get_sat_props(P_s_current)
    T_s_out_new = controlla_fase(hs_prof[-1], P_s_current, _hv_c, _hl_c, _Ts_c)[2]

    # Core step
    sol = solve_ivp(core_ode, [t, t + dt], y_core,
                    args=(T_p_cold_new, h), method='LSODA', rtol=1e-6, atol=1e-8)
    y_core = sol.y[:, -1]

    # Energy balances
    # The secondary inlet enthalpy can vary with P_s:
    _h_s_in_c = steamTable.h_pt(P_s_current, T_s_in)
    Q_prim = m_dot_I   * cp_p  * (T_p_hot - T_p_cold_new)
    Q_sec  = m_dot_ann * 1000  * (hs_prof[N] - _h_s_in_c)


    _hv_d, _hl_d, _Ts_d, _, _ = _get_sat_props(P_s_current)
  
    Gtot_step = np.empty(N)
    Re_step   = np.empty(N)
    Ri_step   = np.empty(N)
    x_step    = np.empty(N)
    for i in range(N):
        fase_i, x_i, T_s_i = controlla_fase(hs_prof[i], P_s_current,
                                         _hv_d, _hl_d, _Ts_d)
        Re_i, Ri_i = calcola_R_tot(fase_i, x_i, Rce, Rci, T_s_i, P_s_current)
        Re_step[i] = Re_i
        Ri_step[i] = Ri_i
        Gtot_step[i] = 1.0/Re_i + 1.0/Ri_i
        x_step[i] = x_i
    # History saving
    T1_hist.append(float(y_core[0]))
    T2_hist.append(float(y_core[1]))
    T3_hist.append(float(y_core[2]))
    Tcl_hist.append(float(y_core[3]))
    Tp_hist.append(float(y_core[4]))
    n_hist.append(float(y_core[5]))
    T_cold_hist.append(float(T_p_cold_new))
    T_hot_hist.append(float(T_p_hot))
    T_s_out_hist.append(float(T_s_out_new))
    h_s_hist.append(float(hs_prof[-1]))
    Q_cell_hist.append(Q_step)
    Q_prim_hist.append(Q_prim)
    Q_sec_hist.append(Q_sec)
    P_s_hist.append(P_s_current)   
    t_hist.append(t + dt)
    Gtot_hist.append(float(Gtot_step.sum()))
    Re_hist.append(float(Re_step.sum()))
    Ri_hist.append(float(Ri_step.sum()))
    x_mean_hist.append(float(x_step.mean()))


# Concatenation with the initial conditions
T1_arr      = np.concatenate([[T1_ss],     T1_hist])
T2_arr      = np.concatenate([[T2_ss],     T2_hist])
T3_arr      = np.concatenate([[T3_ss],     T3_hist])
Tcl_arr     = np.concatenate([[Tcl_ss],    Tcl_hist])
Tp_arr      = np.concatenate([[Tp_ss],     Tp_hist])
n_arr       = np.concatenate([[n0],        n_hist])
T_cold_arr   = np.concatenate([[T_cold_ss], T_cold_hist])
T_hot_arr    = np.concatenate([[T_hot_ss],  T_hot_hist])
T_s_out_arr = np.concatenate([[Ts_out_ss], T_s_out_hist])
h_s_arr     = np.concatenate([[hs_out_ss], h_s_hist])
Q_cell_arr  = np.concatenate([[Q_cell_ss], Q_cell_hist])
Q_prim_arr  = np.concatenate([[Q_prim_ss], Q_prim_hist])
Q_sec_arr   = np.concatenate([[Q_sec_ss],  Q_sec_hist])
P_s_arr     = np.concatenate([[P_s0],      P_s_hist])   
t_arr       = np.concatenate([[0.0],       t_hist])
frs_full    = np.concatenate([[flowrateII], frs_arr])
Gtot_arr = np.concatenate([[Gtot_ss], Gtot_hist])
Re_arr   = np.concatenate([[Re_ss],   Re_hist])
Ri_arr   = np.concatenate([[Ri_ss],   Ri_hist])
x_mean_arr    = np.concatenate([[x_mean_ss],    x_mean_hist])


_hv_f, _hl_f, _Ts_f, _, _ = _get_sat_props(P_s_current)

Gtot_final_prof = np.empty(N)
Re_final_prof   = np.empty(N)
Ri_final_prof   = np.empty(N)
x_final_prof    = np.empty(N)
q_final_prof    = np.empty(N)

for i in range(N):
    fase_i, x_i, T_II_i = controlla_fase(hs_prof[i], P_s_current,
                                         _hv_f, _hl_f, _Ts_f)
    Re_i, Ri_i = calcola_R_tot(fase_i, x_i, Rce, Rci, T_II_i, P_s_current)
    G_i = 1.0/Re_i + 1.0/Ri_i

    Re_final_prof[i] = Re_i
    Ri_final_prof[i] = Ri_i
    Gtot_final_prof[i] = G_i
    x_final_prof[i] = x_i
    q_final_prof[i] = max(0.0, Tp_prof[i] - T_II_i) * G_i

t_fin = time.perf_counter()
print(f"Total time: {t_fin - t_start:.2f} s")

# ============================================================
# PLOT
# ============================================================

idx  = int(t_salto / dt)
win  = max(1, int(1.0 / dt))
if delta_fr > 0:
    Q_pre  = Q_cell_arr[max(0, idx-win):idx].mean()
    Qp_pre = Q_prim_arr[max(0, idx-win):idx].mean()
else: 
    Q_pre = Q_cell_ss
    Qp_pre = Q_prim_ss
Q_post = Q_cell_arr[-win:].mean()
Qp_post= Q_prim_arr[-win:].mean()
Ps_pre = P_s_arr[max(0, idx-win):idx].mean()
Ps_post= P_s_arr[-win:].mean()

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()

axs[0].plot(t_arr, frs_full, color='green', lw=2, label='Secondary flow-rate')
axs[0].set_ylabel('Flow-rate [kg/s]')
axs[0].set_title('Secondary flow-rate')
axs[0].legend(loc='upper left')
axs[0].grid(True, linestyle=':')

axs[1].plot(t_arr, n_arr, color='blue', lw=2, label='Neutronic Density')
axs[1].set_ylabel('n/n₀')
axs[1].set_title('Neutronic Density')
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle=':')

axs[2].plot(t_arr, P_s_arr, color='purple', lw=2, label='Secondary P')
axs[2].axhline(P_s0, color='black', ls='--', alpha=0.5, label=f'P_s0 = {P_s0:.1f} bar')
axs[2].axvline(t_salto, color='gray', ls=':', lw=1.2)
axs[2].set_ylabel('Pressure [bar]')
axs[2].set_title('Secondary pressure')
axs[2].legend(loc='upper left')
axs[2].grid(True, linestyle=':')

axs[3].plot(t_arr, Q_cell_arr*n_sub/1e6, color='steelblue', lw=1.8, label='Q cell-sum')
axs[3].plot(t_arr, Q_prim_arr*n_sub/1e6, color='tomato', lw=1.4, ls='--', label='Q bil. primary')
axs[3].plot(t_arr, Q_sec_arr*n_sub/1e6, color='seagreen', lw=1.4, ls=':', label='Q bil. secondary')
axs[3].axvline(t_salto, color='navy', ls='--', lw=1.4, label=f'Jump {delta_fr*100:+.0f}% @ t={t_salto} s')
axs[3].axhline(Q_pre*n_sub/1e6, color='gray', ls=':', lw=1.2,
              label=f'Q pre  = {Q_pre*n_sub/1e6:.3f} MW')
axs[3].axhline(Q_post*n_sub/1e6, color='darkorange', ls=':', lw=1.2,
              label=f'Q post = {Q_post*n_sub/1e6:.3f} MW  ({(Q_post/Q_pre-1)*100:+.1f}%)')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Q [MW]')
axs[3].set_title('Transient powers')
axs[3].legend(fontsize=9, loc='lower right' if delta_fr >= 0 else 'upper right')
axs[3].grid(alpha=0.35)

fig.suptitle(f'Transient behavior (Δflow {delta_fr*100:+.0f}%)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()

axs[0].plot(t_arr, T1_arr, color='red', label="Fuel Centerline")
axs[0].axhline(T1_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[0].set_title(f"Fuel Centerline Temperature")
axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Temperature (°C)")
axs[0].grid(True, linestyle=':'); axs[0].legend()

axs[1].plot(t_arr, T3_arr, color='orange', lw=2, label='Fuel Surface')
axs[1].axhline(T3_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[1].set_title(f"Fuel Surface Temperature")
axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Temperature (°C)")
axs[1].grid(True, linestyle=':'); axs[1].legend() 

axs[2].plot(t_arr, Tcl_arr, color='brown', lw=2, label='Cladding')
axs[2].axhline(Tcl_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[2].set_title(f"Cladding Temperature")
axs[2].set_xlabel("Time (s)"); axs[2].set_ylabel("Temperature (°C)")
axs[2].grid(True, linestyle=':'); axs[2].legend()

axs[3].plot(t_arr, Gtot_arr/(L_sg*D_1_ext*np.pi + L_sg*D_2_int*np.pi), color='green', lw=2, label='Overall heat transfer coefficient (G)')
axs[3].set_ylabel(' U ( W/m²/K ) ')
axs[3].legend(loc='upper left')
axs[3].grid(True, linestyle=':')

fig.suptitle(f'Transient behavior (Δflow {delta_fr*100:+.0f}%)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()

axs[0].plot(t_arr, Tp_arr, color='blue', label="Mean Coolant")
axs[0].axhline(Tp_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[0].set_title(f"Average Coolant Temperature — Flow rate variation {delta_fr*100:+.0f}%")
axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Temperature (°C)")
axs[0].grid(True, linestyle=':'); axs[0].legend()

axs[1].plot(t_arr, T_cold_arr, color='blue', label="Primary output T (cold leg)")
axs[1].axhline(T_cold_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[1].set_title(f"OutletTemperature SG (Cold Leg)")
axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Temperature (°C)")
axs[1].grid(True, linestyle=':'); axs[1].legend()

axs[2].plot(t_arr, T_hot_arr, color='blue', label="T inlet SG (hot leg)")
axs[2].axhline(T_hot_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[2].set_title(f"Inlet Temperature SG (Hot Leg)")
axs[2].set_xlabel("Time (s)"); axs[2].set_ylabel("Temperature (°C)")
axs[2].grid(True, linestyle=':'); axs[2].legend()

axs[3].plot(t_arr, T_s_out_arr, color='blue', label="Secondary output T")
axs[3].axhline(Ts_out_ss, color='black', ls='--', alpha=0.5, label='SS')
axs[3].set_title(f"Secondary Outlet Temperature")
axs[3].set_xlabel("Time (s)"); axs[3].set_ylabel("Temperatre (°C)")
axs[3].grid(True, linestyle=':'); axs[3].legend()

fig.suptitle(f'Transient behavior (Δflow {delta_fr*100:+.0f}%)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()  


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(t_arr, Gtot_arr, color='darkgreen', lw=2, label='Average Gtot')
ax[0, 0].axhline(Gtot_ss, color='black', ls='--', alpha=0.5, label='SS')
ax[0, 0].axvline(t_salto, color='gray', ls=':', lw=1.2)
ax[0, 0].set_title("Average conductance SG")
ax[0, 0].set_xlabel("Time (s)"); ax[0, 0].set_ylabel("Gtot [W/K]")
ax[0, 0].grid(True, linestyle=':'); ax[0, 0].legend()

ax[0, 1].plot(t_arr, Re_arr, color='sienna', lw=2, label='Mean Re')
ax[0, 1].axhline(Re_ss, color='black', ls='--', alpha=0.5, label='SS')
ax[0, 1].axvline(t_salto, color='gray', ls=':', lw=1.2)
ax[0, 1].set_title("Average thermal resistance, outer side")
ax[0, 1].set_xlabel("Time (s)"); ax[0, 1].set_ylabel("Re [K/W]")
ax[0, 1].grid(True, linestyle=':'); ax[0, 1].legend()

ax[1, 0].plot(t_arr, Ri_arr, color='darkcyan', lw=2, label='Mean Ri')
ax[1, 0].axhline(Ri_ss, color='black', ls='--', alpha=0.5, label='SS')
ax[1, 0].axvline(t_salto, color='gray', ls=':', lw=1.2)
ax[1, 0].set_title("Average thermal resistance, inner side")
ax[1, 0].set_xlabel("Time (s)"); ax[1, 0].set_ylabel("Ri [K/W]")
ax[1, 0].grid(True, linestyle=':'); ax[1, 0].legend()

ax[1, 1].plot(t_arr, x_mean_arr, color='black', lw=2, label='Average x')
ax[1, 1].axhline(x_mean_ss, color='black', ls='--', alpha=0.5, label='SS')
ax[1, 1].axvline(t_salto, color='gray', ls=':', lw=1.2)
ax[1, 1].set_title("Mean secondary quality")
ax[1, 1].set_xlabel("TTime (s)"); ax[1, 1].set_ylabel("x [-]")
ax[1, 1].grid(True, linestyle=':'); ax[1, 1].legend()
plt.tight_layout(); plt.show()


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(z_centers, Gtot_ss_prof, color='darkgreen', lw=2, label='SS')
ax[0, 0].plot(z_centers, Gtot_final_prof, color='limegreen', lw=2, ls='--', label='final')
ax[0, 0].set_title("Axial profile Gtot")
ax[0, 0].set_xlabel("z [m]"); ax[0, 0].set_ylabel("Gtot [W/K]")
ax[0, 0].grid(True, linestyle=':'); ax[0, 0].legend()

ax[0, 1].plot(z_centers, Re_ss_prof, color='sienna', lw=2, label='Re SS')
ax[0, 1].plot(z_centers, Re_final_prof, color='peru', lw=2, ls='--', label='Re final')
ax[0, 1].plot(z_centers, Ri_ss_prof, color='darkcyan', lw=2, label='Ri SS')
ax[0, 1].plot(z_centers, Ri_final_prof, color='deepskyblue', lw=2, ls='--', label='Ri final')
ax[0, 1].set_title("Axial profiles of Re and Ri")
ax[0, 1].set_xlabel("z [m]"); ax[0, 1].set_ylabel("R [K/W]")
ax[0, 1].grid(True, linestyle=':'); ax[0, 1].legend()

ax[1, 0].plot(z_centers, x_ss_prof, color='black', lw=2, label='SS')
ax[1, 0].plot(z_centers, x_final_prof, color='gray', lw=2, ls='--', label='final')
ax[1, 0].set_title("Secondary quality")
ax[1, 0].set_xlabel("z [m]"); ax[1, 0].set_ylabel("x [-]")
ax[1, 0].grid(True, linestyle=':'); ax[1, 0].legend()

ax[1, 1].plot(z_centers, q_ss_prof*n_sub/1e6, color='steelblue', lw=2, label='SS')
ax[1, 1].plot(z_centers, q_final_prof*n_sub/1e6, color='tomato', lw=2, ls='--', label='final')
ax[1, 1].set_title("Local Power exchange")
ax[1, 1].set_xlabel("z [m]"); ax[1, 1].set_ylabel("Local Q [MW]")
ax[1, 1].grid(True, linestyle=':'); ax[1, 1].legend()
plt.tight_layout(); plt.show()


print(f"\n=== Transient summary ===")
print(f"  Secondary flow rate variation : {delta_fr*100:+.1f}%")
print(f"  P_s pre  = {Ps_pre:.3f} bar   -->   P_s post = {Ps_post:.3f} bar   (ΔP = {Ps_post-Ps_pre:+.3f} bar)")
print(f"  Q pre    = {Q_pre*n_sub/1e6:.3f} MW  -->   Q post   = {Q_post*n_sub/1e6:.3f} MW  ({(Q_post/Q_pre-1)*100:+.1f}%)")
