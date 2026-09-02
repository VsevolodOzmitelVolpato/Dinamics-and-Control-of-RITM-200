import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def h_conv_c(Passo, D_ext, m_dot_channel, mu, k_w, cp, rho):
    # Calcolo Diametro Idraulico (Dh) per triangular pitch
    # Area fluida della singola cella (triangolo tra 3 barre)
    area_flow = (np.sqrt(3) / 4 * Passo**2) - (np.pi * D_ext**2 / 8)

    Dh = D_ext * ((2 * np.sqrt(3) / np.pi) * (Passo / D_ext)**2 - 1) 

    #Velocità e Reynolds
    v = m_dot_channel / (rho * area_flow)
    Re = (rho * v * Dh) / mu

    #Prandtl
    Pr = (mu * cp) / k_w
    #Nusselt (Dittus-Boelter + Correzione Weisman per Rod Bundles)
    C_weisman = 1.106 * (Passo / D_ext) - 0.013
    Nu = 0.023 * (Re**0.8) * (Pr**0.4) * C_weisman

    #Coefficiente di scambio h
    h = (Nu * k_w) / Dh

    return h, Dh


def Total_fuel_mass(um235, enrich):

    Na = 6.0225e23
    MMol235 = 235.0439 #mass molare U-235
    MMol238 = 238.05 #mass molare U-238
    atomi235 = um235 * 1000 * Na / MMol235 #numero di atomi di U-235 in 1 g di combustibile
    atomi238 = (1 - enrich) * (atomi235 / enrich) #numero di atomi di U-238 in 1 g di combustibile
    MMtotal = (1 - enrich) * MMol238 + enrich * MMol235 #massa molare media del combustibile

    return (atomi238 + atomi235) / Na * MMtotal / 1000 # massa totale del combustibile in kg


#VALORI GEOMETRICI CORE
Passo_ch = 9.69 / 1000 # m
H_core = 1.2 # m
r_f = 0.0059 / 2 - 0.0002 # m, raggio fuel
r_ci = 0.0059 / 2 # m, raggio interno clad
r_ce = 0.0069 / 2 # m, raggio esterno clad
num_ass = 199
num_rod_ass = 69

# Suddivisione del fuel per il coefficiente di conduttività
r1 = r_f / np.sqrt(3)
r2 = r_f * np.sqrt(2 / 3)
V1 = np.pi * H_core * r1**2
V2 = np.pi * H_core * (r2**2 - r1**2)
V3 = np.pi * H_core * (r_f**2 - r2**2)


#VALORI FUEL e CLADDING
Mf = Total_fuel_mass(438, 0.1406)
Mf_rod = Mf / (num_ass * num_rod_ass)
P0 = 175e6 # W
rho_fuel = 10600 # kg/m³
rho_clad = 6500 # kg/m³

cp_clad = 0.4 * 1000 # J/(kg*K)
cp_f = lambda T: 4.186e-2 * (710 + 0.6 * T - 146 * T**-2 + 4 * (T / 1000)**7) # J/(kg*K) #CORRELAZIONE UTILIZZATA DA GAGANOV
k_f = lambda T: (100 / (11.8 + 0.0238 * T)) + 8.775e-15 * T**3 # W/(m*K)
k_clad = 17 # W/(m*K)

M1 = rho_fuel * V1
M2 = rho_fuel * V2
M3 = rho_fuel * V3
V_clad = np.pi * H_core * (r_ce**2 - r_ci**2)
M_clad = rho_clad * V_clad


#VALORI COOLANT
flowrate = 902.78 # kg/s

T_inlet = 277 + 273.15 # K, temperatura di ingresso
Tc0 = 295 + 273.15 # K, temperatura media iniziale

mu_c = 8.42e-05 # Pa*s
k_w_c = 0.5384 # W/(m*K)
cp_c = 5.78 * 1000 # J/(kg*K)
rho_c = 702.45 # kg/m³
m_dot_channel = flowrate / (num_ass * num_rod_ass)
area_flow = (np.sqrt(3) / 4 * Passo_ch**2) - (np.pi * (r_ce * 2)**2 / 8)
Mc = 2 * area_flow * H_core * rho_c

h_c, Dh = h_conv_c(Passo_ch, r_ce * 2, m_dot_channel, mu_c, k_w_c, cp_c, rho_c)
h_gap = 1593.77 # W/(m²*K)


#RESISTENZE TERMICHE
def R_cond(r_in, r_out, k):
    return np.log(r_out / r_in) / (2 * np.pi * H_core * k)

def R_gap_fun():
    return 1 / (2 * np.pi * r_f * H_core * h_gap)

def R_clad_fun():
    return np.log(r_ce / r_ci) / (2 * np.pi * H_core * k_clad)

def R_conv_fun():
    return 1 / (2 * np.pi * r_ce * H_core * h_c)

# Modello concentrato per la guess iniziale
R_fuel_in = 1 / (4 * np.pi * H_core * k_f(1000 + 273.15))
R_gap = R_gap_fun()
R_clad = R_clad_fun()
R_conv = R_conv_fun()
R_tot = R_fuel_in + R_gap + R_clad + R_conv

P_rod = P0 / (num_ass * num_rod_ass)
T_clad_ext_guess = Tc0 + P_rod * R_conv
T_clad_int_guess = Tc0 + P_rod * (R_conv + R_clad)
T_fuel_surf_guess = Tc0 + P_rod * (R_conv + R_clad + R_gap)
Tf0_guess = Tc0 + P_rod * R_tot


# Calcolo condizioni iniziali con fsolve
def residuals_ss(y):
    
    T1, T2, T3, Tcl, Tc = y

    #ripartizione della potenza termica nel fuel
    q1 = P_rod * V1 / (V1 + V2 + V3)
    q2 = P_rod * V2 / (V1 + V2 + V3)
    q3 = P_rod * V3 / (V1 + V2 + V3)

    #resistenze termiche fuel in funzione della temperatura
    R12 = 1 / (4 * np.pi * H_core * k_f(T1))
    R23 = R_cond(r1, r2, k_f(T2))
    R3s = R_cond(r2, r_f, k_f(T3))
    R_gap = R_gap_fun()
    R_cld = R_clad_fun()
    R_cnv = R_conv_fun()

    #calcolo delle potenze termiche scambiate
    Q12 = (T1  - T2) / R12
    Q23 = (T2  - T3) / R23
    Q3s = (T3  - Tcl) / (R3s + R_gap)
    Qcl = (Tcl - Tc) / (R_cld + R_cnv)

    # residui (generazione + ingresso - uscita)
    res1 = q1 - Q12
    res2 = q2 + Q12 - Q23
    res3 = q3 + Q23 - Q3s
    resCl = Q3s - Qcl
    resTc = Qcl - 2 * m_dot_channel * cp_c * (Tc - T_inlet)

    return [res1, res2, res3, resCl, resTc]

y_guess = [Tf0_guess, (Tf0_guess + T_fuel_surf_guess) / 2, T_fuel_surf_guess, T_clad_ext_guess, Tc0]
y_ss = fsolve(residuals_ss, y_guess)

T1_ss, T2_ss, T3_ss, Tcl_ss, Tc_ss = y_ss

print("\nStato stazionario (modello a 3 nodi nel fuel)")
print(f"T_fuel centro = {T1_ss - 273.15:.2f} °C")
print(f"T_fuel intermedio = {T2_ss - 273.15:.2f} °C")
print(f"T_fuel superficie = {T3_ss - 273.15:.2f} °C")
print(f"T_clad = {Tcl_ss - 273.15:.2f} °C")
print(f"T_coolant = {Tc_ss - 273.15:.2f} °C")
print(f"T_inlet = {T_inlet - 273.15:.2f} °C")


#DEFINIZIONE DELL'EQUAZIONE DIFFERENZIALE PER ANDAMENTO TEMPERATURA FUEL E COOLANT

def fuel_thermal_dynamics(t, y):

    T1, T2, T3, Tcl, Tc = y

    P = P0 / (num_rod_ass * num_ass)
    if t > 10:
        P *= 1.1

    q1 = P * V1 / (V1 + V2 + V3)
    q2 = P * V2 / (V1 + V2 + V3)
    q3 = P * V3 / (V1 + V2 + V3)

    R12 = 1 / (4 * np.pi * H_core * k_f(T1))
    R23 = R_cond(r1, r2, k_f(T2))
    R3s = R_cond(r2, r_f, k_f(T3))
    R_gap = R_gap_fun()
    R_cld = R_clad_fun()
    R_cnv = R_conv_fun()

    Q12 = (T1 - T2) / R12
    Q23 = (T2 - T3) / R23
    Q3s = (T3 - Tcl) / (R3s + R_gap)
    Qcl = (Tcl - Tc) / (R_cld + R_cnv)

    dT1 = (q1 - Q12) / (M1 * cp_f(T1))
    dT2 = (q2 + Q12 - Q23) / (M2 * cp_f(T2))
    dT3 = (q3 + Q23 - Q3s) / (M3 * cp_f(T3))
    dTcl = (Q3s - Qcl) / (M_clad * cp_clad)
    dTc  = (Qcl - 2 * m_dot_channel * cp_c * (Tc - T_inlet)) / (Mc * cp_c)

    return [dT1, dT2, dT3, dTcl, dTc]


y0  = list(y_ss)
sol = solve_ivp(fuel_thermal_dynamics, [0, 100], y0, method='LSODA')

#PLOT
#volendo si potrebbe plottare anche il cladding

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0]-273.15, color='red', label="Centro Fuel")
plt.axhline(y=T1_ss-273.15, color='black', linestyle='--', alpha=0.5, label='Stazionario ($T_{f0}$)')
plt.title("Risposta Termica del Centro del Combustibile a un incremento di Potenza del 10%")
plt.xlabel("Tempo (s)")
plt.ylabel("Temperatura (°C)")
plt.grid(True, linestyle=':')
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[2]-273.15, color='orange', linewidth=2, label='Superficie Fuel ($T_f$)')
plt.axhline(y=T3_ss-273.15, color='black', linestyle='--', alpha=0.5, label='Stazionario ($T_{f0}$)')
plt.title("Risposta Termica della Superficie del Combustibile a un incremento di Potenza del 10%")
plt.xlabel("Tempo (s)")
plt.ylabel("Temperatura (°C)")
plt.grid(True, linestyle=':')
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[4]-273.15, color='blue', label="Coolant")
plt.axhline(y=Tc_ss-273.15, color='black', linestyle='--', alpha=0.5, label='Stazionario ($T_{c0}$)')
plt.title("Risposta Termica del Coolant a un incremento di Potenza del 10%")
plt.xlabel("Tempo (s)")
plt.ylabel("Temperatura (°C)")
plt.grid(True, linestyle=':')
plt.legend()

plt.show()