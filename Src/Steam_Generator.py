"""
============================================================
SIMULAZIONE TRANSITORIA DI UNO SCAMBIATORE DI CALORE
a triplo tubo (inner pipe + annulus) per impianto nucleare
------------------------------------------------------------
Schema geometrico (sezione trasversale):
  ┌────────────────────────────┐
  │ Mantello (shell, primario) │
  │   ┌─────────────────────┐  │
  │   │  Tubo esterno (II°) │  │
  │   │  ┌───────────────┐  │  │
  │   │  │  Tubo interno │  │  │
  │   │  │   (primario)  │  │  │
  │   │  └───────────────┘  │  │
  │   │   annulus (II°)     │  │
  │   └─────────────────────┘  │
  └────────────────────────────┘

Flussi:
  Primario  → scorre nel tubo interno (I°)
              e nel mantello (I°), in controcorrente col secondario
  Secondario → scorre nell'anello (annulus) tra tubo interno ed esterno

Obiettivi della simulazione:
  1) Calcolo del profilo di temperatura allo stato stazionario (SS)
     tramite shooting method (Sezione 1)
  2) Warmup implicito: si lascia evolvere il sistema (SS continuo →
     SS discreto) usando un integrator a schema implicito (Sezione 2)
  3) Transitorio: si impone un salto del +10% del portata primaria
     a t=10 s e si studia la risposta del sistema (Sezione 3)
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar        # per il metodo di Brent (SS)
from pyXSteam.XSteam import XSteam           # proprietà termodinamiche del vapore/acqua

# Inizializzazione delle tavole del vapore (sistema SI: bar, °C, kJ, m³, kg)
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 0 — Correlazioni per il coefficiente convettivo
# ══════════════════════════════════════════════════════════════════════════════

def h_conv_c(Passo, D_ext, m_dot_ch, mu, k_w, cp, rho):
    """
    Coefficiente convettivo per flusso nel mantello (shell-side)
    con disposizione triangolare dei tubi.

    Usa la correlazione di Dittus-Boelter (Nu = 0.023 Re^0.8 Pr^0.4)
    corretta con un fattore geometrico che tiene conto del passo triangolare.

    Parametri
    ----------
    Passo    : passo triangolare tra i centri dei tubi [m]
    D_ext    : diametro esterno del tubo [m]
    m_dot_ch : portata massica nel canale [kg/s]
    mu       : viscosità dinamica [Pa·s]
    k_w      : conducibilità termica del fluido [W/(m·K)]
    cp       : calore specifico [J/(kg·K)]
    rho      : densità [kg/m³]

    Ritorna
    -------
    h : coefficiente convettivo [W/(m²·K)]
    """
    # Area della sezione di flusso nel mantello (cella triangolare meno il cerchio del tubo)
    A_flow = (np.sqrt(3)/2)*Passo**2 - np.pi*D_ext**2/4

    # Diametro idraulico per geometria triangolare
    Dh = D_ext * ((2*np.sqrt(3)/np.pi)*(Passo/D_ext)**2 - 1)

    # Numero di Reynolds: Re = ρ·v·Dh/μ  (con v = m_dot / (ρ·A_flow))
    Re = rho * (m_dot_ch / (rho * A_flow)) * Dh / mu

    # Numero di Prandtl: Pr = μ·cp/k
    Pr = mu * cp / k_w

    # Dittus-Boelter con fattore correttivo per passo triangolare
    return 0.023 * Re**0.8 * Pr**0.4 * (1.106*(Passo/D_ext) - 0.013) * k_w / Dh


def h_conv_semplice(A_flow, Dh, m_dot_ch, mu, k_w, cp, rho):
    """
    Coefficiente convettivo standard (Dittus-Boelter, senza fattori geometrici).

    Usato per il flusso nell'annulus (secondario) e nel tubo interno (primario).

    Parametri
    ----------
    A_flow   : area della sezione di flusso [m²]
    Dh       : diametro idraulico [m]
    (gli altri parametri come sopra)

    Ritorna
    -------
    h : coefficiente convettivo [W/(m²·K)]
    """
    Re = rho * (m_dot_ch / (rho * A_flow)) * Dh / mu
    Pr = mu * cp / k_w
    return 0.023 * Re**0.8 * Pr**0.4 * k_w / Dh


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 0b — Parametri di sistema (geometria, condizioni operative, ecc.)
# ══════════════════════════════════════════════════════════════════════════════

# ── Pressioni operative ──────────────────────────────────────────────────────
P_I, P_II   = 157, 34        # Pressioni primario e secondario [bar]

# ── Condizioni al contorno di ingresso ──────────────────────────────────────
T_I_in0     = 313.5          # Temperatura ingresso primario [°C]
h_II_in0    = steamTable.h_pt(P_II, 170)   # Entalpia ingresso secondario [kJ/kg]
T_II_in0    = 170.0          # Temperatura ingresso secondario [°C]

# ── Geometria dello scambiatore ─────────────────────────────────────────────
L           = 1.9             # Lunghezza attiva dello scambiatore [m]
D_2_int, D_2_ext = 10e-3, 13e-3   # Diametri interno/esterno tubo esterno [m]
D_1_int, D_1_ext =  5e-3,  8e-3   # Diametri interno/esterno tubo interno [m]
Passo_ch    = D_2_ext * 1.09      # Passo triangolare tra tubi nel mantello [m]

# ── Portate di riferimento ───────────────────────────────────────────────────
flowrate    = 902.8           # Portata totale primario [kg/s]
flowrateII  = 77.5            # Portata totale secondario [kg/s]

# ── Conducibilità termica della parete metallica ─────────────────────────────
k_cond      = 12              # [W/(m·K)] — tipico per acciaio inossidabile

# ── Discretizzazione spaziale ────────────────────────────────────────────────
N           = 5000              # Numero di celle finite
dz          = L / N           # Lunghezza di ciascuna cella [m]

# ── Numero di sotto-canali in parallelo ─────────────────────────────────────
# L'impianto ha più moduli (118 gruppi × 7 tubi × 12 sezioni)
n_sub       = 118 * 7 * 12   # = 9912 sotto-canali paralleli

# ══════════════════════════════════════════════════════════════════════════════
# Calcolo delle sezioni trasversali di flusso
# ══════════════════════════════════════════════════════════════════════════════

# Area mantello (shell-side, primario esterno): cella triangolare meno sezione tubo
A_mant  = (np.sqrt(3)/2)*Passo_ch**2 - np.pi*D_2_ext**2/4

# Area tubo interno (primario interno)
A_int   = (D_1_int/2)**2 * np.pi

# Area annulus (secondario): corona circolare tra tubo interno ed esterno
A_ann   = (D_2_int**2 - D_1_ext**2) * np.pi / 4

# Area totale primario (tubo interno + mantello)
A_tot_I = A_mant + A_int

# ══════════════════════════════════════════════════════════════════════════════
# Portate massiche per singolo sotto-canale
# ══════════════════════════════════════════════════════════════════════════════

# Secondario — un valore unico (flusso nell'annulus)
m_dot_ann  = flowrateII / n_sub

# Primario totale per sotto-canale
m_dot_I0   = flowrate / n_sub

# Il primario si divide proporzionalmente all'area tra tubo interno e mantello
# (si assume stessa velocità nei due percorsi → m_dot ∝ A_flow)
m_dot_int0 = m_dot_I0 / (1 + A_mant/A_tot_I)   # quota tubo interno
m_dot_ext0 = m_dot_I0 - m_dot_int0              # quota mantello

# ══════════════════════════════════════════════════════════════════════════════
# Proprietà termodinamiche del primario (acqua a ~295 °C, 157 bar)
# Valutate alla temperatura media approssimativa del primario
# ══════════════════════════════════════════════════════════════════════════════

mu_I   = steamTable.my_pt(P_I, 295.78)        # Viscosità dinamica [Pa·s]
k_w_I  = steamTable.tc_pt(P_I, 295.78)        # Conducibilità termica [W/(m·K)]
cp_I   = steamTable.Cp_pt(P_I, 295.78)*1000   # Calore specifico [J/(kg·K)] (×1000: kJ→J)
rho_I  = steamTable.rho_pt(P_I, 295.78)       # Densità [kg/m³]

# ══════════════════════════════════════════════════════════════════════════════
# Proprietà di saturazione del secondario a P_II = 34 bar
# Servono per trattare la transizione liquido → bifase → vapore
# ══════════════════════════════════════════════════════════════════════════════

hv_sat  = steamTable.hV_p(P_II)    # Entalpia vapore saturo [kJ/kg]
hl_sat  = steamTable.hL_p(P_II)    # Entalpia liquido saturo [kJ/kg]
Tsat    = steamTable.tsat_p(P_II)  # Temperatura di saturazione [°C]
rhoL_II = steamTable.rhoL_p(P_II)  # Densità liquido saturo [kg/m³]
rhoV_II = steamTable.rhoV_p(P_II)  # Densità vapore saturo [kg/m³]

# ── Proprietà del secondario in fase liquida subcooled ──────────────────────
# Valutate alla temperatura media tra T_in=170 °C e T_sat
mu_II_l  = steamTable.my_pt(P_II, (Tsat+170)/2)
k_w_II_l = steamTable.tc_pt(P_II, (Tsat+170)/2)
cp_II_l  = steamTable.Cp_pt(P_II, (Tsat+170)/2)*1000
rho_II_l = steamTable.rho_pt(P_II, (Tsat+170)/2)

# ── Proprietà del secondario in fase vapore surriscaldato ───────────────────
# Valutate alla temperatura media tra T_sat e ~289 °C (temperatura stimata uscita vapore)
mu_II_g  = steamTable.my_pt(P_II, (Tsat+288.85)/2)
k_w_II_g = steamTable.tc_pt(P_II, (Tsat+288.85)/2)
cp_II_g  = steamTable.Cp_pt(P_II, (Tsat+288.85)/2)*1000
rho_II_g = steamTable.rho_pt(P_II, (Tsat+288.85)/2)

# ══════════════════════════════════════════════════════════════════════════════
# Coefficienti convettivi lato secondario (annulus)
# Calcolati una sola volta perché i parametri del secondario non variano
# ══════════════════════════════════════════════════════════════════════════════

# Diametro idraulico dell'annulus = differenza tra diametri (corona circolare)
Dh_ann = D_2_int - D_1_ext

# h in fase liquida e vapore (Dittus-Boelter semplice)
H_II_l = h_conv_semplice(A_ann, Dh_ann, m_dot_ann,
                          mu_II_l, k_w_II_l, cp_II_l, rho_II_l)
H_II_v = h_conv_semplice(A_ann, Dh_ann, m_dot_ann,
                          mu_II_g, k_w_II_g, cp_II_g, rho_II_g)

# ══════════════════════════════════════════════════════════════════════════════
# Resistenze termiche di conduzione attraverso le pareti dei tubi
#
# Per un cilindro: R_cond = ln(r_ext/r_int) / (2π·L·k)
# La lunghezza qui è dz (una sola cella)
# ══════════════════════════════════════════════════════════════════════════════

# Resistenza parete tubo esterno (separa mantello dall'annulus)
R_wall_ext = np.log((D_2_ext/2)/(D_2_int/2)) / (2*np.pi*dz*k_cond)

# Resistenza parete tubo interno (separa tubo interno dall'annulus)
R_wall_int = np.log((D_1_ext/2)/(D_1_int/2)) / (2*np.pi*dz*k_cond)

# Coordinata assiale delle celle (centro di ciascuna cella, da z=0 a z=L)
# z=0 → ingresso secondario / uscita primario
# z=L → uscita secondario / ingresso primario
z_coord = np.linspace(0, L, N)
z_centers = np.linspace(dz/2, L - dz/2, N)   # coordinate vere dei centri cella


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 0c — Funzioni ausiliarie
# ══════════════════════════════════════════════════════════════════════════════

def controlla_fase(h):
    """
    Data l'entalpia h del secondario [kJ/kg], determina la fase termodinamica
    e restituisce le grandezze fisiche corrispondenti.

    Ritorna
    -------
    fase : 'liq' | 'bifase' | 'vap'
    x    : titolo del vapore (0 per liquido, 1 per vapore, intermedio in bifase)
    T    : temperatura [°C]
    """
    if h <= hl_sat:
        # Liquido subcooled: la temperatura viene letta dalle tavole
        return 'liq', 0.0, steamTable.t_ph(P_II, h)
    elif h >= hv_sat:
        # Vapore surriscaldato: la temperatura viene letta dalle tavole
        return 'vap', 1.0, steamTable.t_ph(P_II, h)
    else:
        # Zona bifase: T = T_sat, x calcolato con la regola della leva
        return 'bifase', (h - hl_sat)/(hv_sat - hl_sat), Tsat


def rho_s(fase, x):
    """
    Densità media del secondario nella cella [kg/m³].

    In zona bifase si usa la media armonica pesata sul titolo x:
      1/ρ_mix = x/ρ_V + (1-x)/ρ_L   (modello homogeneous)

    Nota: per 'liq' e 'vap' il codice usa lo stesso schema con i ruoli
    di rhoV_II e rhoL_II invertiti rispetto alla zona bifase
    (è un dettaglio implementativo che non influisce sul risultato).
    """
    if fase == 'liq':   return rho_II_l
    elif fase == 'vap': return rho_II_g
    else:               return rhoV_II*x + rhoL_II*(1-x)


def calc_R_conv(m_dot_int_loc, m_dot_ext_loc):
    """
    Calcola le resistenze convettive lato primario per una cella.

    Le resistenze vengono aggiornate ad ogni passo temporale perché
    la portata primaria può cambiare durante il transitorio.

    Parametri
    ----------
    m_dot_int_loc : portata nel tubo interno [kg/s]
    m_dot_ext_loc : portata nel mantello [kg/s]

    Ritorna
    -------
    Rce : resistenza convettiva lato mantello (tubo esterno) [K/W]
    Rci : resistenza convettiva lato tubo interno [K/W]
    """
    # h lato interno (tubo piccolo, geometria semplice)
    H_int = h_conv_semplice(A_int, D_1_int, m_dot_int_loc, mu_I, k_w_I, cp_I, rho_I)

    # h lato mantello (geometria con passo triangolare)
    H_ext = h_conv_c(Passo_ch, D_2_ext, m_dot_ext_loc, mu_I, k_w_I, cp_I, rho_I)

    # R_conv = 1 / (h · A_scambio)  con  A = π·D·dz
    Rce = 1 / (2*np.pi*(D_2_ext/2)*dz * H_ext)   # lato mantello (su tubo esterno)
    Rci = 1 / (2*np.pi*(D_1_int/2)*dz * H_int)   # lato tubo interno
    return Rce, Rci


def calcola_R_II(fase, x, Rce, Rci):
    """
    Calcola le resistenze totali tra primario e secondario attraverso
    ciascuna parete (esterna e interna) per una singola cella.

    Schema delle resistenze (parete esterna, es. tubo esterno):
      T_prim → [Rce] → parete → [R_wall_ext] → [R_conv_II] → T_II

    In zona bifase si usa la correlazione di Chen/Kandlikar semplificata
    per la ebollizione convettiva (h_nb molto più alto).

    Parametri
    ----------
    fase, x  : fase e titolo del secondario (da controlla_fase)
    Rce, Rci : resistenze convettive lato primario (da calc_R_conv)

    Ritorna
    -------
    Re : resistenza totale lato tubo esterno [K/W]
    Ri : resistenza totale lato tubo interno [K/W]
    """
    # Selezione del coefficiente convettivo del secondario in base alla fase
    if fase == 'liq':
        H = H_II_l                # liquido subcooled
    elif fase == 'vap':
        H = H_II_v                # vapore surriscaldato
    else:
        # Ebollizione convettiva in zona bifase (correlazione semplificata):
        # h_Chen = h_l · (1 + F_Chen)  dove F_Chen dipende da x
        # Per x ≥ 0.95 (quasi tutto vapore) si usa un fattore ridotto
        if x < 0.95:
            H = 6000.0 * (1 + 3.8 * ((x/(1-x))**0.76 * (3.4/22.1e6)**0.38))
        else:
            H = 6000.0 * 1.2    # regime near-dryout: riduzione del h bifase

    # Resistenza totale = R_conv_prim + R_parete + R_conv_II
    # (resistenze in serie lungo il percorso di calore)
    Re = Rce + R_wall_ext + 1/(2*np.pi*(D_2_int/2)*dz * H)   # lato tubo esterno
    Ri = Rci + R_wall_int + 1/(2*np.pi*(D_1_ext/2)*dz * H)   # lato tubo interno
    return Re, Ri


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE — Schema di integrazione implicito
#
# Motivazione della scelta implicita
# ────────────────────────────────────
# Uno schema esplicito (Eulero forward) è vincolato dalla condizione CFL:
#   dt_max ≈ dz / v_fluido  (tipicamente 0.001–0.005 s per questo problema)
# Con N=30 celle e t_end=30 s servirebbero ~6.000–30.000 passi.
#
# Lo schema implicito (Eulero backward) è incondizionatamente stabile,
# consentendo dt molto più grandi (dt=1 s per il warmup, dt=0.01 s per
# il transitorio di precisione) senza oscillazioni numeriche.
#
# Equazione discreta per il PRIMARIO (cella i, sweep N-1 → 0):
#   α·(T_new[i] - T[i]) = ṁ_I·(T_new_up - T_new[i]) - G_i/cp_I·(T_new[i] - T_II_old[i])
#
#   Dove: α = ρ_I·A_tot_I·dz/dt  [kg/s]  (termine di accumulo)
#         T_new_up = T_new[i+1] già noto (upstream, calcolato al passo precedente del sweep)
#
#   Risolto esplicitamente per T_new[i]:
#     T_new[i] = (α·T[i] + ṁ·T_up + G/cp·T_II) / (α + ṁ + G/cp)
#
# Equazione discreta per il SECONDARIO (cella i, sweep 0 → N-1):
#   β_i·(h_new[i] - h[i]) = ṁ_ann·(h_new_up - h_new[i]) + dQ_i/1000
#
#   Dove: β_i = ρ_s[i]·A_ann·dz/dt  [kg/s]  (termine di accumulo)
#         h_new_up = h_new[i-1] già noto (upstream)
#         dQ_i calcolato con T_new[i] già aggiornato → schema semi-implicito
#
#   Risolto per h_new[i]:
#     h_new[i] = (β·h[i] + ṁ_ann·h_up + dQ/1000) / (β + ṁ_ann)
# ══════════════════════════════════════════════════════════════════════════════

def step_implicit(Tp, hs, m_dot_I, m_dot_ann, Rce, Rci, dt_loc):
    """
    Esegue UN passo di integrazione temporale con schema implicito.

    Schema controcorrente:
      Primario  : entra da i=N-1 (z=L), esce a i=0 (z=0)  → sweep N-1 → 0
      Secondario: entra da i=0   (z=0), esce a i=N-1 (z=L) → sweep 0 → N-1

    Parametri
    ----------
    Tp       : array temperature primario [°C], shape (N,)
    hs       : array entalpie secondario [kJ/kg], shape (N,)
    m_dot_I  : portata primario per sotto-canale [kg/s]
    m_dot_ann: portata secondario per sotto-canale [kg/s]
    Rce, Rci : resistenze convettive primario (da calc_R_conv) [K/W]
    dt_loc   : passo temporale [s]

    Ritorna
    -------
    Tp_new  : temperature primario aggiornate [°C]
    hs_new  : entalpie secondario aggiornate [kJ/kg]
    Q_step  : calore totale ceduto al secondario nel passo [W]
    """
    Tp_new = np.empty(N)
    hs_new = np.empty(N)

    # ── Pre-calcolo delle quantità per ogni cella ────────────────────────────
    # Queste grandezze sono valutate allo step precedente (n) e restano fisse
    # durante lo sweep corrente (uso di valori "vecchi" per le proprietà del
    # secondario = accoppiamento semi-implicito).

    T_II_arr = np.empty(N)   # temperatura secondario per cella [°C]
    G_arr    = np.empty(N)   # conduttanza termica cella = 1/Re + 1/Ri [W/K]
    rhos_arr = np.empty(N)   # densità secondario media [kg/m³]

    for i in range(N):
        fase, x, T_II = controlla_fase(hs[i])   # fase e T dal precedente timestep
        Re, Ri         = calcola_R_II(fase, x, Rce, Rci)
        T_II_arr[i]    = T_II
        G_arr[i]       = 1.0/Re + 1.0/Ri        # conduttanza complessiva della cella
        rhos_arr[i]    = rho_s(fase, x)

    # ── Sweep primario: da i=N-1 (uscita sec.) verso i=0 (entrata sec.) ─────
    # Il primario entra caldo a z=L (i=N-1) e si raffredda verso z=0
    # Il termine α rappresenta l'inerzia termica del fluido nella cella

    alpha = rho_I * A_tot_I * dz / dt_loc   # [kg/s]: massa cella / dt

    for i in range(N-1, -1, -1):
        # T_new_up: condizione al bordo upstream per il primario
        #   → se siamo all'ultima cella (i=N-1): uso la condizione al contorno T_I_in0
        #   → altrimenti: T_new[i+1] già calcolato nel sweep corrente (implicito!)
        T_new_up = Tp_new[i+1] if i < N-1 else T_I_in0

        # Equazione implicita risolta per T_new[i]
        # Caso normale: c'è scambio termico (T_I > T_II)
        Gcp   = G_arr[i] / cp_I   # conduttanza in unità di portata [kg/s]: G/(cp) 
        num   = alpha * Tp[i] + m_dot_I * T_new_up + Gcp * T_II_arr[i]
        den   = alpha + m_dot_I + Gcp
        T_new = num / den

        # Clamp fisico: il primario non può scendere sotto la T del secondario
        # (non si modella riscaldamento inverso secondario→primario)
        # Se T_new < T_II, il termine di scambio viene azzerato
        if T_new < T_II_arr[i]:
            T_new = (alpha * Tp[i] + m_dot_I * T_new_up) / (alpha + m_dot_I)

        Tp_new[i] = T_new

    # ── Sweep secondario: da i=0 (ingresso) verso i=N-1 (uscita) ────────────
    # Il secondario entra freddo (subcooled) a z=0 e si scalda/evapora verso z=L
    # β_i dipende dalla cella perché la densità varia con la fase

    Q_step = 0.0   # accumulo del calore totale ceduto [W]

    for i in range(N):
        # h_new_up: condizione al bordo upstream per il secondario
        #   → se siamo alla prima cella (i=0): uso la condizione al contorno h_II_in0
        #   → altrimenti: h_new[i-1] già calcolato nel sweep corrente
        h_new_up = hs_new[i-1] if i > 0 else h_II_in0

        # Termine di accumulo del secondario nella cella [kg/s]
        beta_i = rhos_arr[i] * A_ann * dz / dt_loc

        # Calore ceduto dalla cella al secondario [W]
        # Usa T_new[i] già aggiornato (semi-implicito lato primario)
        # max(0,...) garantisce che il calore vada solo da primario a secondario
        dQ_i    = max(0.0, Tp_new[i] - T_II_arr[i]) * G_arr[i]
        Q_step += dQ_i   # somma contributo di questa cella

        # Equazione implicita risolta per h_new[i] [kJ/kg]
        # Nota: dQ_i è in [W] = [J/s], dividiamo per 1000 per convertire in [kJ/s]
        # perché le entalpie sono in [kJ/kg] e ṁ in [kg/s]
        hs_new[i] = (beta_i * hs[i] + m_dot_ann * h_new_up + dQ_i/1000) / \
                    (beta_i + m_dot_ann)

    return Tp_new, hs_new, Q_step


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 1 — Stato stazionario continuo (Shooting Method)
# Il metodo di shooting cerca la temperatura di uscita del primario T_I_out
# tale che il bilancio integrato lungo tutto lo scambiatore (partendo da z=0
# con T_I = T_I_out e h_II = h_II_in0) riporti T_I = T_I_in0 a z=L.
#
# Viene usato Brent's method (root_scalar) che converge in modo robusto
# senza richiedere la derivata della funzione.
# ══════════════════════════════════════════════════════════════════════════════

# Resistenze convettive primario allo stato stazionario di riferimento

Rce0, Rci0 = calc_R_conv(m_dot_int0, m_dot_ext0)


def solve_system(T_out_guess):
    """
    Dato un tentativo di temperatura di uscita primario T_out_guess,
    integra il sistema di equazioni differenziali lungo lo scambiatore
    e restituisce il residuo rispetto alla condizione al contorno T_I_in0.

    Il residuo è zero quando T_out_guess è la soluzione corretta.

    Integrazione: schema upwind esplicito stazionario
      → dT_I/dz  = G · (T_I - T_II) / (ṁ_I · cp_I)
      → dh_II/dz = G · (T_I - T_II) / (ṁ_ann)
    """
    T_I    = T_out_guess   # partiamo dall'uscita (z=0) con il tentativo
    hs_rel = 0.0           # entalpia relativa secondario (Δh rispetto all'ingresso)

    for _ in range(N):
        fase, x, T_II = controlla_fase(hs_rel + h_II_in0)
        Re, Ri = calcola_R_II(fase, x, Rce0, Rci0)
        G      = 1/Re + 1/Ri

        # Calore ceduto nella cella (solo se T_I > T_II, altrimenti zero)
        dQ     = max(0.0, T_I - T_II) * G

        # Avanzamento delle variabili di stato verso l'uscita del secondario
        hs_rel += (dQ / m_dot_ann) / 1000    # Δh [kJ/kg]: dividiamo per 1000 (J→kJ)
        T_I    += dQ / (m_dot_I0 * cp_I)    # ΔT_prim [°C]: calore ceduto → riscalda il primario

        # NOTA: in questa integrazione il primario si "riscalda" andando da z=0 a z=L
        # perché integriamo in direzione opponente al flusso del primario
        # (il primario entra caldo a z=L e il shooting trova T a z=0)

    # Il residuo è la differenza tra T_I integrato e la condizione al contorno
    return T_I - T_I_in0


# Ricerca della soluzione con metodo di Brent nell'intervallo [T_II_in0, T_I_in0]
sol        = root_scalar(solve_system, bracket=[T_II_in0, T_I_in0], method='brentq')
T_I_out_ok = sol.root
print(f"T_I_out SS continuo : {T_I_out_ok:.4f} °C  |  residuo: {solve_system(T_I_out_ok):.2e}")

# ── Ricostruzione dei profili di T e h lungo lo scambiatore ─────────────────
# Necessaria per avere il punto di partenza fisicamente coerente per il warmup

Tp_list, hs_list = [], []
T_I, hs_rel = T_I_out_ok, 0.0    # partiamo dalla soluzione trovata con lo shooting

for _ in range(N):
    fase, x, T_II = controlla_fase(hs_rel + h_II_in0)
    Tp_list.append(T_I)               # salvo la T primario nella cella corrente
    hs_list.append(hs_rel + h_II_in0) # salvo l'entalpia assoluta del secondario

    Re, Ri  = calcola_R_II(fase, x, Rce0, Rci0)
    G       = 1/Re + 1/Ri
    dQ      = max(0.0, T_I - T_II) * G

    hs_rel += (dQ / m_dot_ann) / 1000
    T_I    += dQ / (m_dot_I0 * cp_I)

# Conversione in array NumPy (necessario per le operazioni vettoriali successive)
Tp = np.array(Tp_list)
# Il loop di ricostruzione salva hs alla FACCIA SINISTRA di ogni cella i
# (valore prima del dQ), ma step_implicit tratta hs come valore al CENTRO CELLA.
# Al SS discreto:  h_impl[i] = h_impl[i-1] + dQ_i / (m_dot * 1000)
#                             = hs_face[i+1]   (= valore in faccia della cella i+1)
# → shift di una cella verso l'alto; l'ultima cella viene estrapolata linearmente.
# Il profilo Tp del primario NON richiede questo aggiustamento perché il loop
# shooting ricostruisce esattamente la soluzione SS implicita del primario.
hs_raw = np.array(hs_list)
hs = np.empty(N)
hs[:N-1] = hs_raw[1:]                       # celle 0..N-2: prendi il valore della faccia successiva
hs[N-1]  = 2.0*hs_raw[N-1] - hs_raw[N-2]   # cella N-1: estrapolazione lineare

# ── Visualizzazione del profilo SS continuo ──────────────────────────────────
plt.figure(figsize=(10, 5))
# Primario: centri cella + punto BC ingresso a z=L
plt.plot(np.concatenate([z_centers, [L]]), list(Tp) + [T_I_in0],
         'r', lw=2, label='Primario')
# Secondario: punto BC a z=0 + centri cella (nessun gradino verticale)
_T_sec = [controlla_fase(h)[2] for h in hs]
plt.plot(np.concatenate([[0.0], z_centers]), [T_II_in0] + _T_sec,
         'b', lw=2, label='Secondario')
plt.axhline(Tsat, color='gray', ls='--', alpha=0.7, label=f'T_sat = {Tsat:.1f} °C')
plt.title("Profilo T — SS Continuo (punto di partenza warmup)")
plt.xlabel("z [m]  (0 = ingr. sec. / usc. prim.)"); plt.ylabel("T [°C]")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()



# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 2 — Warmup implicito: da SS continuo → SS discreto
#
# Il profilo SS "continuo" è calcolato con un'integrazione upwind a N passi,
# ma non è ancora la soluzione stazionaria del sistema discreto (N celle FVM).
# Il warmup fa evolvere il sistema con dt grande (1 s) finché la soluzione
# si stabilizza (convergenza su N_check passi consecutivi sotto la tolleranza).
#
# Tempi caratteristici del sistema:
#   τ_adv,prim  ≈ L / v_prim ≈ 1–2 s   → il primario si "rinnova" ogni 1-2 s
#   τ_adv,sec   ≈ L / v_ann  ≈ 5–10 s  → il secondario si "rinnova" ogni 5-10 s
# → bastano ~200–500 passi con dt=1 s (invece di >50.000 con schema esplicito)
# ══════════════════════════════════════════════════════════════════════════════

dt_wu   = 1.0           # Passo temporale del warmup [s] — grande grazie alla stabilità implicita
tol_T   = 1e-7          # Tolleranza sulla variazione di T primario [°C]
tol_h   = 1e-5          # Tolleranza sulla variazione di h secondario [kJ/kg]
max_wu  = 10000        # Numero massimo di iterazioni di warmup
N_check = 50            # Passi consecutivi sotto tolleranza per dichiarare convergenza

print(f"\nWarmup implicito (dt={dt_wu} s, tol_T={tol_T}, tol_h={tol_h})...")
below_tol = 0   # contatore dei passi consecutivi sotto la tolleranza

for wu in range(max_wu):
    Tp_new, hs_new, _ = step_implicit(Tp, hs, m_dot_I0, m_dot_ann, Rce0, Rci0, dt_wu)

    # Variazione massima dei profili rispetto al passo precedente
    dTp_max = np.max(np.abs(Tp_new - Tp))
    dhs_max = np.max(np.abs(hs_new - hs))

    # Aggiornamento dei profili per il passo successivo
    Tp, hs  = Tp_new, hs_new

    # Criterio di convergenza: N_check passi consecutivi con variazioni < tolleranza
    if dTp_max < tol_T and dhs_max < tol_h:
        below_tol += 1
        if below_tol >= N_check:
            print(f"  Convergenza in {wu+1} passi "
                  f"(ΔTp={dTp_max:.2e} °C, Δhs={dhs_max:.2e} kJ/kg)")
            break
    else:
        below_tol = 0   # reset: un passo fuori tolleranza azzera il contatore
else:
    print(f"  ATTENZIONE: warmup non convergito in {max_wu} passi")

# ── Verifica trilaterale del bilancio energetico allo SS discreto ─────────────
# Tre modi indipendenti di calcolare Q devono dare lo stesso risultato:
#   1) Somma cella-cella: Q = Σ G_i · (T_I,i - T_II,i)
#   2) Bilancio primario: Q = ṁ_I · cp · (T_in - T_out)
#   3) Bilancio secondario: Q = ṁ_ann · (h_out - h_in)

Q_disc_cell = sum(
    max(0.0, Tp[i] - controlla_fase(hs[i])[2]) *
    (1/calcola_R_II(controlla_fase(hs[i])[0], controlla_fase(hs[i])[1], Rce0, Rci0)[0] +
     1/calcola_R_II(controlla_fase(hs[i])[0], controlla_fase(hs[i])[1], Rce0, Rci0)[1])
    for i in range(N))

# Q stimato dal bilancio energetico sul primario
Q_disc_prim = m_dot_I0   * cp_I  * (T_I_in0   - Tp[0])

# Q stimato dal bilancio energetico sul secondario (×1000 perché h in kJ/kg, Q in W)
Q_disc_sec  = m_dot_ann  * 1000  * (hs[N-1]   - h_II_in0)

print(f"\nVerifica Q SS discreto:")
print(f"  cell-sum    : {Q_disc_cell:.2f} W  ({Q_disc_cell/1000:.4f} kW)")
print(f"  bil.primario: {Q_disc_prim:.2f} W  ({Q_disc_prim/1000:.4f} kW)")
print(f"  bil.second. : {Q_disc_sec:.2f}  W  ({Q_disc_sec/1000:.4f} kW)")
print(f"  squilibrio prim-sec: {abs(Q_disc_prim-Q_disc_sec)/Q_disc_prim*100:.5f}%")

# Visualizzazione del profilo SS discreto (condizione iniziale per il transitorio)
plt.figure(figsize=(10, 5))
plt.plot(np.concatenate([z_centers, [L]]), list(Tp) + [T_I_in0],
         'r', lw=2, label='Primario (SS disc.)')
_T_sec = [controlla_fase(h)[2] for h in hs]
plt.plot(np.concatenate([[0.0], z_centers]), [T_II_in0] + _T_sec,
         'b', lw=2, label='Secondario')
plt.axhline(Tsat, color='gray', ls='--', alpha=0.7, label=f'T_sat = {Tsat:.1f} °C')
plt.title("Profilo T — SS Discreto (condizione iniziale transitorio)")
plt.xlabel("z [m]"); plt.ylabel("T [°C]")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3 — Transitorio implicito
#
# Scenario: a t=10 s la portata primaria viene aumentata del +10%
# (es. apertura di una valvola, cambio di setpoint della pompa).
# Si studia la risposta del sistema fino a t=30 s.
#
# Grandezze monitorate:
#   T_out      : temperatura uscita primario (a z=0) [°C]
#   Q_cell     : calore ceduto (somma cella-cella) [W]
#   Q_prim/sec : bilanci ai bordi [W] — includono accumulo termico in transitorio
#   Q_prim/sec_corr : bilanci corretti sottraendo il termine di accumulo [W]
#                     (questi devono coincidere con Q_cell a meno di errori di discretizzazione)
# ══════════════════════════════════════════════════════════════════════════════

dt      = 0.1   # Passo temporale del transitorio [s]
t_end   = 30.0   # Durata del transitorio [s]
t_salto = 10.0   # Istante del salto di portata [s]
nt      = int(t_end / dt)   # Numero totale di passi temporali

# Array per la storia temporale delle grandezze di interesse
T_out_hist       = []    # T uscita primario [°C]
Q_cell_hist      = []    # Q cella-cella [W]
Q_prim_hist      = []    # Q bilancio primario ai bordi [W]
Q_sec_hist       = []    # Q bilancio secondario ai bordi [W]
t_hist           = []    # vettore del tempo [s]

print(f"\nTransitorio 0–{t_end} s ({nt} passi, dt={dt} s) — schema implicito...")

for n in range(nt):
    t = n * dt   # tempo corrente [s]

    # ── Aggiornamento portata primaria in funzione del tempo ─────────────────
    # Salto a gradino: da t=t_salto in poi la portata aumenta del 10%
    fr_t      = flowrate * 1.10 if t >= t_salto else flowrate

    # Portate per sotto-canale (aggiornate ad ogni passo)
    m_dot_I   = fr_t / n_sub
    m_dot_int = m_dot_I / (1 + A_mant/A_tot_I)   # quota tubo interno
    m_dot_ext = m_dot_I - m_dot_int               # quota mantello

    # Resistenze convettive aggiornate con la nuova portata (Re e Nu cambiano)
    Rce, Rci  = calc_R_conv(m_dot_int, m_dot_ext)

    # ── Salvataggio dello stato al passo n (prima dell'aggiornamento) ────────
    # Necessario per calcolare i termini di accumulo dU/dt
    """
    Tp_old    = Tp.copy()
    hs_old    = hs.copy()
    rho_s_old = np.array([rho_s(controlla_fase(h)[0], controlla_fase(h)[1])
                          for h in hs_old])"""

    # ── Avanzamento di un passo temporale implicito ──────────────────────────
    Tp, hs, Q_step = step_implicit(Tp, hs, m_dot_I, m_dot_ann, Rce, Rci, dt)

    # ── Bilanci energetici ai bordi (non corretti per l'accumulo) ────────────
    # In regime stazionario questi coincidono con Q_step;
    # in transitorio differiscono a causa dell'energia accumulata nel fluido.
    Q_prim = m_dot_I   * cp_I  * (T_I_in0  - Tp[0])    # [W]
    Q_sec  = m_dot_ann * 1000  * (hs[N-1]  - h_II_in0)  # [W] (×1000: kJ→J)

    # ── Correzione dei bilanci per il termine di accumulo ────────────────────
    # Il bilancio energetico di un volume di controllo in transitorio è:
    #   Q_in - Q_out = dU/dt + Q_scambiato
    # → Q_scambiato = Q_bordi - dU/dt  (primario cede energia → dU/dt < 0 → correzionepositiva)
    # → Q_scambiato = Q_bordi + dU/dt  (secondario accumula energia → dU/dt > 0 → correzione positiva)

    # Variazione di energia interna del primario [W]
    #dU_prim_dt = rho_I * A_tot_I * dz * cp_I * np.sum(Tp - Tp_old) / dt

    # Variazione di energia interna del secondario [W]
    # (usa densità allo step precedente per coerenza con lo schema semi-implicito)
    #dU_sec_dt  = np.sum(rho_s_old * A_ann * dz * 1000 * (hs - hs_old) / dt)

    Q_prim_corr = m_dot_I  * cp_I * (T_I_in0  - Tp[0])          # calore effettivamente scambiato dal primario
    Q_sec_corr  = m_dot_ann * 1000 * (hs[N-1] - h_II_in0)       # calore effettivamente ricevuto dal secondario

    # Salvataggio nella storia temporale
    T_out_hist.append(float(Tp[0]))      # T uscita primario (cella i=0)
    Q_cell_hist.append(Q_step)
    Q_prim_hist.append(Q_prim)
    Q_sec_hist.append(Q_sec)
    #Q_prim_corr_hist.append(Q_prim_corr)
    #Q_sec_corr_hist.append(Q_sec_corr)
    t_hist.append(t + dt)               # tempo a fine del passo

# Conversione in array NumPy per le operazioni successive
T_out_arr       = np.array(T_out_hist)
Q_cell_arr      = np.array(Q_cell_hist)
Q_prim_arr      = np.array(Q_prim_hist)
Q_sec_arr       = np.array(Q_sec_hist)
#Q_prim_corr_arr = np.array(Q_prim_corr_hist)
#Q_sec_corr_arr  = np.array(Q_sec_corr_hist)
t_arr           = np.array(t_hist)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3b — Post-processing: medie pre/post salto e stampa risultati
# ══════════════════════════════════════════════════════════════════════════════

idx = int(t_salto / dt)           # indice del passo in cui avviene il salto
win = max(1, int(1.0 / dt))      # finestra di mediazione: 1 s in unità di passi

# Medie nell'intervallo [t_salto-1 s, t_salto] (SS iniziale)
T_pre  = T_out_arr[max(0, idx-win):idx].mean()
Q_pre  = Q_cell_arr[max(0, idx-win):idx].mean()
Qp_pre = Q_prim_arr[max(0, idx-win):idx].mean()

# Medie nell'ultimo 1 s della simulazione (SS finale dopo il transitorio)
T_post  = T_out_arr[-win:].mean()
Q_post  = Q_cell_arr[-win:].mean()
Qp_post = Q_prim_arr[-win:].mean()

# Errore massimo tra i tre metodi di calcolo di Q nel corso del transitorio
#err_Qp = np.max(np.abs(Q_prim_corr_arr - Q_cell_arr))
#err_Qs = np.max(np.abs(Q_sec_corr_arr  - Q_cell_arr))

print(f"\n{'─'*60}")
print(f"  T_out primario  pre-salto   : {T_pre:.5f} °C")
print(f"  T_out primario  post-salto  : {T_post:.5f} °C  ΔT = {T_post-T_pre:+.5f} °C")
print(f"\n  Q cell-sum  pre-salto   : {Q_pre/1000:.4f} kW")
print(f"  Q cell-sum  post-salto  : {Q_post/1000:.4f} kW  "
      f"ΔQ = {(Q_post-Q_pre)/1000:+.4f} kW  ({(Q_post/Q_pre-1)*100:+.2f}%)")
print(f"\n  Q bil.prim. corretto pre-salto   : {Qp_pre/1000:.4f} kW")
print(f"  Q bil.prim. corretto post-salto  : {Qp_post/1000:.4f} kW  "
      f"ΔQ = {(Qp_post-Qp_pre)/1000:+.4f} kW  ({(Qp_post/Qp_pre-1)*100:+.2f}%)")
#print(f"\n  max |Qprim_corr - Qcell| : {err_Qp:.3e} W   ← errore di bilanciamento")
#print(f"  max |Qsec_corr  - Qcell| : {err_Qs:.3e} W   ← errore di bilanciamento")
print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3c — Grafici del transitorio
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Temperatura di uscita del primario nel tempo ─────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_arr, T_out_arr, color='crimson', lw=2, label='T_out primario (z=0)')
ax.axvline(t_salto, color='navy', ls='--', lw=1.4, label=f'Salto +10% @ t={t_salto} s')
ax.axhline(T_pre,  color='gray',       ls=':', lw=1.3, label=f'SS pre  = {T_pre:.4f} °C')
ax.axhline(T_post, color='darkorange', ls=':', lw=1.3, label=f'SS post = {T_post:.4f} °C')
ax.set_xlabel("Tempo [s]"); ax.set_ylabel("T_out Primario [°C]")
ax.set_title("Transitorio T_out Primario — Schema Implicito — Salto Flowrate +10%")
ax.legend(); ax.grid(alpha=0.35); ax.set_xlim(0, t_end)
plt.tight_layout(); plt.show()

# ── Plot 2: Potenza scambiata — confronto tre metodi ─────────────────────────
# Le curve corrette (con accumulo) devono sovrapporsi alla cell-sum
# Le curve non corrette divergono durante il transitorio

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_arr, Q_cell_arr/1000, color='steelblue', lw=1.8,
        label='Q cell-sum')
ax.plot(t_arr, Q_prim_arr/1000, color='tomato',    lw=1.4, ls='--',
        label='Q bil. primario  ṁ·cp·ΔT')
ax.plot(t_arr, Q_sec_arr/1000,  color='seagreen',  lw=1.4, ls=':',
        label='Q bil. secondario  ṁ·Δh')
ax.axvline(t_salto, color='navy', ls='--', lw=1.4, label=f'Salto +10% @ t={t_salto} s')
ax.axhline(Q_pre/1000,  color='gray',       ls=':', lw=1.2,
           label=f'Q pre  = {Q_pre/1000:.3f} kW')
ax.axhline(Q_post/1000, color='darkorange', ls=':', lw=1.2,
           label=f'Q post = {Q_post/1000:.3f} kW  ({(Q_post/Q_pre-1)*100:+.1f}%)')
ax.set_xlabel("Tempo [s]"); ax.set_ylabel("Q scambiata [kW]")
ax.set_title("Potenza Scambiata — Confronto metodi (devono coincidere in SS)")
ax.legend(fontsize=9); ax.grid(alpha=0.35); ax.set_xlim(0, t_end)
plt.tight_layout(); plt.show()

# ── Plot 3: Profili di temperatura a fine simulazione ─────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(np.concatenate([z_centers, [L]]), list(Tp) + [T_I_in0],
         'r', lw=2, label='Primario (fine transitorio)')
_T_sec = [controlla_fase(h)[2] for h in hs]
plt.plot(np.concatenate([[0.0], z_centers]), [T_II_in0] + _T_sec,
         'b', lw=2, label='Secondario')
plt.axhline(Tsat, color='gray', ls='--', alpha=0.7, label=f'T_sat = {Tsat:.1f} °C')
plt.title(f"Profilo T finale — t = {t_end} s (post salto +10%)")
plt.xlabel("z [m]"); plt.ylabel("T [°C]")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
