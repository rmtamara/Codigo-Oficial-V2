"""
En este script se resuelve la temperatura de salida del fluido Therminol
VP - 1 asi como tambien del agua. El primero se desarrolla en dos
problemas: output en el campo solar y en el tren de intercambiadores,
donde simultaneamente se calcula la temperatura de salida del agua.
"""
import pickle
import numpy as np
import CoolProp.CoolProp as CP
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
import math
from scipy.signal import savgol_filter
from tqdm import tqdm

# ------------------------------------------------------------------------
#                            Base de Datos
# ------------------------------------------------------------------------

file = 'base.xlsx'
base_mes = pd.read_excel(file, sheet_name='Hoja7')
rad = base_mes['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
temp = base_mes['Temperatura'] + 273.15
dia = base_mes['Dia']
mes = base_mes['Mes']
hora = base_mes['hora']
minuto = base_mes['minuto']
hora_minuto = base_mes['Hora total']
v = base_mes['Velocidad']

fit_rad = CubicSpline(hora_minuto, rad)
fit_temp = CubicSpline(hora_minuto, temp)
fit_vel = CubicSpline(hora_minuto, v)

# ------------------------------------------------------------------------
#                           Working Fluid
# ------------------------------------------------------------------------

wb = 'TherminolVP1.xlsx'
BD = pd.read_excel(wb, sheet_name='LiquidProperties')
T = BD['Temperature\n[°C]\n'] + 273.15
RHO = BD['Liquid density\n[kg/m3]']
CAP = BD['Liquid heat capacity\n[kJ/kg - K]'] * 1000
H = BD['Liquid enthalpy\n[kJ/kg]'] * 1000
K = BD['Liquid thermal conductivity\n[W/m- K]']
MU = BD['Liquid viscosity\n[mPa * s]'] * 0.001
P = BD['Vapor pressure\n[kPa]']

fit_cp = CubicSpline(T, CAP)
fit_rho = CubicSpline(T, RHO)
fit_h = CubicSpline(T, H)
fit_k = CubicSpline(T, K)
fit_mu = CubicSpline(T, MU)

# ------------------------------------------------------------------------
#                             Parametros
# ------------------------------------------------------------------------

sigma = 5.6697 * 10**(-8)
eta1 = 0.94
eta2 = 0.96
eta3 = 0.94
eta4 = 0.93
eta5 = 0.94
eta6 = 0.94
eta7 = 0.95
eta8 = 0.95

# --------------------------
#       Campo Solar
# --------------------------

w = 5.76
L = 600
A = w * L * 3

# ---------------------------
#           Vidrio
# ---------------------------

Dg = 0.115
Dg_i = 0.109
rho_g = 2323
Ag = np.pi* Dg * L
Vg = np.pi/4 * (Dg**2 - Dg_i**2)*L
eg = 0.85
rg = Dg/2
cp_g = 0.75*1000

# ---------------------------
#         Absorvedor
# ---------------------------

Da = 0.07
Da_i = 0.066
Aa = np.pi * Da * L 
ea = 0.18
ra = Da/2
rho_a = 7480
cp_a = 0.49*1000
Va = np.pi/4 * (Da**2 - Da_i**2) * L

# ---------------------------
#         Superheater
# ---------------------------

Lsup = 4
Dsup = 0.016
Dsup_i = 0.012
Asup_o = np.pi * Dsup * Lsup
Asup_i = np.pi * Dsup_i * Lsup
Vt_sup = 0.25
Vs_sup = 0.63
Asup = 500

# ---------------------------
#         Evaporador
# ---------------------------

Levp = 4
Devp = 0.016
Devp_i = 0.012
Vt_evp = 0.2
Vs_evp = 1.86
Aevp = 51

tevap = 509.188

# ----------------------------------------
#               Preheater
# ----------------------------------------

Lpre = 4
Dpre = 0.016
Dpre_i = 0.012
Vt_pre = 0.063
Vs_pre = 0.26
Apre = 9

# ------------------------------------------------------------------------
#                        Funciones auxiliares
# ------------------------------------------------------------------------

def kelvin(t):
    return t + 273.15

def celsius(t):
    return t - 273.15

def coef_global(h1, h2):
    output = 1/(1/h1 + 1/h2)
    return output

# ------------------------------------
#            Campo Solar
# ------------------------------------

def Re(m, t):
    """
    m: caudal
    t: temperatura
    """
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Da_i * mu)
    return re

def Pr(t):
    """
    t: temperatura
    """
    cp = fit_cp(t)
    mu = fit_mu(t)
    k = fit_k(t)
    pr = cp * mu / k
    return pr

def Nu(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re(m, t)
    pr = Pr(t)
    f = (1.82 * np.log10(re) - 1.64)**(-2)
    if re < 2300:
        nu = 5.22
    else:
        nu = (f/8 * (re - 1000) * pr/
             (1 + 12.7 * np.sqrt(f/8) * (pr**(2/3) - 1)))
    return nu

def dif_temp_htf(m, tin, tout):
    """
    m: caudal
    tin: temperatura de entrada
    tout: temperatura de salida
    """
    cp = fit_cp(tout)
    output = m * cp * (tin - tout)
    return output

def conv_htf(h, t, t2):
    """
    h: coef convectivo
    t: tempertura pared del absorbedor
    t2: temperatura htf
    """
    output = h * Aa * (t - t2)
    return output

def q_rad_a(t, t2):
    """
    t: temperatura del absorbedor
    t2: temperatura del vidrio
    """
    output = Aa * sigma * (t**4 - t2**4)/(1/ea + (1 - eg)/eg * ra/rg)
    return output

def coef_conv_wind(v):
    """
    v: velocidad del viento
    """
    output = 4 * v**0.58 * Dg
    return output

def conv_g(v, t, t2):
    """
    v: vel del viento
    t: temperatura del vidrio
    t2: temperatura ambiente
    """
    h = coef_conv_wind(v)
    output = h * Ag * (t - t2)
    return output

def q_rad_g(t, t2):
    """
    t: temperatura vidrio
    t2: temperatura cielo
    """
    tsky = 0.0553 * t2**1.5
    output = sigma * eg * Ag * (t**4 - tsky**4)
    return output

# ------------------------------
#          Superheater
# ------------------------------

def Re_w_sup(m, t):
    """
    m: caudal
    t: temperatura
    """
    p = 3.1 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Lsup)
    return re

def Re_htf_sup(m, t):
    """
    m: caudal
    t: temperatura
    """
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dsup_i * mu)
    return re

def Pr_w(t):
    """
    t: temperatura
    """
    p = 3.1 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'P', p, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    k = CP.PropsSI('L', 'T', t, 'P', p, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_sup(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re_htf_sup(m, t)
    pr = Pr(t)
    f = (1.82 * np.log10(re) - 1.64)**(-2)
    if re < 2300:
        nu = 5.22
    else:
        nu = (f/8 * (re - 1000) * pr/
             (1 + 12.7 * np.sqrt(f/8) * (pr**(2/3) - 1)))
    return nu

def Nu_shell_sup(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re_w_sup(m, t)
    pr = Pr_w(t)
    if re>1 and re<5 * 10**2:
        nu = 1.04 * re**0.4 * pr**0.36
    elif re>5 * 10**2 and re<10**3:
        nu = 0.71 * re**0.5 * pr**0.36
    elif re>10**3 and re<2 * 10**5:
        nu = 0.35 * re**0.6 * pr**0.36
    else:
        nu = 0.031 * re**0.8 * pr**0.36
    return nu

def dif_temp_w(m, t, t2):
    """
    m: caudal
    t: temperatura de entrada
    t2: temperatura de salida
    """
    p = 3.1 * 10**6
    cp = CP.PropsSI('C', 'T', t2, 'P', p, 'Water')
    output = m * cp * (t - t2)
    return output

def conv_tube_sup(h, t, t2):
    """
    h: coef convectivo
    t: temperatura de entrada
    t2: temperatura de salida
    """
    output = h * Asup * (t - t2)
    return output

def conv_shell_sup(h, t, t2):
    """
    h: coeficiente convectivo
    t: temperatura de entrada
    t2: temperatura de salida
    """
    output = h * Asup * (t - t2)
    return output

# ------------------------------
#          Evaporador
# ------------------------------

def Re_htf_evp(m, t):
    """
    m: caudal
    t: temperatura
    """
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Devp_i * mu)
    return re

def Re_w_evp(m, t):
    """
    m: caudal
    t: temperatura
    """
    p = 3.12 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Levp)
    return re

def Pr_w_evp(t):
    """
    t: temperatura
    """
    p = 3.12 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'P', p, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    k = CP.PropsSI('L', 'T', t, 'P', p, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_evp(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re_htf_evp(m, t)
    pr = Pr(t)
    f = (1.82 * np.log10(re) - 1.64)**(-2)
    if re < 2300:
        nu = 5.22
    else:
        nu = (f/8 * (re - 1000) * pr/
             (1 + 12.7 * np.sqrt(f/8) * (pr**(2/3) - 1)))
    return nu

def Nu_shell_evp(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re_w_evp(m, t)
    pr = Pr_w_evp(t)
    if re>1 and re<5 * 10**2:
        nu = 1.04 * re**0.4 * pr**0.36
    elif re>5 * 10**2 and re<10**3:
        nu = 0.71 * re**0.5 * pr**0.36
    elif re>10**3 and re<2 * 10**5:
        nu = 0.35 * re**0.6 * pr**0.36
    else:
        nu = 0.031 * re**0.8 * pr**0.36
    return nu

def conv_tube_evp(h, t, t2):
    """
    h: coef convectivo
    t: temperatura de entrada
    t2: temperatura de salida
    """
    output = h * Aevp * (t - t2)
    return output

def q_evap(m, t, tpre):
    """
    m: therminol flow rate
    t: output temperature therminol
    """
    mw = m_w
    cp_w = CP.PropsSI('C', 'T', tevap, 'Q', 0, 'Water')
    kw_evp = CP.PropsSI('L', 'T', tevap, 'P', pout_evp, 'Water')
    k_evp = fit_k(t)
    hg = CP.PropsSI('H', 'T', tevap, 'Q', 1, 'Water')
    hl = CP.PropsSI('H', 'T', tevap, 'Q', 0, 'Water')
    alphao_evp = Nu_shell_evp(mw, tevap) * kw_evp/Devp
    alphai_evp = Nu_tubes_evp(m, t) * k_evp/Devp_i
    U_evp = coef_global(alphao_evp, alphai_evp)
    Qconv = mw * (hg - hl) + m_w * cp_w * (tevap - tpre)
    E = abs(U_evp * Aevp * (t - tevap))
    if E < Qconv:
        m_w_sup = E/((hg - hl) + cp_w * (tevap - tpre))
    else:
        m_w_sup = m_w
    return m_w_sup

# ------------------------------
#          Preheater
# ------------------------------

def Re_w_pre(m, t):
    """
    m: caudal
    t: temperatura
    """
    p = 3.21 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Lpre)
    return re

def Re_htf_pre(m, t):
    """
    m: caudal
    t: temperatura
    """
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dpre_i * mu)
    return re

def Pr_w_pre(t):
    """
    t: temperatura
    """
    p = 3.21 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'P', p, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    k = CP.PropsSI('L', 'T', t, 'P', p, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_pre(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re_htf_pre(m, t)
    pr = Pr(t)
    f = (1.82 * np.log10(re) - 1.64)**(-2)
    if re < 2300:
        nu = 5.22
    else:
        nu = (f/8 * (re - 1000) * pr/
             (1 + 12.7 * np.sqrt(f/8) * (pr**(2/3) - 1)))
    return nu

def Nu_shell_pre(m, t):
    """
    m: caudal
    t: temperatura
    """
    re = Re_w_pre(m, t)
    pr = Pr_w_pre(t)
    if re>1 and re<5 * 10**2:
        nu = 1.04 * re**0.4 * pr**0.36
    elif re>5 * 10**2 and re<10**3:
        nu = 0.71 * re**0.5 * pr**0.36
    elif re>10**3 and re<2 * 10**5:
        nu = 0.35 * re**0.6 * pr**0.36
    else:
        nu = 0.031 * re**0.8 * pr**0.36
    return nu

def dif_temp_w_pre(m, t, t2):
    """
    m: caudal
    t: temperatura de entrada
    t2: temperatura de salida
    """
    p = 3.21 * 10**6
    cp = CP.PropsSI('C', 'T', t2, 'P', p, 'Water')
    output = m * cp * (t - t2)
    return output

def conv_tube_pre(h, t, t2):
    """
    h: coeficiente convectivo
    t: temperatura de entrada
    t2: temperatura de alida
    """
    output = h * Apre * (t - t2)
    return output

def conv_shell_pre(h, t, t2):
    """
    h: coeficiente convectivo
    t: temperatura de entrada
    t2: temperatura de salida
    """
    output = h * Apre * (t - t2)
    return output

# -----------------------------------------
#             Power output
# -----------------------------------------

def qsup(mw, mhtf, Thtf, Tw):
    """
    mw: Caudal de vapor en el superheater
    mhtf: Caudal de therminol en el superheater
    Thtf: temperatura therminol vp - 1
    Tw: temperatura del agua
    """
    kw = CP.PropsSI('L', 'T', Tw, 'P', pout_sup, 'Water')
    k = fit_k(Thtf)
    alphao = Nu_shell_sup(mw, Tw) * kw / Dsup
    alphai = Nu_tubes_sup(mhtf, Thtf) * k / Dsup_i
    U = coef_global(alphao, alphai)
    q = U * Asup * (Thtf - Tw)
    return q

def qevp(mhtf, Thtf):
    """
    mhtf: Caudal de therminol en el evaporador
    Thtf: temperatura de therminol vp - 1
    """
    kw = CP.PropsSI('L', 'T', tevap, 'P', pout_evp, 'Water')
    k = fit_k(Thtf)
    alphao = Nu_shell_evp(m_w, tevap) * kw / Devp
    alphai = Nu_tubes_evp(mhtf, Thtf) * k/ Devp_i
    U = coef_global(alphao, alphai)
    q = U * Aevp * (Thtf - tevap)
    return q

def qpre(mhtf, Thtf, Tw):
    """
    mhtf: caudal de therminol en el preheater
    Thtf: temperatura therminol vp -1
    Tw: temperatura del agua
    """
    kw = CP.PropsSI('L', 'T', Tw, 'P', pout_pre, 'Water')
    k = fit_k(Thtf)
    alphao = Nu_shell_pre(m_w, Tw) * kw / Dpre
    alphai = Nu_tubes_pre(mhtf, Thtf) * k/ Dpre_i
    U = coef_global(alphao, alphai)
    q = U * Apre * (Thtf - Tw)
    return q

def turbina(m, T):
    """
    m: Caudal de agua desde el superheater
    T: temperatura del agua
    """
    pout = 4.9 * 10**3
    sin = CP.PropsSI('S', 'T', T, 'P', pout_sup, 'Water')
    hin = CP.PropsSI('H', 'T', T, 'P', pout_sup, 'Water')
    sout = sin
    hout = CP.PropsSI('H', 'S', sout, 'P', pout, 'Water')
    tout = CP.PropsSI('T', 'S', sout, 'P', pout, 'Water')
    w = m * eta_tur * (hin - hout)
    return (w, tout)

# -----------------------------------------
#             Runge Kutta
# -----------------------------------------

def Rk1(fun, h, tn, yn, m, Tin):
    k1 = h*fun(tn, yn, m, Tin)
    return k1


def Rk2(fun, h, tn, yn, m, Tin):
    k1 = Rk1(fun, h, tn, yn, m, Tin)
    k2 = h*fun(tn+h/2, yn+1/2*k1, m, Tin)
    return k2


def Rk3(fun, h, tn, yn, m, Tin):
    k2 = Rk2(fun, h, tn, yn, m, Tin)
    k3 = h*fun(tn+h/2, yn+1/2*k2, m, Tin)
    return k3


def Rk4(fun, h, tn, yn, m, Tin):
    k3 = Rk3(fun, h, tn, yn, m, Tin)
    k4 = h*fun(tn+h, yn+k3, m, Tin)
    return k4

def paso_rk4(fun, h, tn, yn, m, Tin):
    """
    fun: funcion de la ODE
    h: paso de integracion
    tn: tiempo n derivada
    yn: valor de la funcion en n
    m: caudal
    Tin: temperatura de entrada
    """
    k1 = Rk1(fun, h, tn, yn, m, Tin)
    k2 = Rk2(fun, h, tn, yn, m, Tin)
    k3 = Rk3(fun, h, tn, yn, m, Tin)
    k4 = Rk4(fun, h, tn, yn, m, Tin)
    yn1 = yn+1/6*(k1+2*k2+2*k3+k4)
    return yn1

# ------------------------------------------------------------------------
#                                  ODE
# ------------------------------------------------------------------------

# Inicalizacion 

dt = 0.2 # cada x segundos
n_times = int(23 * 60 * 60/ dt) + 1
time = np.linspace(0, 23 * 60 * 60, n_times)

# Variables campo solar

u = np.zeros((len(time), 3))
m_htf = np.zeros(len(time))
tin = np.zeros(len(time))

# Variable tren de intercambiadores

u3 = np.zeros((1, 5))
time3 = np.array([])
m_sgs = np.array([])
m_sup = np.array([])
Qpre = np.zeros(1)
Qevp = np.zeros(1)
Qsup = np.zeros(1)
Wt = np.zeros(1)

# Parametro tren de intercambiadores

m_w = 1.6
m_rat_pb = 20
T_w_in = kelvin(104)
pout_sup = 3.1 * 10**6
pout_evp = 3.12 * 10**6
pout_pre = 3.21 * 10**6

# Parametro turbina

eta_gen = 0.99
eta_tur = 0.6

# VARIABLES PI CAMPO SOLAR

m_htf = np.zeros(len(time))
err_sf = np.zeros(len(time))
ierr_sf = np.zeros(len(time))
op = np.zeros(len(time))
op2 = np.zeros(len(time))
Kc_sf = 0.01
taui_sf = 1e6
SP_sf = kelvin(400)
OP_hi = 1
OP_lo = 0

# ------------------------------------------------------------------------
#                          FUNCION CAMPO SOLAR
# ------------------------------------------------------------------------

def dTdt(t, V, m, Tin):
    """
    t: tiempo a resolver la ODE
    V: vector de incognitas
    m: caudal
    Tin: temperatura de entrada
    """
    # Incognitas
    T_htf, T_a, T_g = V

    # Parametros del campo solar
    t_amb = fit_temp(t/3600)
    vel = fit_vel(t/3600)
    DNI = fit_rad(t/3600)

    # Prop. termodinamicas
    rho = fit_rho(T_htf)
    cp = fit_cp(T_htf)
    k = fit_k(T_htf)

    # ------------------------------------------------------------
    #                             Aceite
    # ------------------------------------------------------------
        
    h = Nu(m, T_htf) * k/Da_i
    a1 = dif_temp_htf(m, Tin, T_htf)
    a2 = conv_htf(h, T_a, T_htf)
    v1 = (a1 + a2)/(rho * Va * cp)

    # ------------------------------------------------------------
    #                          Absorbedor
    # ------------------------------------------------------------

    b1 = (DNI * A * eta1 * eta2 * eta3 * 
          eta4 * eta5 * eta6 * eta7 * eta8)
    b2 = q_rad_a(T_a, T_g)
    v2 = (b1 - a2 - b2)/(rho_a * Va * cp_a)

    # ------------------------------------------------------------
    #                          Vidrio
    # ------------------------------------------------------------

    c1 = conv_g(vel, T_g, t_amb)
    c2 = q_rad_g(T_g, t_amb)
    v3 = (b2 - c1 - c2)/(rho_g * Vg * cp_g)

    return np.array([v1, v2, v3])

# ------------------------------------------------------------------------
#                  FUNCION TREN DE INTERCAMBIADORES
# ------------------------------------------------------------------------

def dpbdt(t, V, m, Tin):

    """
    t: tiempo de integracion
    V: vector de incognitas
    m: caudal
    Tin: temperatura de entrada
    """
        
    # Incognitas

    Tsup_htf, Tsup_w, Tevp_htf, Tpre_htf, Tpre_w = V

    # -----------------------------
    #     Prop termodinamicas
    # -----------------------------

    # Aceite

    rho = fit_rho(Tsup_htf)
    k = fit_k(Tsup_htf)
    cp = fit_cp(Tsup_htf)

    # Agua
 
    kw = CP.PropsSI('L', 'T', Tsup_w, 'P', pout_sup, 'Water') 
    rhow = CP.PropsSI('D', 'T', Tsup_w, 'P', pout_sup, 'Water')
    cpw = CP.PropsSI('C', 'T', Tsup_w, 'P', pout_sup, 'Water')

    # ----------------------------------------------------------------
    #                            DERIVADA
    # ----------------------------------------------------------------

    # ---------------------------
    #      Caudal evaporado
    # ---------------------------

    m_w_sup = q_evap(m, Tevp_htf, Tpre_w)

    alphao = Nu_shell_sup(m_w_sup, Tsup_w) * kw/Dsup
    alphai = Nu_tubes_sup(m, Tsup_htf) * k/Dsup_i
    U = coef_global(alphao, alphai)

    # DIF TEMP

    a1 = dif_temp_htf(m, Tin, Tsup_htf)/(rho * Vt_sup * cp)
    b1 = dif_temp_w(m_w_sup, tevap, Tsup_w)/(rhow * Vs_sup * cpw)

    # Conveccion

    a2 = conv_tube_sup(U, Tsup_w, Tsup_htf)/(rho * Vt_sup * cp)
    b2 = conv_shell_sup(U, Tsup_htf, Tsup_w)/(rhow * Vs_sup * cpw)

    # Derivada

    v1 = a1 + a2
    v2 = b1 + b2

    # ----------------------------------------------------------------
    #                           Evaporador
    # ----------------------------------------------------------------

    # -----------------------------
    #     Prop termodinamicas
    # -----------------------------

    # Aceite

    rho_evp = fit_rho(Tevp_htf)
    k_evp = fit_k(Tevp_htf)
    cp_evp = fit_cp(Tevp_htf)

    # Agua
 
    kw_evp = CP.PropsSI('L', 'T', tevap, 'P', pout_evp, 'Water') 
    rhow_evp = CP.PropsSI('D', 'T', tevap, 'P', pout_evp, 'Water')
    cpw_evp = CP.PropsSI('C', 'T', tevap, 'P', pout_evp, 'Water')

    # ---------------------------
    #          DERIVADA
    # ---------------------------

    alphao_evp = Nu_shell_evp(m_w, tevap) * kw_evp/Devp
    alphai_evp = Nu_tubes_evp(m, Tevp_htf) * k_evp/Devp_i
    U_evp = coef_global(alphao_evp, alphai_evp)

    # DIF TEMP

    c1 = dif_temp_htf(m, Tsup_htf, Tevp_htf)/(rho_evp * Vt_evp * cp_evp)

    # Conveccion

    c2 = conv_tube_evp(U_evp, tevap, Tevp_htf)/(rho_evp * Vt_evp * cp_evp) 

    # Derivada

    v3 = c1 + c2
    
    # ----------------------------------------------------------------
    #                         PREHEATER
    # ----------------------------------------------------------------

    # -----------------------------
    #     Prop termodinamicas
    # -----------------------------

    # Aceite

    rho_pre = fit_rho(Tpre_htf)
    k_pre = fit_k(Tpre_htf)
    cp_pre = fit_cp(Tpre_htf)

    # Agua
 
    kw_pre = CP.PropsSI('L', 'T', Tpre_w, 'P', pout_pre, 'Water') 
    rhow_pre = CP.PropsSI('D', 'T', Tpre_w, 'P', pout_pre, 'Water')
    cpw_pre = CP.PropsSI('C', 'T', Tpre_w, 'P', pout_pre, 'Water')

    # ----------------------------------------------------------------
    #                            DERIVADA
    # ----------------------------------------------------------------

    alphao_pre = Nu_shell_pre(m_w, Tpre_w) * kw_pre/Dpre
    alphai_pre = Nu_tubes_pre(m, Tpre_htf) * k_pre/Dpre_i
    U_pre = coef_global(alphao_pre, alphai_pre)

    # DIF TEMP

    d1 = dif_temp_htf(m, Tevp_htf, Tpre_htf)/(rho_pre * Vt_pre * cp_pre)
    e1 = dif_temp_w_pre(m_w, T_w_in, Tpre_w)/(rhow_pre * Vs_pre * cpw_pre)

    # Conveccion

    d2 = conv_tube_pre(U_pre, Tpre_w, Tpre_htf)/(rho_pre * Vt_pre * cp_pre)
    e2 = conv_shell_pre(U_pre, Tpre_htf, Tpre_w)/(rhow_pre * Vs_pre * cpw_pre)

    # Derivada

    v4 = d1 + d2
    v5 = e1 + e2

    return np.array([v1, v2, v3, v4, v5])

# Iteracion

# Condiciones iniciales

V_0 = [kelvin(240), kelvin(240), kelvin(90)]
u3_0 = np.array([kelvin(368), kelvin(365), kelvin(260), kelvin(235), kelvin(190)])
T_in = kelvin(240)
m_sf = 10
Qpre_0 = qpre(m_rat_pb, u3_0[3], u3_0[4])
Qevp_0 = qevp(m_rat_pb, u3_0[2])
Qsup_0 = qsup(m_w, m_rat_pb, u3_0[0], u3_0[1])
Wt_0 = turbina(m_w, u3_0[1])[0]

u[0, :] = V_0
u3[0, :] = u3_0
m_htf[0] = m_sf
tin[0] = T_in
Qpre[0] = Qpre_0
Qevp[0] = Qevp_0
Qsup[0] = Qsup_0
Wt[0] = Wt_0

for j in tqdm(range(1, len(time))):
    t_eval = [time[j - 1], time[j]]
    T_sf = paso_rk4(dTdt, dt, time[j - 1], u[j - 1, :], m_sf, T_in)
    u[j, :] = T_sf 
    PV_sf = u[j,0]
    err_sf[j] = SP_sf - PV_sf
    ierr_sf[j] = ierr_sf[j - 1] + dt * err_sf[j]
    P = - Kc_sf * err_sf[j]
    I = Kc_sf/taui_sf * ierr_sf[j]
    OP_sf = P + I
    op[j] = OP_sf   
    m_sf = min(1000, max(10, m_sf + OP_sf))
    m_htf[j] = m_sf
    op2[j] = OP_sf

    if u[j, 0]>kelvin(380):
        if m_htf[j]>m_rat_pb:
            m_pb = m_rat_pb
        else:
            m_pb = m_htf[j]

        Tin_pb = u[j, 0]

        T_pb = paso_rk4(dpbdt, dt, time[j - 1], u3[-1], m_pb, Tin_pb)
        u3 = np.append(u3, [[T_pb[0], T_pb[1], T_pb[2], T_pb[3], T_pb[4]]], axis = 0)
        
        time3 = np.append(time3, time[j])
        m_w_sup = q_evap(m_pb, T_pb[2], T_pb[4])
        Qpre = np.append(Qpre, qpre(m_pb, u3[-1, 3], u3[-1, 4]))
        Qevp = np.append(Qevp, qevp(m_pb, u3[-1, 2]))
        Qsup = np.append(Qsup, qsup(m_w_sup, m_pb, u3[-1, 0], u3[-1, 1]))
        Wt = np.append(Wt, turbina(m_w_sup, u3[-1, 1])[0])
        m_sgs = np.append(m_sgs, m_pb)
        m_sup = np.append(m_sup, m_w_sup)
        T_in = u3[-1, 3]

Qpb = Qpre + Qevp + Qsup

caudal = open('caudal_sept2.pkl', 'wb')
pickle.dump(m_htf, caudal)
caudal.close()

tiempo = open('tiempo_sept2.pkl', 'wb')
pickle.dump(time, tiempo)
tiempo.close()

temp_sf = open('temp_sept2.pkl', 'wb')
pickle.dump(u, temp_sf)
temp_sf.close()

tiempo2 = open('tiempo2_sept2.pkl', 'wb')
pickle.dump(time3, tiempo2)
tiempo2.close()

temp_sgs = open('temp_sgs_sept2.pkl', 'wb')
pickle.dump(u3, temp_sgs)
temp_sgs.close()

power = open('power_sept2.pkl', 'wb')
pickle.dump(Qpb, power)
power.close()

work = open('work_sept2.pkl', 'wb')
pickle.dump(Wt, work)
work.close()

q_sup = open('q_sup_sept2.pkl', 'wb')
pickle.dump(m_sup, q_sup)
q_sup.close()

