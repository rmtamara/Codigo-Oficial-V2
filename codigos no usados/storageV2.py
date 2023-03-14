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
base_mes = pd.read_excel(file, sheet_name='Sheet3')
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
#                             Molten salt
# ------------------------------------------------------------------------

def cp_ms(t):
    T = t - 273.15
    output = 1443 + 0.172 * T
    return output

def mu_ms(t):
    T = t - 273.15
    output = ((22.174 - 0.12 * T + 2.281 * 10**(-4) * T**2 
               - 1.474 * 10**(-7) * T**3)/1000)
    return output

def k_ms(t):
    T = t - 273.15
    output = 0.443 + 1.9*10**(-4) * T
    return output

def h_ms(t):
    T = t - 273.15
    output = 0.086 * T**2 + 1396.0182 * T
    return output

def den_ms(t):
    T = t - 273.15
    output = 2090 - 0.636 * T
    return output

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
A = w * L * 4

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
Asup = 366

# ---------------------------
#         Evaporador
# ---------------------------

Levp = 4
Devp = 0.016
Devp_i = 0.012
Vt_evp = 0.2
Vs_evp = 1.86
Aevp = 30

tevap = 509.188

# ----------------------------------------
#               Preheater
# ----------------------------------------

Lpre = 4
Dpre = 0.016
Dpre_i = 0.012
Vt_pre = 0.063
Vs_pre = 0.26
Apre = 5

# ------------------------------------------
#         Thermal Energy Storage
# ------------------------------------------

# Heat exchanger oil/salt
Aos = 16 
Vt_os = 0.111
Vs_os =  0.255
Dos = 0.0127
Dos_i = 0.0125
Los = 3.4
rt = 8
Ab = np.pi * rt**2
ht = 7

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

# Thermal energy storage

def Re_htf_hx1(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dos_i * mu)
    return re

def Re_ms_hx1(m, t):
    mu = mu_ms(t)
    re = m/(mu * Los)
    return re

def Pr_ms_hx1(t):
    cp =  cp_ms(t)
    mu = mu_ms(t)
    k = k_ms(t)
    pr = cp * mu / k
    return pr

def Nu_tubes_hx1(m, t):
    re = Re_htf_hx1(m, t)
    pr = Pr(t)
    f = (1.82 * np.log10(re) - 1.64)**(-2)
    if re < 2300:
        nu = 5.22
    else:
        nu = (f/8 * (re - 1000) * pr/
             (1 + 12.7 * np.sqrt(f/8) * (pr**(2/3) - 1)))
    return nu

def Nu_shell_hx1(m, t):
    re = Re_ms_hx1(m, t)
    pr = Pr_ms_hx1(t)
    if re>1 and re<5 * 10**2:
        nu = 1.04 * re**0.4 * pr**0.36
    elif re>5 * 10**2 and re<10**3:
        nu = 0.71 * re**0.5 * pr**0.36
    elif re>10**3 and re<2 * 10**5:
        nu = 0.35 * re**0.6 * pr**0.36
    else:
        nu = 0.031 * re**0.8 * pr**0.36
    return nu 

def conv_shell(h, t, t2):
    """
    t: temperature tube out
    t2: temperature shell out
    h: coef global de transferencia de calor
    """
    output = h * Aos * (t - t2)
    return output

def conv_tube(h, t, t2):
    """
    t: temperature shell out
    t2: temperature tube out
    h: coef global de transeferencia de calor
    """
    output = h * Aos * (t - t2)
    return output

def coef_global(h1, h2):
    output = 1/(1/h1 + 1/h2)
    return output

def dif_temp_ms(m, t, t2):
    cp = cp_ms(t2)
    output = m * cp * (t - t2)
    return output

# -----------------------------------------
#             Runge Kutta 2
# -----------------------------------------

def Rk1_2(fun, h, tn, yn, m, m2, Tin):
    k1 = h*fun(tn, yn, m, m2, Tin)
    return k1


def Rk2_2(fun, h, tn, yn, m, m2, Tin):
    k1 = Rk1_2(fun, h, tn, yn, m, m2, Tin)
    k2 = h*fun(tn+h/2, yn+1/2*k1, m, m2, Tin)
    return k2


def Rk3_2(fun, h, tn, yn, m, m2, Tin):
    k2 = Rk2_2(fun, h, tn, yn, m, m2, Tin)
    k3 = h*fun(tn+h/2, yn+1/2*k2, m, m2, Tin)
    return k3


def Rk4_2(fun, h, tn, yn, m, m2, Tin):
    k3 = Rk3_2(fun, h, tn, yn, m, m2, Tin)
    k4 = h*fun(tn+h, yn+k3, m, m2, Tin)
    return k4

def paso_rk4_2(fun, h, tn, yn, m, m2, Tin):
    """
    fun: funcion de la ODE
    h: paso de integracion
    tn: tiempo n derivada
    yn: valor de la funcion en n
    m: caudal
    Tin: temperatura de entrada
    """
    k1 = Rk1_2(fun, h, tn, yn, m, m2, Tin)
    k2 = Rk2_2(fun, h, tn, yn, m, m2, Tin)
    k3 = Rk3_2(fun, h, tn, yn, m, m2, Tin)
    k4 = Rk4_2(fun, h, tn, yn, m, m2, Tin)
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

# Variables almacenamiento
u2 = np.zeros((1, 2))
m_ms = np.array([])
time2 = np.array([])

# Variable tren de intercambiadores

u3 = np.zeros((1, 5))
time3 = np.array([])
m_sgs = np.array([])
m_sup = np.array([])
Qpre = np.zeros(1)
Qevp = np.zeros(1)
Qsup = np.zeros(1)
Wt = np.zeros(1)

# Parametro almacenamiento
m_ms0 = 1

# Parametro tren de intercambiadores

m_w = 0.9
m_rat_pb = 15
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
Kc_sf = 0.1
taui_sf = 1e6
SP_sf = kelvin(400)
OP_hi = 1
OP_lo = 0

# Variable PI TES
err_tes = np.zeros(1)
ierr_tes = np.zeros(1)
Kc_tes = 10
taui_tes = 1e6
SP_tes = kelvin(400)

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

def dTESdt(t, V, mt, mm, Tin):

    # Incognitas
    T_htf, T_ms = V

    # Parametros
    Tin_ms = kelvin(296)

    # ----------------------------------------------------------------
    #                        Prop. termodinamicas
    # ----------------------------------------------------------------
    
    # Aceite
    rho = fit_rho(T_htf)
    cp = fit_cp(T_htf)
    k = fit_k(T_htf)

    # Sal disuelta
    rhoms = den_ms(T_ms)
    cpms = cp_ms(T_ms)
    kms = k_ms(T_ms)

    # ----------------------------------------------------------------
    #                       Derivada temperatura
    # ----------------------------------------------------------------

    # Aceite     
    alphao = Nu_shell_hx1(mm, T_ms) * kms/Dos
    alphai = Nu_tubes_hx1(mt, T_htf) * k/Dos_i
    U = coef_global(alphao, alphai)
    a1 = dif_temp_htf(mt, Tin, T_htf)
    a2 = conv_tube(U, T_ms, T_htf)
    v1 = (a1 + a2)/(rho * Vt_os * cp)

    # Sal disuelta
        
    b1 = dif_temp_ms(mm, Tin_ms, T_ms)
    b2 = conv_shell(U, T_htf, T_ms)
    v2 = (b1 + b2)/(rhoms * Vs_os * cpms)

    # ----------------------------------------------------------------
    #           Variacion de altura en el estanque
    # ----------------------------------------------------------------

    # v3 = dot_m/(Ab * rhoms) - Lsf * (-0.636) * v2/rhoms

    return np.array([v1, v2])

# Iteracion

# Condiciones iniciales

V_0 = [kelvin(240), kelvin(240), kelvin(90)]
u2_0 = [kelvin(290), kelvin(380)]
T_in = kelvin(240)
m_sf = 10
m_ms0 = 1

u[0, :] = V_0
m_htf[0] = m_sf
u2[0, :] = u2_0

for j in tqdm(range(1, len(time))):
    t_eval = [time[j - 1], time[j]]    
    u[j, :] = odeint(dTdt, u[j - 1, :], t_eval, 
                     tfirst = True, args = (m_sf, T_in))[1]
    PV_sf = u[j,0]
    err_sf[j] = SP_sf - PV_sf
    ierr_sf[j] = ierr_sf[j - 1] + dt * err_sf[j]
    P = - Kc_sf * err_sf[j]
    I = Kc_sf/taui_sf * ierr_sf[j]
    OP_sf = P + I
    op[j] = OP_sf 
    """ 
    if OP_sf > OP_hi:
        OP_sf = -1
        ierr_sf[j] = ierr_sf[j] - err_sf[j] * dt
    elif OP_sf < OP_lo:
        OP_sf = 1
        ierr_sf[j] = ierr_sf[j] - err_sf[j] * dt 
    """   
    m_sf = min(1000, max(10, m_sf + OP_sf))
    m_htf[j] = m_sf
    op2[j] = OP_sf

    if celsius(u[j, 0])>380 and m_sf - m_rat_pb>1:
        m_htf_tes = m_sf - m_rat_pb
        T_tes = paso_rk4_2(dTESdt, dt, time[j - 1], u2[-1],
                         m_htf_tes, m_ms0, u[j, 0])
        u2 = np.append(u2, [[T_tes[0], T_tes[1]]], axis = 0)

        PV_tes = u2[-1, 1]
        err_tes = np.append(err_tes, SP_tes - PV_tes)
        ierr_sf = np.append(ierr_sf, ierr_sf[-1] + dt * err_sf[-1])
        P_tes = - Kc_tes * err_tes[-1]
        I_tes = Kc_tes/taui_tes * ierr_tes[-1]
        OP_tes = P_tes + I_tes
        m_ms0 = min(1000, max(1, m_ms0 + OP_tes))
        m_ms = np.append(m_ms, m_ms0)
        time2 = np.append(time2, time[j])
