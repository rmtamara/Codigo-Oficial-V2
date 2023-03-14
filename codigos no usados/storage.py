"""
En este script se modela de forma transiente una planta termosolar
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

# ------------------------------------------------------------------------
#                            Base de Datos
# ------------------------------------------------------------------------

file = 'base.xlsx'
base_mes = pd.read_excel(file, sheet_name='Hoja1')
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

# Solar Field
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
nloops = 3
A = w * L * nloops 

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

# ------------------------------------------
#         Thermal Energy Storage
# ------------------------------------------

# Heat exchanger oil/salt
Aos = 83.02 
Vt_os = 0.111
Vs_os =  0.255
Dos = 0.0127
Dos_i = 0.0125
Los = 3.4
rt = 8
Ab = np.pi * rt**2
ht = 7

# Heat exchanger salt/steam

# ------------------------------------------
#              Superheater
# ------------------------------------------

Lsup = 4
Dsup = 0.019
Dsup_i = 0.015
Asup_o = np.pi * Dsup * Lsup
Asup_i = np.pi * Dsup_i * Lsup
Vt_sup = 13.41
Vs_sup = 11.46
Asup = 4995

# ------------------------------------------
#                Evaporador
# ------------------------------------------

Levp = 4
Devp = 0.016
Devp_i = 0.012
Vt_evp = 0.2
Vs_evp = 1.86
Aevp = 25

tevap = 509.188

# ----------------------------------------
#               Preheater
# ----------------------------------------

Lpre = 4
Dpre = 0.016
Dpre_i = 0.012
Vt_pre = 0.063
Vs_pre = 0.26
Apre = 30

# ------------------------------------------------------------------------
#                        Funciones auxiliares
# ------------------------------------------------------------------------

# Solar Field
def kelvin(t):
    return t + 273.15

def celsius(t):
    return t - 273.15

def Re(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Da_i * mu)
    return re

def Pr(t):
    cp = fit_cp(t)
    mu = fit_mu(t)
    k = fit_k(t)
    pr = cp * mu / k
    return pr

def Nu(m, t):
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

# Superheater

def Re_w_sup(m, t):
    p = 3.1 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Lsup)
    return re

def Re_htf_sup(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dsup_i * mu)
    return re

def Pr_w(t):
    p = 3.1 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'P', p, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    k = CP.PropsSI('L', 'T', t, 'P', p, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_sup(m, t):
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
    p = 3.1 * 10**6
    cp = CP.PropsSI('C', 'T', t2, 'P', p, 'Water')
    output = m * cp * (t - t2)
    return output

def conv_tube_sup(h, t, t2):
    output = h * Asup * (t - t2)
    return output

def conv_shell_sup(h, t, t2):
    output = h * Asup * (t - t2)
    return output

# Evaporador

def Re_w_evp(m, t):
    p = 3.12 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Levp)
    return re

def Re_htf_evp(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Devp_i * mu)
    return re

def Pr_w_evp(t):
    p = 3.12 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'P', p, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    k = CP.PropsSI('L', 'T', t, 'P', p, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_evp(m, t):
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

def dif_ent_htf(m, t, t2):
    hin = fit_h(t)
    hout = fit_h(t2)
    output = m * (hin - hout)
    return output

def conv_tube_evp(h, t, t2):
    output = h * Aevp * (t - t2)
    return output

def conv_shell_evp(h, t, t2):
    output = h * Aevp * (t - t2)
    return output

# Preheater

def Re_w_pre(m, t):
    p = 3.21 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Lpre)
    return re

def Re_htf_pre(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dpre_i * mu)
    return re

def Pr_w_pre(t):
    p = 3.21 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'P', p, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    k = CP.PropsSI('L', 'T', t, 'P', p, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_pre(m, t):
    re = Re_w_pre(m, t)
    pr = Pr_w_pre(t)
    f = (1.82 * np.log10(re) - 1.64)**(-2)
    if re < 2300:
        nu = 5.22
    else:
        nu = (f/8 * (re - 1000) * pr/
             (1 + 12.7 * np.sqrt(f/8) * (pr**(2/3) - 1)))
    return nu

def Nu_shell_pre(m, t):
    re = Re_htf_pre(m, t)
    pr = Pr(t)
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
    p = 3.21 * 10**6
    cp = CP.PropsSI('C', 'T', t2, 'P', p, 'Water')
    output = m * cp * (t - t2)
    return output

def conv_tube_pre(h, t, t2):
    output = h * Apre * (t - t2)
    return output

def conv_shell_pre(h, t, t2):
    output = h * Apre * (t - t2)
    return output

# ------------------------------------------------------------------------
#                                 ODE
# ------------------------------------------------------------------------

m_htf = np.linspace(20, 40, 21)
time = np.linspace(0, 23*60*60, 69)
n = len(m_htf)
m = len(time)
temp_htf = dict()
temp_abs = dict()
temp_g = dict()

for i in range(n):
    temp_htf[m_htf[i]] = np.zeros(m)
    temp_abs[m_htf[i]] = np.zeros(m)
    temp_g[m_htf[i]] = np.zeros(m)



for i in range(n):
    def dTdt(t, V):
        # Incognitas
        T_htf, T_a, T_g = V

        # Parametros del campo solar
        dot_m = m_htf[i]
        t_amb = fit_temp(t/3600)
        vel = fit_vel(t/3600)
        DNI = fit_rad(t/3600)
        Tin = kelvin(296)

        # Prop. termodinamicas
        rho = fit_rho(T_htf)
        cp = fit_cp(T_htf)
        k = fit_k(T_htf)

        # ----------------------------------------------------------------
        #                             Aceite
        # ----------------------------------------------------------------
        
        h = Nu(dot_m, T_htf) * k/Da_i
        a1 = dif_temp_htf(dot_m, Tin, T_htf)
        a2 = conv_htf(h, T_a, T_htf)
        v1 = (a1 + a2)/(rho * Va * cp)

        # ----------------------------------------------------------------
        #                          Absorbedor
        # ----------------------------------------------------------------

        b1 = (DNI * A * eta1 * eta2 * eta3 * eta4 * 
              eta5 * eta6 * eta7 * eta8) 
        b2 = q_rad_a(T_a, T_g)
        v2 = (b1 - a2 - b2)/(rho_a * Va * cp_a)

        # ----------------------------------------------------------------
        #                          Vidrio
        # ----------------------------------------------------------------

        c1 = conv_g(vel, T_g, t_amb)
        c2 = q_rad_g(T_g, t_amb)
        v3 = (b2 - c1 - c2)/(rho_g * Vg * cp_g)

        return np.array([v1, v2, v3])

    V_0 = [kelvin(296), kelvin(296), kelvin(90)]                
    Sol = solve_ivp(dTdt, t_span=(0, max(time)), y0 = V_0, 
                    t_eval = time, method = 'Radau', rtol = 1e-5)
    temp_htf[m_htf[i]] = celsius(Sol.y[0])
    temp_abs[m_htf[i]] = celsius(Sol.y[1])
    temp_g[m_htf[i]] = celsius(Sol.y[2])

m_PB = 20 # Caudal de operacion en PB
m_TES_list = np.zeros((n,2))
for i in range(n):
    if max(temp_htf[m_htf[i]])>396:
        m_TES_list[i,:] = (m_htf[i] - m_PB, max(temp_htf[m_htf[i]]))

m_tes = max(m_TES_list[:,0])
m_sf = m_tes + m_PB


# Tiempo de integracion para TES
time_tes = []

# Temperaturas superiores a la de almacenamiento en TES
tr_tes = 370 # Rated condition in high temperature tank
temp_tes = []
for j in range(m):
    if temp_htf[m_sf][j] > tr_tes:
        time_tes.append(time[j])
        temp_tes.append(temp_htf[m_sf][j])

fit_TES = CubicSpline(time_tes, temp_tes)
# ------------------------------------------------------------------------
#                              Bloque TES
# ------------------------------------------------------------------------

Tin_ms = kelvin(296)
Tout_ms = kelvin(393)

m_ms = np.linspace(1, 3, 3)

temp_htf_hx1 = dict()
temp_ms_hx1 = dict()
height = dict()

for i in range(len(m_ms)):
    temp_htf_hx1[m_ms[i]] = np.zeros(len(time_tes))
    temp_ms_hx1[m_ms[i]] = np.zeros(len(time_tes))
    height[m_ms[i]] = np.zeros(len(time_tes))

for i in range(len(m_ms)):
    def dTESdt(t, V):
        # Incognitas
        T_htf, T_ms, Lsf = V
        T_in = kelvin(fit_TES(t))

        # Parametros

        dot_m = m_ms[i]

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
        alphao = Nu_shell_hx1(dot_m, T_ms) * kms/Dos
        alphai = Nu_tubes_hx1(1.57, T_htf) * k/Dos_i
        U = coef_global(alphao, alphai)
        a1 = dif_temp_htf(1.57, T_in, T_htf)
        a2 = conv_tube(U, T_ms, T_htf)
        v1 = (a1 + a2)/(rho * Vt_os * cp)

        # Sal disuelta
        
        b1 = dif_temp_ms(dot_m, Tin_ms, T_ms)
        b2 = conv_shell(U, T_htf, T_ms)
        v2 = (b1 + b2)/(rhoms * Vs_os * cpms)

        # ----------------------------------------------------------------
        #           Variacion de altura en el estanque
        # ----------------------------------------------------------------

        v3 = dot_m/(Ab * rhoms) - Lsf * (-0.636) * v2/rhoms

        return np.array([v1, v2, v3])

    Vtes_0 = [kelvin(296), kelvin(393), 0.5]                
    Sol_tes = solve_ivp(dTESdt, t_span=(min(time_tes), max(time_tes)), 
                        y0 = Vtes_0, t_eval = time_tes, 
                        method = 'Radau', rtol = 1e-5)
    temp_htf_hx1[m_ms[i]] = celsius(Sol_tes.y[0])
    temp_ms_hx1[m_ms[i]] = celsius(Sol_tes.y[1])
    height[m_ms[i]] = Sol_tes.y[2]

# ------------------------------------------------------------------------
#                            Bloque SGS
# ------------------------------------------------------------------------

# Superheater

time_sup = [] # Tiempo donde se cumple temperatura superior a 383 C.
temp_sup = [] # Temperatura de entrada del HTF.

for i in range(len(time)):
    if temp_htf[m_PB][i]>383:
        time_sup.append(time[i])
        temp_sup.append(temp_htf[m_PB][i])

fit_sup = CubicSpline(time_sup, temp_sup)

def dSGSdt(t, V):

    # Incognitas: Temperaturas de salida de ambos fluidos
    T_htf, T_wsup, Tevp_htf, Tpre_htf, Tpre_w = V

    # Parametros

    T_in = kelvin(fit_sup(t))
    T_w_in = tevap
    pout_sup = 3.1 * 10**6
    pout_evp = 3.12 * 10**6
    pout_pre = 3.21 * 10**6
    m_w = 1.8

    # --------------------------------
    #          Superheater
    # --------------------------------

    # Aceite

    rho = fit_rho(T_htf)
    k = fit_k(T_htf)
    cp = fit_cp(T_htf)

    # Agua

    kw = CP.PropsSI('L', 'T', T_wsup, 'P', pout_sup, 'Water') 
    rhow = CP.PropsSI('D', 'T', T_wsup, 'P', pout_sup, 'Water')
    cpw = CP.PropsSI('C', 'T', T_wsup, 'P', pout_sup, 'Water')

    # Global heat transfer coef.

    alphao = Nu_shell_sup(m_w, T_wsup) * kw/Dsup_i
    alphai = Nu_tubes_sup(m_PB, T_htf) * k/Dsup
    U = coef_global(alphao, alphai)

    # Dif temp

    a1 = dif_temp_htf(m_PB, T_in, T_htf)/(rho * Vt_sup * cp)
    b1 = dif_temp_w(m_w, T_w_in, T_wsup)/(rhow * Vs_sup * cpw)

    # Conveccion

    a2 = conv_tube_sup(U, T_wsup, T_htf)/(rho * Vt_sup * cp)
    b2 = conv_shell_sup(U, T_htf, T_wsup)/(rhow * Vs_sup * cpw)

    # Derivada

    v1 = a1 + a2
    v2 = b1 + b2

    # --------------------------------
    #          Evaporator
    # --------------------------------

    # Aceite

    rho_evp = fit_rho(Tevp_htf)
    k_evp = fit_k(Tevp_htf)
    cp_evp = fit_cp(Tevp_htf)

    # Agua

    rhow_evp = CP.PropsSI('D', 'T', tevap, 'P', pout_evp, 'Water')
    cpw_evp = CP.PropsSI('C', 'T', tevap, 'P', pout_evp, 'Water')
    kw_evp = CP.PropsSI('L', 'T', tevap, 'P', pout_evp, 'Water')

    # Global heat transfer coef.

    alphao_evp = Nu_shell_evp(m_w, tevap) * kw_evp/Devp
    alphai_evp = Nu_tubes_evp(m_PB, Tevp_htf) * k_evp/Devp_i
    U_evp = coef_global(alphao_evp, alphai_evp)

    # Diferencia de entalpia

    c1 = dif_ent_htf(m_PB, T_htf, Tevp_htf)/(rho_evp * Vt_evp * cp_evp)

    # Conveccion

    c2 = conv_tube_evp(U_evp, tevap, Tevp_htf)/(rho_evp * Vt_evp * cp_evp)

    v3 = c1 + c2

    # ------------------------------
    #           Preheater
    # ------------------------------

    # Propiedades termodinamicas

    # Aceite

    rho_pre = fit_rho(Tpre_htf)
    k_pre = fit_k(Tpre_htf)
    cp_pre = fit_cp(Tpre_htf)

    # Agua

    kw_pre = CP.PropsSI('L', 'T', Tpre_w, 'Q', 0, 'Water') 
    rhow_pre = CP.PropsSI('D', 'T', Tpre_w, 'Q', 0, 'Water')
    cpw_pre = CP.PropsSI('C', 'T', Tpre_w, 'Q', 0, 'Water')

    # Global heat transfer coef

    alphao_pre = Nu_shell_pre(m_PB, Tpre_htf) * k_pre/Dpre_i
    alphai_pre = Nu_tubes_pre(m_w, Tpre_w) * kw_pre/Dpre
    U_pre = coef_global(alphao_pre, alphai_pre)

    # Diferencia de temp

    d1 = dif_temp_htf(m_PB, Tevp_htf, Tpre_htf)
    e1 = dif_temp_w_pre(m_w, 104+273.15, Tpre_w)

    # Conveccion
    
   
    d2 = conv_shell_pre(U_pre, Tpre_w, Tpre_htf)
    e2 = conv_tube_pre(U_pre, Tpre_htf, Tpre_w)

    # Derivada

    v4 = (d1 + d2)/(rho_pre * Vs_pre * cp_pre)
    v5 = (e1 + e2)/(rhow_pre * Vt_pre * cpw_pre)


    return np.array([v1, v2, v3, v4, v5])

Vsup_0 = [kelvin(380), kelvin(383), kelvin(317), kelvin(296), kelvin(230)]                
Sol_sup = solve_ivp(dSGSdt, t_span=(min(time_sup), max(time_sup)), 
                    y0 = Vsup_0, t_eval = time_sup, 
                    method = 'Radau', rtol = 1e-5)

# Graficos

for i in range(n):
    plt.figure(1)
    plt.title('Temperatura de salida Therminol VP - 1')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Temperatura [C]')
    plt.plot(time, temp_htf[m_htf[i]], label = m_htf[i])
    plt.legend()

    plt.figure(2)
    plt.title('Temperatura en el Tubo Abosrbedor')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Temperatura [C]')
    plt.plot(time, temp_abs[m_htf[i]], label = m_htf[i])
    plt.legend()

    plt.figure(3)
    plt.title('Temperatura en el Tubo de Vidrio')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Temperatura [C]')
    plt.plot(time, temp_g[m_htf[i]], label = m_htf[i])
    plt.legend()

for i in range(len(m_ms)):
    plt.figure(4)
    plt.title('Temperatura de salida Therminol VP - 1 en hx1')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Temperatura [C]')
    plt.plot(time_tes, temp_htf_hx1[m_ms[i]], label = m_ms[i])
    plt.legend()

    plt.figure(5)
    plt.title('Temperatura de salida Molten Salt en hx1')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Temperatura [C]')
    plt.plot(time_tes, temp_ms_hx1[m_ms[i]], label = m_ms[i])
    plt.legend()

    plt.figure(6)
    plt.title('Altura del estanque')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Altura [m]')
    plt.plot(time_tes, height[m_ms[i]], label = m_ms[i])
    plt.legend()

plt.figure(7)
plt.title('Temperatura de salida del HTF en superheater')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [C]')
plt.plot(time_sup, celsius(Sol_sup.y[0]), label = 'temp salida')
plt.plot(time_sup, fit_sup(time_sup), label = 'temp entrada')
plt.legend()

plt.figure(8)
plt.title('Temperatura de salida del agua en superheater')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [C]')
plt.plot(time_sup, celsius(Sol_sup.y[1]))

plt.figure(9)
plt.title('Temperatura del HTF en evaporador')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [C]')
plt.plot(time_sup, celsius(Sol_sup.y[2]), label = 'temp salida')
plt.plot(time_sup, celsius(Sol_sup.y[0]), label = 'temp entrada')
plt.legend()

plt.figure(10)
plt.title('Temperatura del HTF en preheater')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [C]')
plt.plot(time_sup, celsius(Sol_sup.y[2]), label = 'temp entrada')
plt.plot(time_sup, celsius(Sol_sup.y[3]), label = 'temp salida')
plt.legend()

plt.figure(11)
plt.title('Temperatura del agua en preheater')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [C]')
plt.plot(time_sup, celsius(Sol_sup.y[4]))

plt.show()





