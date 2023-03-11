import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

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

Lsup = 4 * 4
Dsup = 0.016
Dsup_i = 0.012
Asup_o = np.pi * Dsup * Lsup
Asup_i = np.pi * Dsup_i * Lsup
Vt_sup = 13.41
Vs_sup = 11.46

# ---------------------------
#         Evaporador
# ---------------------------

Levp = 4
Devp = 0.016
Devp_i = 0.012
Vt_evp = 0.2
Vs_evp = 1.86

tevap = 509.188

# ----------------------------------------
#               Preheater
# ----------------------------------------

Lpre = 4
Dpre = 0.016
Dpre_i = 0.012
Vt_pre = 0.063
Vs_pre = 0.26
Apre = 38.158

# ------------------------------
#       Heax exchanger 1
# ------------------------------

Lhx1 = 4
Dhx1 = 0.016
Dhx1_i = 0.012
Vt_hx1 = 0.2
Vs_hx1 = 1.86


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

def dif_temp_htf(m, tin, tout):
    """
    m: caudal
    tin: temperatura de entrada
    tout: temperatura de salida
    """
    cp = fit_cp(tout)
    output = m * cp * (tin - tout)
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

def Pr(t):
    cp = fit_cp(t)
    mu = fit_mu(t)
    k = fit_k(t)
    pr = cp * mu / k
    return pr

# Superheater

def dif_temp_w(m, t, t2):
    p = 3.1 * 10**6
    cp = CP.PropsSI('C', 'T', t2, 'P', p, 'Water')
    output = m * cp * (t - t2)
    return output

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

# Evaporador

def Re_htf_evp(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Devp_i * mu)
    return re

def Re_w_evp(m, t):
    p = 3.12 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'P', p, 'Water')
    re = m/(mu * Lsup)
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

# Preheater

def dif_temp_w_pre(m, t, t2):
    p = 3.21 * 10**6
    cp = CP.PropsSI('C', 'T', t2, 'P', p, 'Water')
    output = m * cp * (t - t2)
    return output

def Re_w_pre(m, t):
    p = 3.21 * 10**6
    mu = CP.PropsSI('V', 'T', t, 'Q', 0, 'Water')
    re = m/(mu * Lpre)
    return re

def Re_htf_pre(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dpre_i * mu)
    return re

def Pr_w_pre(t):
    p = 3.21 * 10**6
    cp = CP.PropsSI('C', 'T', t, 'Q', 0, 'Water')
    mu = CP.PropsSI('V', 'T', t, 'Q', 0, 'Water')
    k = CP.PropsSI('L', 'T', t, 'Q', 0, 'Water') 
    pr = cp * mu/k
    return pr

def Nu_tubes_pre(m, t):
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

# Intercambiador heat exchanger

def Re_htf_hx1(m, t):
    mu = fit_mu(t)
    re = 4 * m/(np.pi * Dhx1_i * mu)
    return re

def Re_ms_hx1(m, t):
    mu = mu_ms(t)
    re = m/(mu * Lhx1)
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
# ------------------------------------------------------------------------
#                              SOLAR FIELD
# ------------------------------------------------------------------------

T_in = kelvin(296)
max_rad = np.linspace(700, 1000, 4)
vel = 1.39
tamb = 18.15
m_htf = 15
n = len(max_rad)
tout = np.zeros(n)
Tg = kelvin(200)


for i in range(n):

    DNI = max_rad[i]

    def temp(x):
        Thtf = x

        Qhtf = - dif_temp_htf(m_htf, T_in, Thtf)
        Qg_rad = q_rad_g(Thtf, tamb)
        Qg_conv = conv_g(vel, Thtf, tamb)

        f1 = (DNI * A * eta1 * eta2 * eta3 * eta4 * 
              eta5 * eta6 * eta7 * eta8 - Qhtf - Qg_rad - Qg_conv)

        return f1

    root = newton(temp, kelvin(400))
    tout[i] = celsius(root)


# ------------------------------------------------------------------------
#                            POWER BLOCK
# ------------------------------------------------------------------------

# -----------------------------
#             Agua
#------------------------------

# Punto 1: Entrada a la Turbina

T1 = kelvin(375)
P1 = 2.35 * 10**6
s1 = CP.PropsSI('S', 'T', T1, 'P', P1, 'Water')
h1 = CP.PropsSI('H', 'T', T1, 'P', P1, 'Water')

# Punto 2: Entrada al Condensador

s2 = s1
P2 = 0.0067 * 10**6
T2 = CP.PropsSI('T', 'S', s2, 'P', P2, 'Water')
h2 = CP.PropsSI('H', 'S', s2, 'P', P2, 'Water')

# Punto 3: Entrada a la bomba

P3 = P2
T3 = T2
h3 = CP.PropsSI('H', 'P', P3, 'Q', 0, 'Water')
s3 = CP.PropsSI('S', 'P', P3, 'Q', 0, 'Water')
d3 = CP.PropsSI('D', 'P', P3, 'Q', 0, 'Water')
v3 = 1/d3

# Punto 4: Entrada al Preheater

T4 = kelvin(104)
P4 = 3.21 * 10**6

# Punto 5: Entrada al Evaporador

P5 = 3.12 * 10**6
T5 = CP.PropsSI('T', 'P', P5, 'Q', 0, 'Water')

# Punto 6: Entrada al superheater

T6 = T5
P6 = P5

# Salida del superheater

T7 = kelvin(380)
P8 = 3.1 * 10**6

# ------------------------
#         Aceite
# ------------------------

m_w = np.linspace(0.8, 2, 7)
tout_sup = np.zeros((n, len(m_w)))
tout_evp = np.zeros((n, len(m_w)))
tout_pre = np.zeros((n, len(m_w)))

for i in range(len(m_w)):
    for j in range(n):
        def temp_pb(x):
            Tsup_htf, Tevp_htf, Tpre_htf = x

            # Superheater
            Qsup_htf = dif_temp_htf(20, kelvin(tout[j]), Tsup_htf)
            Qsup_w = dif_temp_w(m_w[i], T6, T7)
            f1 = Qsup_htf + Qsup_w

            #Evaporador

            hg = CP.PropsSI('H', 'T', T5, 'Q', 1, 'Water')
            hl = CP.PropsSI('H', 'T', T5, 'Q', 0, 'Water')
            Qevp_w = m_w[i] * (hl - hg)
            Qevp_htf = dif_temp_htf(20, Tsup_htf, Tevp_htf)
            f2 = Qevp_htf + Qevp_w

            # Preheater

            Qpre_w = dif_temp_w_pre(m_w[i], T4, T5)
            Qpre_htf = dif_temp_htf(20, Tevp_htf, Tpre_htf)
            f3 = Qpre_htf + Qpre_w

            return np.array([f1, f2, f3])
        
        root_pb = newton(temp_pb, [kelvin(380), kelvin(317), kelvin(296)], rtol = 1e-3)
        tout_sup[j, i] = celsius(root_pb[0])
        tout_evp[j, i] = celsius(root_pb[1])
        tout_pre[j, i] = celsius(root_pb[2])

def Asup(mhtf, mw, tin, tout, tinw, toutw):
    cp = fit_cp(tout)
    cpw = CP.PropsSI('C', 'T', toutw, 'P', P8, 'Water')
    kw = CP.PropsSI('L', 'T', toutw, 'P', P8, 'Water') 
    k = fit_k(tout)
    alphao = Nu_shell_sup(mw, toutw) * kw/Dsup
    alphai = Nu_tubes_sup(mhtf, tout) * k/Dsup_i
    U = coef_global(alphao, alphai)
    Qhtf = mhtf * cp * (tin - tout)
    Qw = mw * cpw * (tinw - toutw)
    A = np.abs(Qw)/(U * (np.abs(tout - toutw)))
    return (Qhtf, Qw, A, U)

def Aevp(mhtf, mw, tin, tout, tinw, toutw):
    cp = fit_cp(tout)
    hg = CP.PropsSI('H', 'T', toutw, 'Q', 1, 'Water')
    hl = CP.PropsSI('H', 'T', tinw, 'Q', 0, 'Water')
    kw = CP.PropsSI('L', 'T', tevap, 'P', P5, 'Water') 
    k = fit_k(tout)
    alphao = Nu_shell_evp(mw, tevap) * kw/Devp
    alphai = Nu_tubes_evp(mhtf, tout) * k/Devp_i
    U = coef_global(alphao, alphai)
    Qhtf = mhtf * cp * (tin - tout)
    Qw = mw * (hl - hg)
    A = np.abs(Qw)/(U * (np.abs(tout - toutw)))
    return (Qhtf, Qw, A, U)

def Apre(mhtf, mw, tin, tout, tinw, toutw):
    cp = fit_cp(tout)
    cpw = CP.PropsSI('C', 'T', toutw, 'P', P4, 'Water')
    kw = CP.PropsSI('L', 'T', toutw, 'P', P4, 'Water') 
    k = fit_k(tout)
    alphao = Nu_shell_pre(mw, toutw) * kw/Dpre
    alphai = Nu_tubes_pre(mhtf, tout) * k/Dpre_i
    U = coef_global(alphao, alphai)
    Qhtf = mhtf * cp * (tin - tout)
    Qw = mw * cpw * (tinw - toutw)
    A = np.abs(Qw)/(U * (np.abs(tout - toutw)))
    return (Qhtf, Qw, A, U)

sup = Asup(20, m_w[4], kelvin(tout[2]), kelvin(tout_sup[2,4]),T6, T7)
evp = Aevp(20, m_w[4], kelvin(tout_sup[2,4]), kelvin(tout_evp[2,4]), T5, T6)
pre = Apre(20, m_w[4], kelvin(tout_evp[2,4]), kelvin(tout_pre[2,4]), T4, T5)

# ------------------------------------------------------------------------
#                     Thermal energy storage
# ------------------------------------------------------------------------

m_vp = np.linspace(1, 5, 5)
m_ms = 1
tout_ms = np.zeros(len(m_vp))
tout_vp = np.zeros(len(m_vp))

def temp_ms(x, m_htf, m_ms):
    T_htf, T_ms = x
 
    cp = fit_cp(T_htf)
    cpms = cp_ms(T_ms)
    a = m_htf * cp * (kelvin(tout[3]) - T_htf)
    b = m_ms * cpms * (kelvin(296) - T_ms)
    f = a + b
    return f

for i in range(len(m_vp)):

    dot_m = m_vp[i]
    tout_vp[i] = celsius(newton(temp_ms, [kelvin(370), kelvin(390)], args = (dot_m, m_ms))[0])
    tout_ms[i] = celsius(newton(temp_ms, [kelvin(370), kelvin(390)], args = (dot_m, m_ms))[1])

def Ams(m_htf, m_ms, tin, tout, tinms, toutms):
    cp = fit_cp(tout)
    cpms = cp_ms(toutms)
    kms = k_ms(toutms) 
    k = fit_k(tout)
    alphao = Nu_shell_hx1(m_ms, toutms) * kms/Dhx1
    alphai = Nu_tubes_hx1(m_htf, tout) * k/Dhx1_i
    U = coef_global(alphao, alphai)
    Qhtf = m_htf * cp * (tin - tout)
    Qw = m_ms * cpms * (tinms - toutms)
    A = np.abs(Qw)/(U * (np.abs(tout - toutms)))
    return (Qhtf, Qw, A, U)

Ahx1 = Ams(2, 1, kelvin(tout[3]), kelvin(tout_vp[1]), kelvin(296), kelvin(tout_ms[1]))

def temp_htf(x, m_htf, m_ms):
    T_htf, T_ms = x
    cp = fit_cp(T_htf)
    cpms = cp_ms(T_ms)
    a = m_htf * cp * (kelvin(350) - T_htf)
    b = m_ms * cpms * (kelvin(380) - T_ms)
    f = a + b
    return f

molten_salt = np.linspace(10, 50, 51)
to_ms = np.zeros(len(molten_salt))
to_htf = np.zeros(len(molten_salt))

for i in range(len(molten_salt)):
    m = molten_salt[i]
    to_htf[i] = celsius(newton(temp_htf, [kelvin(380), kelvin(370)], args = (8, m))[0])
    to_ms[i] = celsius(newton(temp_htf, [kelvin(380), kelvin(370)], args = (8, m))[1])

Ahx1_2 = Ams(8, 15, kelvin(370), kelvin(to_htf[10]), kelvin(380), kelvin(to_ms[10]))


















 






    





    


    




