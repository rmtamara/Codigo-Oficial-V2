import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
plt.close('all')

# ------------------------------------------------------------------------
#                            Base de Datos
# ------------------------------------------------------------------------

file = 'base.xlsx'
base_mes = pd.read_excel(file, sheet_name='Sheet5')
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

# Variables guardadas

caudal = open('Abril/caudal_abr2.pkl', 'rb')
m_htf = pickle.load(caudal)
tiempo = open('Abril/tiempo_abr2.pkl', 'rb')
time = pickle.load(tiempo)
temp_sf = open('Abril/temp_abr2.pkl', 'rb')
u = pickle.load(temp_sf)
tiempo2 = open('Abril/tiempo2_abr2.pkl', 'rb')
time2 = pickle.load(tiempo2)
temp_sgs = open('Abril/temp_sgs_abr2.pkl', 'rb')
u2 = pickle.load(temp_sgs)
power = open('Abril/power_abr2.pkl', 'rb')
Qpb = pickle.load(power)
work = open('Abril/work_abr2.pkl', 'rb')
Wt = pickle.load(work)
msup = open('Abril/q_sup_abr2.pkl', 'rb')
q_sup = pickle.load(msup)

# ------------------------------------------------------------------------
#                    Funciones auxiliares
# ------------------------------------------------------------------------
def celsius(t):
    return t - 273.15

# Promedios centrados

def promedio_movil_centrado(datos, tiempo, ventana):
    mediana_i = int((ventana + 1)/2) - 1
    promedios_centrados = []
    tiempo_centrado = []
    mitades = int((ventana-1)/2)
    
    for i in range(mitades, len(datos)-mitades):
        inicio = i - mitades
        fin = i + mitades + 1
        ventana_actual = datos[inicio:fin]
        promedio_actual = np.mean(ventana_actual)
        promedios_centrados.append(promedio_actual)
        tiempo_actual = tiempo[inicio:fin]
        tiempo_centrado.append(tiempo_actual[mediana_i])

    
    return (promedios_centrados, tiempo_centrado)

# Promedio por hora de caudales

first = int(time2[0]/3600)
last = int(time2[-1]/3600)
steps = last - first + 1
hr = np.linspace(first, last, steps)
htf = dict()
w = dict()

in_1 = np.where((time/3600>time2[0]/3600)&(time/3600<time2[-1]/3600))[0][0]
in_2 = np.where((time/3600>time2[0]/3600)&(time/3600<time2[-1]/3600))[0][-1]

for i in range(len(hr)):
    htf[hr[i]] = []
    w[hr[i]] = []

for i in range(len(time2)):
    h = int(time2[i]/3600)
    w[h].append(q_sup[i])

for i in range(in_1, in_2+1):
    h = int(time[i]/3600)
    htf[h].append(m_htf[i])

for i in range(len(hr)):
    htf[hr[i]] = np.mean(htf[hr[i]])
    w[hr[i]] = np.mean(w[hr[i]])

x = list(htf.values())
y = list(w.values())
# ------------------------------------------------------------------------
#                               Graficos
# ------------------------------------------------------------------------

"""
# GRAFICO CAUDAL TEMPERATURA 
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Tiempo [hr]', fontsize = 50)
ax1.set_ylabel('Caudal [kg/s]', color=color, fontsize = 50)
ax1.plot(time[167581:316979]/3600, m_htf[167581:316979], color=color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
ax1.set_xlim(9.31, 18)
#ax1.set_ylim(10, 15)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Temperatura [C]', color=color, fontsize = 50)
ax2.plot(time[167581:316979]/3600, celsius(u[167581:316979, 0]), color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
ax2.set_xlim(9.32, 17.61)
ax2.set_ylim(0, 500)
ax2.set_xlim(9.31, 18)
#ax2.set_ylim(0, 500)

fig.tight_layout()
plt.show()

"""
# GRAFICO RADIACION TEMPERATURA

"""
fig, ax3 = plt.subplots()

color = 'tab:red'
ax3.set_xlabel('Tiempo [hr]', fontsize = 50)
ax3.set_ylabel('DNI [W/m2]', color=color, fontsize = 50)
ax3.plot(time[167581:316979]/3600, fit_rad(time[167581:316979]/3600), color=color, linewidth = 2)
ax3.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax4 = ax3.twinx()

color = 'tab:blue'
ax4.set_ylabel('Temperatura [C]', color=color, fontsize = 50)  
ax4.plot(time[167581:316979]/3600, celsius(u[167581:316979, 0]), color=color, linewidth = 2)
ax4.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax4.set_xlim(9.32, 17.61)
ax4.set_ylim(0, 500)

#ax4.set_xlim(9.42, 17)
#ax4.set_ylim(0, 500)


fig.tight_layout()  
plt.show()
"""

# GRAFICO POTENCIA TERMICA Y TRABAJO

"""
fig, ax5 = plt.subplots()

color = 'tab:red'
ax5.set_xlabel('Tiempo [hr]', fontsize = 50)
ax5.set_ylabel('Trabajo turbina [MW]', color=color, fontsize = 50)
ax5.plot(time2[4039:len(time2)]/3600, Wt[4040:len(Wt)]/10**6, color=color, linewidth = 2)
ax5.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
ax5.set_xlim(9.31, 18)

ax6 = ax5.twinx()

color = 'tab:blue'
ax6.set_ylabel('Potencia Térmica [MW]', color=color, fontsize = 50)  
ax6.plot(time2[4039:len(time2)]/3600, Qpb[4040:len(Qpb)]/10**6, color=color, linewidth = 2)
ax6.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)

fig.tight_layout()  
plt.show()
"""

# Grafico generacion electrica
eta_gen = 0.99
eta_tur = 0.6

fig, ax7 = plt.subplots()
ax7.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax7.set_ylabel('$Potencia Eléctrica [MW]$', fontsize = 30)
ax7.plot(time2[4039:len(time2)]/3600, eta_gen * Wt[4040:len(Wt)]/10**6, linewidth = 2)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax7.set_xlim(9.32, 17.61)
#ax7.set_xlim(9.42, 17)
#ax7.set_ylim(1.04, 1.046)
#ax7_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax7_little.plot(time2[61724:62863], eta_gen * eta_tur * Wt[61725:62864]/10**6)
plt.show()

# Grafico Radiacion

fig, ax8 = plt.subplots()
ax8.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax8.set_ylabel('$Irradiancia [Wm^{-2}]$', fontsize = 30)
ax8.plot(time/3600, fit_rad(time/3600), linewidth = 2)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

# Grafico caudal temperatura
fig, ax9 = plt.subplots()
color = 'tab:red'
ax9.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax9.set_ylabel('$Caudal [kgs^{-1}]$', color=color, fontsize = 30)
ax9.plot(time/3600, m_htf, color=color, linewidth = 2)
ax9.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax10 = ax9.twinx()

color = 'tab:blue'
ax10.set_ylabel('$Temperatura [°C]$', color=color, fontsize = 30)
ax10.plot(time/3600, celsius(u[:, 0]), color=color, linewidth = 2)
ax10.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()
plt.show()

# Grafico radiacion temperatura
fig, ax11 = plt.subplots()

color = 'tab:red'
ax11.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax11.set_ylabel('$DNI [Wm^{-2}]$', color=color, fontsize = 30)
ax11.plot(time/3600, fit_rad(time/3600), color=color, linewidth = 2)
ax11.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax12 = ax11.twinx()

color = 'tab:blue'
ax12.set_ylabel('$Temperatura [°C]$', color=color, fontsize = 30)  
ax12.plot(time/3600, celsius(u[:,0]), color=color, linewidth = 2)
ax12.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()  
plt.show()




"""
fig, ax13 = plt.subplots()
ax13.set_xlabel('Tiempo [hr]', fontsize = 50)
ax13.set_ylabel('Caudal [kg/s]', fontsize = 50)
ax13.plot(time[167581:316979]/3600, m_htf[167581:316979], linewidth = 2, label = 'Caudal instantáneo')
ax13.plot(hr, x, '-s', linewidth = 2, label = 'Caudal promedio', color = 'red')
ax13.set_xlim(9.31, 18)
ax13.legend(loc = 'lower right', fontsize = 40)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)

plt.show()
"""
"""
fig, ax14 = plt.subplots()
ax14.set_xlabel('Tiempo [hr]', fontsize = 50)
ax14.set_ylabel('Caudal [kg/s]', fontsize = 50)
ax14.plot(time2[4039:len(time2)]/3600, q_sup[4039:len(q_sup)], linewidth = 2, label = 'Caudal instantáneo')
ax14.plot(hr, y, '-s', linewidth = 2, label = 'Caudal promedio', color = 'red')
ax14.set_xlim(9.31, 18)
ax14.legend(loc = 'lower right', fontsize = 40)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
plt.show()
"""
# Promedio centrados caudal agua

ventana = 4501
promedios = np.array(promedio_movil_centrado(q_sup, time2, ventana)[0])
tiempos = np.array(promedio_movil_centrado(q_sup, time2, ventana)[1])

fig, ax14 = plt.subplots()
ax14.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax14.set_ylabel('$Caudal [kgs^{-1}$]', fontsize = 30)
ax14.plot(time2[4039:len(time2)]/3600, q_sup[4039:len(q_sup)], linewidth = 2, label = 'Caudal instantáneo')
ax14.plot(tiempos[0:-1:18000]/3600, promedios[0:-1:18000], '-s', linewidth = 2, label = 'Caudal promedio centrado', color = 'red')
ax14.set_xlim(9.31, 18)
ax14.legend(loc = 'lower right', fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

# Promedio centrado aceite

ventana2 = 3001
time3 = time[167581:316979]
m_htf_sgs = m_htf[167581:316979]
promedios_htf = np.array(promedio_movil_centrado(m_htf_sgs, time3, ventana2)[0])
tiempos_htf = np.array(promedio_movil_centrado(m_htf_sgs, time3, ventana2)[1])

fig, ax16 = plt.subplots()
ax16.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax16.set_ylabel('$Caudal [kg s^{-1}]$', fontsize = 30)
ax16.plot(time[167581:316979]/3600, m_htf[167581:316979], linewidth = 2, label = 'Caudal instantáneo')
ax16.plot(tiempos_htf[0:-1:18000]/3600, promedios_htf[0:-1:18000], '-s', linewidth = 2, label = 'Caudal promedio centrado', color = 'red')
ax16.set_xlim(9.31, 18)
ax16.legend(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

plt.show()

# Promedios centrados para tablas

caudal_agua = promedios[0:-1:18000]
caudal_aceite = promedios_htf[0:-1:18000]
temp_htf_sgs = celsius(u2[1:len(u2),:])
temp_sup_aceite = promedio_movil_centrado(temp_htf_sgs[:, 0], time2, ventana)[0][0:-1:18000]
temp_evp_aceite = promedio_movil_centrado(temp_htf_sgs[:, 2], time2, ventana)[0][0:-1:18000]
temp_pre_aceite = promedio_movil_centrado(temp_htf_sgs[:, 3], time2, ventana)[0][0:-1:18000]
temp_sup_agua = promedio_movil_centrado(temp_htf_sgs[:, 1], time2, ventana)[0][0:-1:18000]
temp_pre_agua = promedio_movil_centrado(temp_htf_sgs[:, 4], time2, ventana)[0][0:-1:18000]