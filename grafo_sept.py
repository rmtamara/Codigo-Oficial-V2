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
base_mes = pd.read_excel(file, sheet_name='Hoja7')
base_mes_20 = pd.read_excel(file, sheet_name='Hoja8')
base_mes_30 = pd.read_excel(file, sheet_name='Hoja11')
rad = base_mes['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
rad_20 = base_mes_20['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
rad_30 = base_mes_30['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
temp = base_mes['Temperatura'] + 273.15
dia = base_mes['Dia']
mes = base_mes['Mes']
hora = base_mes['hora']
minuto = base_mes['minuto']
hora_minuto = base_mes['Hora total']
hora_minuto_20 = base_mes_20['Hora total']
hora_minuto_30 = base_mes_30['Hora total']
v = base_mes['Velocidad']

fit_rad = CubicSpline(hora_minuto, rad)
fit_rad_20 = CubicSpline(hora_minuto_20, rad_20)
fit_rad_30 = CubicSpline(hora_minuto_30, rad_30)
fit_temp = CubicSpline(hora_minuto, temp)
fit_vel = CubicSpline(hora_minuto, v)

# Variables guardadas

caudal = open('Septiembre/caudal_sept2.pkl', 'rb')
m_htf = pickle.load(caudal)
tiempo = open('Septiembre/tiempo_sept2.pkl', 'rb')
time = pickle.load(tiempo)
temp_sf = open('Septiembre/temp_sept2.pkl', 'rb')
u = pickle.load(temp_sf)
tiempo2 = open('Septiembre/tiempo2_sept2.pkl', 'rb')
time2 = pickle.load(tiempo2)
temp_sgs = open('Septiembre/temp_sgs_sept2.pkl', 'rb')
u2 = pickle.load(temp_sgs)
power = open('Septiembre/power_sept2.pkl', 'rb')
Qpb = pickle.load(power)
work = open('Septiembre/work_sept2.pkl', 'rb')
Wt = pickle.load(work)
msup = open('Septiembre/q_sup_sept2.pkl', 'rb')
q_sup = pickle.load(msup)

caudal_20 = open('Sept_20min/caudal_sept_20.pkl', 'rb')
m_htf_20 = pickle.load(caudal_20)
#tiempo = open('Abril/tiempo_abr2.pkl', 'rb')
#time = pickle.load(tiempo)
temp_sf_20 = open('Sept_20min/temp_sept_20.pkl', 'rb')
u_20 = pickle.load(temp_sf_20)
tiempo2_20 = open('Sept_20min/tiempo2_sept_20.pkl', 'rb')
time2_20 = pickle.load(tiempo2_20)
temp_sgs_20 = open('Sept_20min/temp_sgs_sept_20.pkl', 'rb')
u2_20 = pickle.load(temp_sgs_20)
power_20 = open('Sept_20min/power_sept_20.pkl', 'rb')
Qpb_20 = pickle.load(power_20)
work_20 = open('Sept_20min/work_sept_20.pkl', 'rb')
Wt_20 = pickle.load(work_20)
msup_20 = open('Sept_20min/q_sup_sept_20.pkl', 'rb')
q_sup_20 = pickle.load(msup_20)

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
#                              Graficos
# ------------------------------------------------------------------------

# GRAFICO CAUDAL TEMPERATURA 

"""
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Tiempo [hr]', fontsize = 50)
ax1.set_ylabel('Caudal [kg/s]', color=color, fontsize = 40)
ax1.plot(time[181621:289619]/3600, m_htf[181621:289619], color=color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Temperatura [C]', color=color, fontsize = 50)
ax2.plot(time[181621:289619]/3600, celsius(u[181621:289619, 0]), color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
ax2.set_xlim(10.09, 17)
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
ax3.plot(time[181621:289619]/3600, fit_rad(time[181621:289619]/3600), color=color, linewidth = 2)
ax3.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)

ax4 = ax3.twinx()

color = 'tab:blue'
ax4.set_ylabel('Temperatura [C]', color=color, fontsize = 50)  
ax4.plot(time[181621:289619]/3600, celsius(u[181621:289619, 0]), color=color, linewidth = 2)
ax4.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
ax4.set_xlim(10.09, 17)
#ax4.set_ylim(0, 500)


fig.tight_layout()  
plt.show()
"""
"""
# GRAFICO POTENCIA TERMICA Y TRABAJO
fig, ax5 = plt.subplots()

color = 'tab:red'
ax5.set_xlabel('Tiempo [hr]', fontsize = 50)
ax5.set_ylabel('Trabajo turbina [MW]', color=color, fontsize = 50)
ax5.plot(time2[301:len(time2)]/3600, Wt[302:len(Wt)]/10**6, color=color, linewidth = 2)
ax5.tick_params(axis='y', labelcolor=color)
#ax5_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax5_little.plot(time2[61724:62863], Wt[61725:62864]/10**6)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
ax5.set_xlim(10.09, 17)
#ax5.set_ylim(1.05, 1.056)


ax6 = ax5.twinx()

color = 'tab:blue'
ax6.set_ylabel('Potencia Térmica [MW]', color=color, fontsize = 50)  
ax6.plot(time2[301:len(time2)]/3600, Qpb[302:len(Qpb)]/10**6, color=color, linewidth = 2)
ax6.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)

fig.tight_layout()  
plt.show()
"""

fig, axs = plt.subplots(nrows= 3, ncols = 1, sharex = 'all', sharey = 'all', figsize = (8, 10))
fig.subplots_adjust(left = 0.18, top = 0.95)
#axs[0].set_xlabel('$Tiempo [Hr]$', fontsize = 30)
#axs[0].set_ylabel('$Irradiancia [Wm^{-2}]$', fontsize = 30)
axs[0].plot(time/3600, fit_rad(time/3600), linewidth = 2, label = '10 min')
axs[1].plot(time/3600, fit_rad_20(time/3600), linewidth = 2, label = '20 min')
axs[2].plot(time/3600, fit_rad_30(time/3600) , linewidth = 2, label = '30 min')
axs[0].set_title('Resolución 10 min', fontsize = 20)
axs[1].set_title('Resolución 20 min', fontsize = 20)
axs[2].set_title('Resolución 30 min', fontsize = 20)
fig.text(0.04, 0.5, '$Irradiancia [W/m^{2}]$', va = 'center', rotation = 'vertical', fontsize = 30)
fig.text(0.5, 0.04, '$Tiempo [h]$', ha = 'center', fontsize = 30)

for ax in axs:
    ax.tick_params(axis= 'both', which = 'major', labelsize = 15)

plt.show()

# Grafico generacion electrica
eta_gen = 0.99
eta_tur = 0.6

fig, ax7 = plt.subplots()
ax7.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax7.set_ylabel('$Potencia Eléctrica [MW]$', fontsize = 30)
ax7.plot(time2/3600, eta_gen * Wt[1:len(Wt)]/10**6, linewidth = 2, label = '10 min')
ax7.plot(time2_20/3600, eta_gen * Wt_20[1:len(Wt_20)]/10**6, linewidth = 2, color = 'red', label = '20 min')
ax7.legend()
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax7.set_xlim(10.24, 17)
#ax7.set_xlim(9.42, 17)
#ax7.set_ylim(1.04, 1.046)
#ax7_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax7_little.plot(time2[61724:62863], eta_gen * eta_tur * Wt[61725:62864]/10**6)
plt.show()

# Grafico Radiacion
"""
fig, ax8 = plt.subplots()
ax8.set_xlabel('Tiempo [hr]')
ax8.set_ylabel('Irradiancia [W/m2]')
ax8.plot(time/3600, fit_rad(time/3600))
plt.show()
"""

# Grafico caudal temperatura
fig, ax9 = plt.subplots()
color = 'tab:red'
ax9.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax9.set_ylabel('$Caudal [kg/s]$', color=color, fontsize = 30)
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

fig, ax18 = plt.subplots()
color = 'tab:red'
ax18.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax18.set_ylabel('$Caudal [kg/s]$', color=color, fontsize = 30)
ax18.plot(time/3600, m_htf_20, color=color, linewidth = 2)
ax18.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax19 = ax18.twinx()

color = 'tab:blue'
ax19.set_ylabel('$Temperatura [°C]$', color=color, fontsize = 30)
ax19.plot(time/3600, celsius(u_20[:, 0]), color=color, linewidth = 2)
ax19.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()
plt.show()

# Grafico radiacion temperatura
fig, ax11 = plt.subplots()

color = 'tab:red'
ax11.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax11.set_ylabel('$DNI [W/m^{2}]$', color=color, fontsize = 30)
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

fig, ax20 = plt.subplots()

color = 'tab:red'
ax20.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax20.set_ylabel('$DNI [W/m^{2}]$', color=color, fontsize = 30)
ax20.plot(time/3600, fit_rad_20(time/3600), color=color, linewidth = 2)
ax20.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax21 = ax20.twinx()

color = 'tab:blue'
ax21.set_ylabel('$Temperatura [°C]$', color=color, fontsize = 30)  
ax21.plot(time/3600, celsius(u_20[:,0]), color=color, linewidth = 2)
ax21.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()  
plt.show()

fig, ax22 = plt.subplots()
ax22.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax22.set_ylabel('$Caudal [kg/s]$', fontsize = 30)
ax22.plot(time[186841:291239]/3600, m_htf_20[186841:291239], linewidth = 2, label = '20 min', color = 'red')
ax22.plot(time[182341:288539]/3600, m_htf[182341:288539], linewidth = 2, label = '10 min')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax22.legend(fontsize = 20)
ax22.set_xlim(10.37, 17)
#ax7.set_ylim(1.04, 1.046)
#ax7_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax7_little.plot(time2[61724:62863], eta_gen * eta_tur * Wt[61725:62864]/10**6)
plt.show()

# Caudal agua

fig, ax23 = plt.subplots()
ax23.set_xlabel('$Tiempo [h]$', fontsize = 30)
ax23.set_ylabel('$Caudal [kg/s]$', fontsize = 30)
ax23.plot(time2_20/3600, q_sup_20, linewidth = 2, label = '20 min', color = 'red')
ax23.plot(time2/3600, q_sup, linewidth = 2, label = '10 min')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax23.legend(fontsize = 20)
ax23.set_xlim(10.24, 17)
plt.show()

fig, ax24 = plt.subplots()
ax24.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax24.set_ylabel('$Temperatura [°C]$', fontsize = 30)
ax24.plot(time2_20/3600, celsius(u2_20[1:len(u2_20), 1]), linewidth = 2, label = '20 min', color = 'red')
ax24.plot(time2/3600, celsius(u2[1:len(u2), 1]), linewidth = 2, label = '10 min')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax24.legend(fontsize = 20)
ax24.set_xlim(10.24, 17)
plt.show()

"""
fig, ax13 = plt.subplots()
ax13.set_xlabel('Tiempo [hr]', fontsize = 50)
ax13.set_ylabel('Caudal [kg/s]', fontsize = 50)
ax13.plot(time[181621:289619]/3600, m_htf[181621:289619], linewidth = 2, label = 'Caudal instantáneo')
ax13.plot(hr, x, '-s', linewidth = 2, label = 'Caudal promedio', color = 'red')
ax13.set_xlim(10.09, 17)
ax13.legend(fontsize = 40)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)

plt.show()
"""
"""
fig, ax14 = plt.subplots()
ax14.set_xlabel('Tiempo [hr]', fontsize = 50)
ax14.set_ylabel('Caudal [kg/s]', fontsize = 50)
ax14.plot(time2[301:len(time2)]/3600, q_sup[301:len(q_sup)], linewidth = 2, label = 'Caudal instantáneo')
ax14.plot(hr, y, '-s', linewidth = 2, label = 'Caudal promedio', color = 'red')
ax14.set_xlim(10.09, 17)
ax14.legend(fontsize = 40)
plt.yticks(fontsize = 40)
plt.xticks(fontsize = 40)
plt.show()
"""
# Promedio centrados caudal agua

ventana = 18001
promedios = np.array(promedio_movil_centrado(q_sup, time2, ventana)[0])
tiempos = np.array(promedio_movil_centrado(q_sup, time2, ventana)[1])

"""
fig, ax14 = plt.subplots()
ax14.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax14.set_ylabel('$Caudal [kgs^{-1}$]', fontsize = 30)
ax14.plot(time2[301:len(time2)]/3600, q_sup[301:len(q_sup)], linewidth = 2, label = 'Caudal instantáneo')
ax14.plot(tiempos[0:-1:17500]/3600, promedios[0:-1:17500], '-s', linewidth = 2, label = 'Caudal promedio centrado', color = 'red')
ax14.set_xlim(10.09, 17)
ax14.legend(loc = 'lower right', fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()
"""
ventana2 = 18001
time3 = time[181621:289619]
m_htf_sgs = m_htf[181621:289619]
promedios_htf = np.array(promedio_movil_centrado(m_htf_sgs, time3, ventana2)[0])
tiempos_htf = np.array(promedio_movil_centrado(m_htf_sgs, time3, ventana2)[1])
"""
fig, ax13 = plt.subplots()
ax13.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax13.set_ylabel('$Caudal [kgs^{-1}]$', fontsize = 30)
ax13.plot(time[181621:289619]/3600, m_htf[181621:289619], linewidth = 2, label = 'Caudal instantáneo')
ax13.plot(tiempos_htf[0:-1:17500]/3600, promedios_htf[0:-1:17500], '-s', linewidth = 2, label = 'Caudal promedio centrado', color = 'red')
ax13.set_xlim(10.09, 17)
ax13.legend(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

plt.show()
"""
# Promedios centrados para tablas

caudal_agua = promedios[0:-1:17500]
caudal_aceite = promedios_htf[0:-1:17500]
temp_htf_sgs = celsius(u2[1:len(u2),:])
temp_sup_aceite = promedio_movil_centrado(temp_htf_sgs[:, 0], time2, ventana)[0][0:-1:17500]
temp_evp_aceite = promedio_movil_centrado(temp_htf_sgs[:, 2], time2, ventana)[0][0:-1:17500]
temp_pre_aceite = promedio_movil_centrado(temp_htf_sgs[:, 3], time2, ventana)[0][0:-1:17500]
temp_sup_agua = promedio_movil_centrado(temp_htf_sgs[:, 1], time2, ventana)[0][0:-1:17500]
temp_pre_agua = promedio_movil_centrado(temp_htf_sgs[:, 4], time2, ventana)[0][0:-1:17500]