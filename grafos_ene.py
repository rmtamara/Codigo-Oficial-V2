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

def celsius(t):
    return t - 273.15

# Variables guardadas

caudal = open('Enero/caudal_ene.pkl', 'rb')
m_htf = pickle.load(caudal)
tiempo = open('Enero/tiempo_ene.pkl', 'rb')
time = pickle.load(tiempo)
temp_sf = open('Enero/temp_ene.pkl', 'rb')
u = pickle.load(temp_sf)
tiempo2 = open('Enero/tiempo2_ene.pkl', 'rb')
time2 = pickle.load(tiempo2)
temp_sgs = open('Enero/temp_sgs_ene.pkl', 'rb')
u2 = pickle.load(temp_sgs)
power = open('Enero/power_ene.pkl', 'rb')
Qpb = pickle.load(power)
work = open('Enero/work_ene.pkl', 'rb')
Wt = pickle.load(work)
msup = open('Enero/q_sup_ene.pkl', 'rb')
q_sup = pickle.load(msup)

# GRAFICO CAUDAL TEMPERATURA 
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Tiempo [hr]', fontsize = 30)
ax1.set_ylabel('Caudal [kg/s]', color=color, fontsize = 30)
ax1.plot(time[166501:322019]/3600, m_htf[166501:322019], color=color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(14, 19.5)
ax1.set_xlim(9.23, 18)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Temperatura [C]', color=color, fontsize = 30)
ax2.plot(time[166501:322019]/3600, celsius(u[166501:322019, 0]), color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax2.set_ylim(0, 500)
ax2.set_xlim(9.23, 18)

fig.tight_layout()
plt.show()


# GRAFICO RADIACION TEMPERATURA
fig, ax3 = plt.subplots()

color = 'tab:red'
ax3.set_xlabel('Tiempo [hr]')
ax3.set_ylabel('DNI [W/m2]', color=color, fontsize = 30)
ax3.plot(time[166501:322019]/3600, fit_rad(time[166501:322019]/3600), color=color, linewidth = 2)
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_xlim(9.23, 18)
ax3.set_ylim(846, 1025)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax4 = ax3.twinx()

color = 'tab:blue'
ax4.set_ylabel('Temperatura [C]', color=color, fontsize = 30)  
ax4.plot(time[166501:322019]/3600, celsius(u[166501:322019,0]), color=color, linewidth = 2)
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_xlim(9.23, 18)
ax4.set_ylim(0, 500)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)



fig.tight_layout()  
plt.show()

# GRAFICO POTENCIA TERMICA Y TRABAJO
fig, ax5 = plt.subplots()

color = 'tab:red'
ax5.set_xlabel('Tiempo [hr]', fontsize = 30)
ax5.set_ylabel('Trabajo turbina [MW]', color=color, fontsize = 30)
ax5.plot(time2[1887:157585]/3600, Wt[1887:157585]/10**6, color=color, linewidth = 2)
ax5.tick_params(axis='y', labelcolor=color)
#ax5_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax5_little.plot(time2[61724:62863], Wt[61725:62864]/10**6)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax5.set_xlim(9.241, 18)
ax5.set_ylim(1.054, 1.062)

ax6 = ax5.twinx()

color = 'tab:blue'
ax6.set_ylabel('Potencia Térmica [MW]', color=color, fontsize = 30)  
ax6.plot(time2[1887:157585]/3600, Qpb[1887:157585]/10**6, color=color, linewidth = 2)
ax6.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()  
plt.show()

# Grafico generacion electrica
eta_gen = 0.99
eta_tur = 0.6

fig, ax7 = plt.subplots()
ax7.set_xlabel('Tiempo [hr]', fontsize = 30)
ax7.set_ylabel('Potencia Eléctrica [MW]', fontsize = 30)
ax7.plot(time2[1887:157585]/3600, eta_gen * Wt[1887:157585]/10**6, linewidth = 3)
ax7.set_ylim(1.044, 1.052)
ax7.set_xlim(9.241, 18)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
#ax7_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax7_little.plot(time2[61724:62863], eta_gen * eta_tur * Wt[61725:62864]/10**6)
plt.show()

# Grafico Radiacion

fig, ax8 = plt.subplots()
ax8.set_xlabel('Tiempo [hr]')
ax8.set_ylabel('Irradiancia [W/m2]')
ax8.plot(time/3600, fit_rad(time/3600))
plt.show()

# Grafico caudal temperatura
fig, ax9 = plt.subplots()
color = 'tab:red'
ax9.set_xlabel('Tiempo [hr]', fontsize = 30)
ax9.set_ylabel('Caudal [kg/s]', color=color, fontsize = 30)
ax9.plot(time/3600, m_htf, color=color, linewidth = 2)
ax9.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax10 = ax9.twinx()

color = 'tab:blue'
ax10.set_ylabel('Temperatura [C]', color=color, fontsize = 30)
ax10.plot(time/3600, celsius(u[:, 0]), color=color, linewidth = 2)
ax10.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()
plt.show()

# Grafico radiacion temperatura
fig, ax11 = plt.subplots()

color = 'tab:red'
ax11.set_xlabel('Tiempo [hr]')
ax11.set_ylabel('DNI [W/m2]', color=color, fontsize = 30)
ax11.plot(time/3600, fit_rad(time/3600), color=color, linewidth = 2)
ax11.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax12 = ax11.twinx()

color = 'tab:blue'
ax12.set_ylabel('Temperatura [C]', color=color, fontsize = 30)  
ax12.plot(time/3600, celsius(u[:,0]), color=color, linewidth = 2)
ax12.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

fig.tight_layout()  
plt.show()

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

fluid = ('Therminol', 'Water')
bar = dict()
for i in range(len(fluid)):
    bar[fluid[i]] = []
for i in range(len(hr)):
    bar['Therminol'].append(htf[hr[i]]) 
    bar['Water'].append(w[hr[i]]) 

fig, ax13 = plt.subplots()
ax13.set_xlabel('Tiempo [hr]')
ax13.set_ylabel('Irradiancia [W/m2]')
ax13.plot(time[166501:322019]/3600, m_htf[166501:322019], label = 'caudal aceite')
ax13.plot(time2, q_sup, label = 'caudal de vapor de agua')
plt.show()




