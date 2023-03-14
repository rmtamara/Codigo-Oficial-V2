import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd

# ------------------------------------------------------------------------
#                            Base de Datos
# ------------------------------------------------------------------------

file = 'base.xlsx'
base_mes = pd.read_excel(file, sheet_name='Hoja5')
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

caudal = open('Marzo/caudal_mar.pkl', 'rb')
m_htf = pickle.load(caudal)
tiempo = open('Marzo/tiempo_mar.pkl', 'rb')
time = pickle.load(tiempo)
temp_sf = open('Marzo/temp_mar.pkl', 'rb')
u = pickle.load(temp_sf)
tiempo2 = open('Marzo/tiempo2_mar.pkl', 'rb')
time2 = pickle.load(tiempo2)
temp_sgs = open('Marzo/temp_sgs_mar.pkl', 'rb')
u2 = pickle.load(temp_sgs)
power = open('Marzo/power_mar.pkl', 'rb')
Qpb = pickle.load(power)
work = open('Marzo/work_mar.pkl', 'rb')
Wt = pickle.load(work)
msup = open('Marzo/q_sup_mar.pkl', 'rb')
q_sup = pickle.load(msup)

# GRAFICO CAUDAL TEMPERATURA 
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Tiempo [hr]', fontsize = 30)
ax1.set_ylabel('Caudal [kg/s]', color=color, fontsize = 30)
ax1.plot(time[169741:302398]/3600, m_htf[169741:302398], color=color, linewidth = 2)
ax1.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Temperatura [C]', color=color, fontsize = 30)
ax2.plot(time[169741:302398]/3600, celsius(u[169741:302398, 0]), color=color, linewidth = 2)
ax2.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax2.set_xlim(9.42, 17)
ax2.set_ylim(0, 500)

fig.tight_layout()
plt.show()


# GRAFICO RADIACION TEMPERATURA
fig, ax3 = plt.subplots()

color = 'tab:red'
ax3.set_xlabel('Tiempo [hr]')
ax3.set_ylabel('DNI [W/m2]', color=color, fontsize = 30)
ax3.plot(time[169741:302398]/3600, fit_rad(time[169741:302398]/3600), color=color, linewidth = 2)
ax3.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)

ax4 = ax3.twinx()

color = 'tab:blue'
ax4.set_ylabel('Temperatura [C]', color=color, fontsize = 30)  
ax4.plot(time[169741:302398]/3600, celsius(u[169741:302398, 0]), color=color, linewidth = 2)
ax4.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax4.set_xlim(9.42, 17)
ax4.set_ylim(0, 500)


fig.tight_layout()  
plt.show()

# GRAFICO POTENCIA TERMICA Y TRABAJO
fig, ax5 = plt.subplots()

color = 'tab:red'
ax5.set_xlabel('Tiempo [hr]', fontsize = 30)
ax5.set_ylabel('Trabajo turbina [MW]', color=color, fontsize = 30)
ax5.plot(time2[21829:154487]/3600, Wt[21829:154487]/10**6, color=color, linewidth = 2)
ax5.tick_params(axis='y', labelcolor=color)
#ax5_little = fig.add_axes([0.58, 0.58, 0.25, 0.2])
#ax5_little.plot(time2[61724:62863], Wt[61725:62864]/10**6)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax5.set_xlim(9.42, 17)
ax5.set_ylim(1.05, 1.056)


ax6 = ax5.twinx()

color = 'tab:blue'
ax6.set_ylabel('Potencia Térmica [MW]', color=color, fontsize = 30)  
ax6.plot(time2[21829:154487]/3600, Qpb[21829:154487]/10**6, color=color, linewidth = 2)
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
ax7.plot(time2[21829:154487]/3600, eta_gen * Wt[21829:154487]/10**6, linewidth = 2)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
ax7.set_xlim(9.42, 17)
ax7.set_ylim(1.04, 1.046)
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