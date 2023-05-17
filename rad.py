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
base_mes = pd.read_excel(file, sheet_name='Hoja5')
rad = base_mes['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
hora_minuto = base_mes['Hora total']

base_abr = pd.read_excel(file, sheet_name='Sheet5')
rad_abr = base_abr['Radiación Directa Normal (estimado) en 2.0 metros [mean]']



fit_rad = CubicSpline(hora_minuto, rad)
fit_rad_abr = CubicSpline(hora_minuto, rad_abr)

def celsius(t):
    return t - 273.15

time = np.linspace(0, 24, 1000)
# Variables guardadas

fig, ax = plt.subplots()
ax.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax.set_ylabel('$Irradiancia [Wm^-2]$', fontsize = 30)
ax.plot(time, fit_rad(time), linewidth = 2, label = 'Día no variable')
ax.plot(time, fit_rad_abr(time), linewidth = 2, color = 'red', label = 'Día variable')
plt.yticks(fontsize = 20) 
plt.xticks(fontsize = 20)
ax.legend(fontsize = 20)
plt.show()
