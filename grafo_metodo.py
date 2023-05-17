import pvlib
from pvlib.location import Location
from pvlib import clearsky
import pandas as pd
import pytz
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.close('all')

# ------------------------------------------------------------------------
#                           Base de datos
# ------------------------------------------------------------------------

file = 'base.xlsx'
base_ene = pd.read_excel(file, sheet_name='22ene')
base_abr = pd.read_excel(file, sheet_name='Sheet5')
rad_ene = base_ene['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
rad_abr = base_abr['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
hora_min = base_ene['Hora total']

# Parametros
dt = 60 # cada x segundos
n_times = int(23 * 60 * 60/ dt) + 1
time = np.linspace(0, 23 * 60 * 60, n_times)

# ------------------------------------------------------------------------
#                Radiacion dia despejado con pvlib
# ------------------------------------------------------------------------

tz = 'Chile/Continental'
lat, lon, alt = -23.06, -70.38, 10
tus = Location(lat, lon, tz, alt)
times = pd.date_range(start='2017-01-11', end='2017-01-12', 
                      freq='1min', tz= tz)
cs = tus.get_clearsky(times)['dni']
rad_nom = np.array([])

for i in range(len(time)):
    rad_nom = np.append(rad_nom, cs[i])

# ------------------------------------------------------------------------
#                        Radiacion base datos
# ------------------------------------------------------------------------

int_ene = CubicSpline(hora_min, rad_ene)
int_abr = CubicSpline(hora_min, rad_abr)

r = int_abr(time/3600)
x, _ = find_peaks(r, prominence = 10)

# ------------------------------------------------------------------------
#                               Graficos
# ------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax.set_ylabel('$Irradiancia [Wm^{-2}]$', fontsize = 30)
ax.plot(time/3600, rad_nom, linewidth = 2)
ax.fill_between(time/3600, rad_nom, color = 'lavender')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

fig, ax2 = plt.subplots()
ax2.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax2.set_ylabel('$Irradiancia [Wm^{-2}]$', fontsize = 30)
ax2.plot(time/3600, rad_nom, linewidth = 2)
ax2.plot(time/3600, int_ene(time/3600), linewidth = 2, color = 'red')
ax2.fill_between(time/3600, int_ene(time/3600), color = 'mistyrose')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

fig, ax3 = plt.subplots()
ax3.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax3.set_ylabel('$Irradiancia [Wm^{-2}]$', fontsize = 30)
ax3.plot(time/3600, rad_nom, linewidth = 2)
ax3.plot(time/3600, int_abr(time/3600), linewidth = 2, color = 'red')
ax3.fill_between(time/3600, int_abr(time/3600), color = 'mistyrose')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

fig, ax4 = plt.subplots()
ax4.set_xlabel('$Tiempo [Hr]$', fontsize = 30)
ax4.set_ylabel('$Irradiancia [Wm^{-2}]$', fontsize = 30)
ax4.plot(time/3600, rad_nom, linewidth = 2)
ax4.plot(time/3600, int_abr(time/3600), linewidth = 2, color = 'red')
ax4.plot(time[x]/3600, r[x], 'o', markersize = 8, color = 'red')
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.show()

