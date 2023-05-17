"""
En este script se determina que dias (y cuantos) del mes son variable.
Para ello se utiliza find_peaks como metodo para encontrar maximos 
locales.
Para analizar, se debe cambiar la variable 'mes' para analizar caso a caso
mensual. Al igual que en la variable 'base_mes' vinculada a la hoja de
calculo.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy import interpolate
from scipy.signal import find_peaks

# ------------------------------------------------------------------------
#                           Base de datos
# ------------------------------------------------------------------------

file = 'base.xlsx'

variables = pd.read_excel(file, sheet_name='VARIABLES')
base_mes = pd.read_excel(file, sheet_name='OCT')
prom = pd.read_excel(file, sheet_name='PROMEDIO MENSUAL')
month='ene'

hora = variables['hora']
rad = base_mes['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
dia = base_mes['Dia']

# ------------------------------------------------------------------------
#                     Clasificacion dias variables
# ------------------------------------------------------------------------

# Parametros
dt = 60 # cada x segundos
n_times = int(23 * 60 * 60/ dt) + 1
time = np.linspace(0, 23 * 60 * 60, n_times)
mes = 'oct'
mes_par = ['abr', 'jun', 'sept', 'nov'] 
mes_impar = ['ene', 'mar', 'may', 'jul', 'agos', 'dec']

# Variables
dic_rad = dict()
dias_var = []



def lista_dias(mes):
    if mes in mes_par:
        dias = np.linspace(1, 30, 30)
    elif mes in mes_impar:
        dias = np.linspace(1, 31, 31)
    elif mes == 'oct':
        dias = np.linspace(1, 17, 17)
    else:
        dias = np.linspace(1, 28, 28)
    return dias

dias = lista_dias(mes)

for i in range(len(dias)):
    dic_rad[dias[i]] = []

for i in range(len(dia)):
    if dia[i] == 1:
        dic_rad[1].append(rad[i])
    elif dia[i] == 2:
        dic_rad[2].append(rad[i])
    elif dia[i] == 3:
        dic_rad[3].append(rad[i])
    elif dia[i] == 4:
        dic_rad[4].append(rad[i])
    elif dia[i] == 5:
        dic_rad[5].append(rad[i])
    elif dia[i] == 6:
        dic_rad[6].append(rad[i])
    elif dia[i] == 7:
        dic_rad[7].append(rad[i])
    elif dia[i] == 8:
        dic_rad[8].append(rad[i])
    elif dia[i] == 9:
        dic_rad[9].append(rad[i])
    elif dia[i] == 10:
        dic_rad[10].append(rad[i])
    elif dia[i] == 11:
        dic_rad[11].append(rad[i])
    elif dia[i] == 12:
        dic_rad[12].append(rad[i])
    elif dia[i] == 13:
        dic_rad[13].append(rad[i])
    elif dia[i] == 14:
        dic_rad[14].append(rad[i])
    elif dia[i] == 15:
        dic_rad[15].append(rad[i])
    elif dia[i] == 16:
        dic_rad[16].append(rad[i])
    elif dia[i] == 17:
        dic_rad[17].append(rad[i])
    elif dia[i] == 18:
        dic_rad[18].append(rad[i])
    elif dia[i] == 19:
        dic_rad[19].append(rad[i])
    elif dia[i] == 20:
        dic_rad[20].append(rad[i])
    elif dia[i] == 21:
        dic_rad[21].append(rad[i])
    elif dia[i] == 22:
        dic_rad[22].append(rad[i])
    elif dia[i] == 23:
        dic_rad[23].append(rad[i])
    elif dia[i] == 24:
        dic_rad[24].append(rad[i])
    elif dia[i] == 25:
        dic_rad[25].append(rad[i])
    elif dia[i] == 26:
        dic_rad[26].append(rad[i])
    elif dia[i] == 27:
        dic_rad[27].append(rad[i])
    elif dia[i] == 28:
        dic_rad[28].append(rad[i])
    elif dia[i] == 29:
        dic_rad[29].append(rad[i])
    elif dia[i] == 30:
        dic_rad[30].append(rad[i])
    elif dia[i] == 31:
        dic_rad[31].append(rad[i])

def dia_variable(x):
    """
    x: lista
    """
    output = find_peaks(x, threshold = 5)[0]
    return output

for i in range(len(dias)):
    peak = dia_variable(dic_rad[dias[i]])
    if len(peak)>3:
        dias_var.append(dias[i])
