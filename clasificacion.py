# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:50:49 2022

@author: tamar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy import interpolate
from scipy.signal import find_peaks


file = 'base.xlsx'

variables = pd.read_excel(file, sheet_name='VARIABLES')
base_mes = pd.read_excel(file, sheet_name='ENE')
prom = pd.read_excel(file, sheet_name='PROMEDIO MENSUAL')
month='ene'

hora = variables['hora']
rad = base_mes['Radiación Directa Normal (estimado) en 2.0 metros [mean]']
dia = base_mes['Dia']
mes = [['ene','feb','mar','abr','may','jun','jul','agos','sept','oct','nov','dic'],
       [31,28,31,30,31,30,31,31,30,31,30,31]]

HORA=prom['hora']
ene_prom=prom['ene']
jun_prom=prom['jun']

ene_curva = interpolate.CubicSpline(HORA,ene_prom)
jun_curva = interpolate.CubicSpline(HORA,jun_prom)


n = len(base_mes)
m = len(mes[0])
x_values = np.linspace(0,23,100)
list = []

for i in range(m):
    if month=='ene' or month=='mar' or month=='may' or month=='jul' or month=='agos' or month=='oct' or month=='dic':
        list=range(1,32)
    elif month=='abr' or month=='jun' or month=='sept' or month=='nov':
        list=range(1,31)
    else:
        list=range(1,29)

dic = dict()
for i in range(len(list)):
    dic[list[i]]=[]

for i in range(n):
    if month=='ene' or month=='mar' or month=='may' or month=='jul' or month=='agos' or month=='oct' or month=='dic':
        if dia[i]==1:
            dic[1].append(rad[i])
        elif dia[i]==2:
            dic[2].append(rad[i])
        elif dia[i]==3:
            dic[3].append(rad[i])
        elif dia[i]==4:
            dic[4].append(rad[i])
        elif dia[i]==5:
            dic[5].append(rad[i])
        elif dia[i]==6:
            dic[6].append(rad[i])
        elif dia[i]==7:
            dic[7].append(rad[i])
        elif dia[i]==8:
            dic[8].append(rad[i])
        elif dia[i]==9:
            dic[9].append(rad[i])
        elif dia[i]==10:
            dic[10].append(rad[i])
        elif dia[i]==11:
            dic[11].append(rad[i])
        elif dia[i]==12:
            dic[12].append(rad[i])
        elif dia[i]==13:
            dic[13].append(rad[i])
        elif dia[i]==14:
            dic[14].append(rad[i])
        elif dia[i]==15:
            dic[15].append(rad[i])
        elif dia[i]==16:
            dic[16].append(rad[i])
        elif dia[i]==17:
            dic[17].append(rad[i])
        elif dia[i]==18:
            dic[18].append(rad[i])
        elif dia[i]==19:
            dic[19].append(rad[i])
        elif dia[i]==20:
            dic[20].append(rad[i])
        elif dia[i]==21:
            dic[21].append(rad[i])
        elif dia[i]==22:
            dic[22].append(rad[i])
        elif dia[i]==23:
            dic[23].append(rad[i])
        elif dia[i]==24:
            dic[24].append(rad[i])
        elif dia[i]==25:
            dic[25].append(rad[i])
        elif dia[i]==26:
            dic[26].append(rad[i])
        elif dia[i]==27:
            dic[27].append(rad[i])
        elif dia[i]==28:
            dic[28].append(rad[i])
        elif dia[i]==29:
            dic[29].append(rad[i])
        elif dia[i]==30:
            dic[30].append(rad[i])
        elif dia[i]==31:
            dic[31].append(rad[i])
    elif month=='abr' or month=='jun' or month=='sept' or month=='nov':
        if dia[i]==1:
            dic[1].append(rad[i])
        elif dia[i]==2:
            dic[2].append(rad[i])
        elif dia[i]==3:
            dic[3].append(rad[i])
        elif dia[i]==4:
            dic[4].append(rad[i])
        elif dia[i]==5:
            dic[5].append(rad[i])
        elif dia[i]==6:
            dic[6].append(rad[i])
        elif dia[i]==7:
            dic[7].append(rad[i])
        elif dia[i]==8:
            dic[8].append(rad[i])
        elif dia[i]==9:
            dic[9].append(rad[i])
        elif dia[i]==10:
            dic[10].append(rad[i])
        elif dia[i]==11:
            dic[11].append(rad[i])
        elif dia[i]==12:
            dic[12].append(rad[i])
        elif dia[i]==13:
            dic[13].append(rad[i])
        elif dia[i]==14:
            dic[14].append(rad[i])
        elif dia[i]==15:
            dic[15].append(rad[i])
        elif dia[i]==16:
            dic[16].append(rad[i])
        elif dia[i]==17:
            dic[17].append(rad[i])
        elif dia[i]==18:
            dic[18].append(rad[i])
        elif dia[i]==19:
            dic[19].append(rad[i])
        elif dia[i]==20:
            dic[20].append(rad[i])
        elif dia[i]==21:
            dic[21].append(rad[i])
        elif dia[i]==22:
            dic[22].append(rad[i])
        elif dia[i]==23:
            dic[23].append(rad[i])
        elif dia[i]==24:
            dic[24].append(rad[i])
        elif dia[i]==25:
            dic[25].append(rad[i])
        elif dia[i]==26:
            dic[26].append(rad[i])
        elif dia[i]==27:
            dic[27].append(rad[i])
        elif dia[i]==28:
            dic[28].append(rad[i])
        elif dia[i]==29:
            dic[29].append(rad[i])
        elif dia[i]==30:
            dic[30].append(rad[i])
    else:
        if dia[i]==1:
            dic[1].append(rad[i])
        elif dia[i]==2:
            dic[2].append(rad[i])
        elif dia[i]==3:
            dic[3].append(rad[i])
        elif dia[i]==4:
            dic[4].append(rad[i])
        elif dia[i]==5:
            dic[5].append(rad[i])
        elif dia[i]==6:
            dic[6].append(rad[i])
        elif dia[i]==7:
            dic[7].append(rad[i])
        elif dia[i]==8:
            dic[8].append(rad[i])
        elif dia[i]==9:
            dic[9].append(rad[i])
        elif dia[i]==10:
            dic[10].append(rad[i])
        elif dia[i]==11:
            dic[11].append(rad[i])
        elif dia[i]==12:
            dic[12].append(rad[i])
        elif dia[i]==13:
            dic[13].append(rad[i])
        elif dia[i]==14:
            dic[14].append(rad[i])
        elif dia[i]==15:
            dic[15].append(rad[i])
        elif dia[i]==16:
            dic[16].append(rad[i])
        elif dia[i]==17:
            dic[17].append(rad[i])
        elif dia[i]==18:
            dic[18].append(rad[i])
        elif dia[i]==19:
            dic[19].append(rad[i])
        elif dia[i]==20:
            dic[20].append(rad[i])
        elif dia[i]==21:
            dic[21].append(rad[i])
        elif dia[i]==22:
            dic[22].append(rad[i])
        elif dia[i]==23:
            dic[23].append(rad[i])
        elif dia[i]==24:
            dic[24].append(rad[i])
        elif dia[i]==25:
            dic[25].append(rad[i])
        elif dia[i]==26:
            dic[26].append(rad[i])
        elif dia[i]==27:
            dic[27].append(rad[i])
        elif dia[i]==28:
            dic[28].append(rad[i])

x = dict()
dia_variable = dict()
for i in range(len(list)):
    x[list[i]] = []
    dia_variable[list[i]]=0

for i in range(1,len(list)+1):
    x[i] = find_peaks(dic[i], threshold = 5)[0]
    

for i in range(len(x)):
    if len(x[list[i]])>3:
        dia_variable[list[i]]=1
    else:
        dia_variable[list[i]]=0
        

for i in range(len(x)):
    if dia_variable[list[i]]==1:
        print('Dia variable: %s' %(list[i]))


#indices = []
#filtro= []
#hora_filtro = []

#for i in range(len(x[2])):
 #   indices.append(x[2][i])
    
#for i in range(len(indices)):
 #   filtro.append(dic[2][indices[i]])
  #  hora_filtro.append(hora[indices[i]])
    


#x = find_peaks(dic[1])
plt.clf()
plt.figure(num=1, figsize=(30,10))
plt.plot(hora,dic[2], color = 'blue', label='02/01/2017')
plt.plot(hora,ene_curva(hora), dashes=[6, 2],color ='orange', label = 'Promedio mensual')
plt.xlabel('Hora [hr]')
plt.ylabel('Irradiancia [W/m2]')
plt.legend(loc = 'upper left')
#plt.plot(hora_filtro, filtro,'x')
#plt.show()
    
        
        
    
        
        
    
    
    
    