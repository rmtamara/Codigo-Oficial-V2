import numpy as np

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

# Datos de ejemplo
datos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ventana = 3

# Calcular el promedio móvil centrado
promedios = promedio_movil_centrado(datos, datos, ventana)[0]
tiempos = promedio_movil_centrado(datos, datos, ventana)[1]

