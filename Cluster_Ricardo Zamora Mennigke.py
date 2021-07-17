# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:40:49 2020

@author: rzamoram
"""

#Tarea 3
#Ricardo Zamora Mennigke

#Ejercicio 1
##En este ejercicio vamos a usar la tabla de datos SpotifyTop2018 40 V2.csv


##a) Cargue la tabla de datos SpotifyTop2018 40 V2.csv
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("SpotifyTop2018_40_V2.csv", decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

##b) Ejecute un Clustering Jer´arquico con la agregaci´on del Salto M´aximo, Salto M´ınimo, Promedio y Ward. Grafique el dendograma con cortes para dos y tres cl´usteres

###Clustering Jerárquico: paquetes
import numpy as np
from   math import pi
from   sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# Import the dendrogram function and the ward, single, complete, average, linkage and fcluster clustering function from SciPy
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
from scipy.spatial.distance import pdist


###Función para calcular los centroides de cada cluster
def centroide(num_cluster, datos, clusters):
  ind = clusters == num_cluster
  return(pd.DataFrame(datos[ind].mean()).T)

###Función para graficar los gráficos de Barras para la interpretación de clústere
def bar_plot(centros, labels, cluster = None, var = None):
    from math import ceil, floor
    from seaborn import color_palette
    colores = color_palette()
    minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
    def inside_plot(valores, labels, titulo):
        plt.barh(range(len(valores)), valores, 1/1.5, color = colores)
        plt.xlim(minimo, ceil(centros.max()))
        plt.title(titulo)
    if var is not None:
        centros = np.array([n[[x in var for x in labels]] for n in centros])
        colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
        labels = labels[[x in var for x in labels]]
    if cluster is None:
        for i in range(centros.shape[0]):
            plt.subplot(1, centros.shape[0], i + 1)
            inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
            plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([]) 
    else:
        pos = 1
        for i in cluster:
            plt.subplot(1, len(cluster), pos)
            inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
            plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([]) 
            pos += 1
            
###Función para graficar los gráficos tipo Radar para la interpretación de clústeres            
def radar_plot(centros, labels):
    from math import pi
    centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if 
                        max(n) != min(n) else (n/n * 50) for n in centros.T])
    angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angulos += angulos[:1]
    ax = plt.subplot(111, polar = True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angulos[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
           ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
           color = "grey", size = 8)
    plt.ylim(-10, 100)
    for i in range(centros.shape[1]):
        valores = centros[:, i].tolist()
        valores += valores[:1]
        ax.plot(angulos, valores, linewidth = 1, linestyle = 'solid', 
                label = 'Cluster ' + str(i))
        ax.fill(angulos, valores, alpha = 0.3)
    plt.legend(loc='upper right', bbox_to_anchor = (0.1, 0.1))

###Estimacion de agregaciones
ward_res = ward(datos)         #Ward
single_res = single(datos)     #Salto mínimo
complete_res = complete(datos) #Salto Máximo
average_res = average(datos)   #Promedio

###Dendograma con metodo promedio
dendrogram(average_res,labels= datos.index.tolist())
# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [80000, 80000], '--', c='k')
ax.plot(limites, [40000, 40000], '--', c='k')
ax.text(limites[1], 80000, ' dos clústeres', va='center', fontdict={'size': 15})
ax.text(limites[1], 40000, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")

###Dendograma con metodo salto maximo
dendrogram(complete_res,labels= datos.index.tolist())
# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [125000, 125000], '--', c='k')
ax.plot(limites, [75000, 75000], '--', c='k')
ax.text(limites[1], 125000, ' dos clústeres', va='center', fontdict={'size': 15})
ax.text(limites[1], 75000, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")

###Dendograma con metodo salto minimo
dendrogram(single_res,labels= datos.index.tolist())
# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [30000, 30000], '--', c='k')
ax.plot(limites, [23700, 23700], '--', c='k')
ax.text(limites[1], 30000, ' dos clústeres', va='center', fontdict={'size': 15})
ax.text(limites[1], 23700, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")

###Dendograma con metodo Ward con 2 y 3 cluesters
dendrogram(ward_res,labels= datos.index.tolist())
# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [200000, 200000], '--', c='k')
ax.plot(limites, [120000, 120000], '--', c='k')
ax.text(limites[1], 200000, ' dos clústeres', va='center', fontdict={'size': 15})
ax.text(limites[1], 120000, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")


##c) Usando tres cl´usteres interprete los resultados del ejercicio anterior para el caso de agregaci´on de Ward usando gr´aficos de barras y gr´aficos tipo Radar.

#Interpretación con 3 clústeres - Gráficos de Barras
grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='binary'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos.columns)

#Interpretación 3 Clústeres - Gráfico Radar plot con Ward
grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)

##d) Grafique usando colores sobre las dos primeras componentes del plano principal en el An´alisis en Componentes Principales los cl´usteres obtenidos seg´un la clasificaci´on Jer´arquica (usando tres cl´usteres).

import matplotlib.pyplot as plt
from prince import PCA

class ACP:
    def __init__(self, datos, n_componentes = 3): 
        self.__datos = datos
        self.__modelo = PCA(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations(datos)
        self.__coordenadas_ind = self.__modelo.row_coordinates(datos)
        self.__contribucion_ind = self.__modelo.row_contributions(datos)
        self.__cos2_ind = self.__modelo.row_cosine_similarities(datos)
        self.__var_explicada = [x * 100 for x in self.__modelo.explained_inertia_]
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    @property
    def modelo(self):
        return self.__modelo
    @property
    def correlacion_var(self):
        return self.__correlacion_var
    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind
    @property
    def contribucion_ind(self):
        return self.__contribucion_ind
    @property
    def cos2_ind(self):
        return self.__cos2_ind
    @property
    def var_explicada(self):
        return self.__var_explicada
        self.__var_explicada = var_explicada
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, color = 'yellow')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'green', linestyle = '--')
        plt.axvline(x = 0, color = 'red', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        c = plt.Circle((0, 0), radius = 1, color = 'blue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'green', linestyle = '--')
        plt.axvline(x = 0, color = 'red', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'blue', ha = 'center', va = 'center')
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'green', linestyle = '--')
        plt.axvline(x = 0, color = 'red', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'yellow', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'blue', ha = 'center', va = 'center')
                

acp = ACP(datos,n_componentes=3)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal()
acp.plot_circulo()
acp.plot_sobreposicion()


#Ejercicio 2
##En este ejercicio vamos a realizar un Clustering Jer´arquico para la tabla SAheart.csv

import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 3")
os.getcwd()
ejemplo10 = pd.read_csv("SAheart.csv", delimiter = ';', decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

##a) Efectue un Clustering usando solo las variables numericas
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)
datosnum = datos_dummy.iloc[:,0:7] ##Usando solo numericas del archivo de datos, se eliminan las categoricas
print(datosnum.head())

###Estimacion de agregaciones
ward_res = ward(datosnum)         #Ward
single_res = single(datosnum)     #Salto mínimo
complete_res = complete(datosnum) #Salto Máximo
average_res = average(datosnum)   #Promedio


###Dendograma con metodo Ward con 2 y 3 cluesters
dendrogram(ward_res,labels= datosnum.index.tolist())
# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [350, 350], '--', c='k')
ax.text(limites[1], 350, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")


#Interpretación con 3 clústeres - Gráficos de Barras
grupos = fcluster(linkage(pdist(datosnum), method = 'ward', metric='binary'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datosnum, grupos), 
                              centroide(1, datosnum, grupos),
                              centroide(2, datosnum, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datosnum.columns)

#Interpretación 3 Clústeres - Gráfico Radar plot con Ward
grupos = fcluster(linkage(pdist(datosnum), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datosnum, grupos), 
                              centroide(1, datosnum, grupos),
                              centroide(2, datosnum, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datosnum.columns)




#b) Efectue un Clustering Jer´arquico usando las variables num´ericas y las variables categ´oricas

def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod

datos["famhist"] = recodificar(datos["famhist"], {'Present':1,'Absent':2})
datos["chd"] = recodificar(datos["chd"], {'No':0,'Si':1})
print(datos.head())
print(datos.dtypes)
# Conviertiendo la variables en Dummy
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)



###Estimacion de agregaciones
ward_res = ward(datos_dummy)         #Ward
single_res = single(datos_dummy)     #Salto mínimo
complete_res = complete(datos_dummy) #Salto Máximo
average_res = average(datos_dummy)   #Promedio


###Dendograma con metodo Ward con 2 y 3 cluesters
dendrogram(ward_res,labels= datos_dummy.index.tolist())
# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [337, 337], '--', c='k')
ax.text(limites[1], 337, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")


#Interpretación con 3 clústeres - Gráficos de Barras
grupos = fcluster(linkage(pdist(datos_dummy), method = 'ward', metric='binary'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos_dummy, grupos), 
                              centroide(1, datos_dummy, grupos),
                              centroide(2, datos_dummy, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos_dummy.columns)

#Interpretación 3 Clústeres - Gráfico Radar plot con Ward
grupos = fcluster(linkage(pdist(datos_dummy), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos_dummy, grupos), 
                              centroide(1, datos_dummy, grupos),
                              centroide(2, datos_dummy, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos_dummy.columns)


#c) Explique las diferencias de los dos ejercicios anteriores ¿Cu´al le parece m´as interesante? ¿Por qu´e?


##3.Dada la siguiente matriz de disimilitudes entre cuatro individuos A1, A2, A3 y A4, a mano
import os
import pandas as pd
from PIL import Image
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 3")
os.getcwd()
image = Image.open("SaltoMinimo.jpeg")
plt.imshow(image)

image = Image.open("SaltoMaximo.jpeg")
plt.imshow(image)

image = Image.open("SaltoPromedio.jpeg")
plt.imshow(image)


##Ejercicio 4
###a) Se define la distancia de Chebychev como sigue
import math
#from numpy.polynomial import chebyshev as Chebyshev
#from chebyshev import Chebyshev
#ch = Chebyshev(0, math.pi / 12, 8, math.sin)
#ch.eval(0.1)
math.sin(0.1)

class Chebyshev:
    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func
        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                  for k in range(n)]) for j in range(n)]

    def eval(self, x):
        a,b = self.a, self.b
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)          
        for cj in self.c[-2:0:-1]:           
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]  



import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 3")
os.getcwd()
datos = pd.read_csv('EjemploEstudiantes.csv',delimiter=';',decimal=",",index_col=0)
print(datos)

#b) Calcule la matriz de distancias usando la distancia de Chebychev para la tabla de datos EjemploEstudiantes.csv.
#eval(datos)
grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros) ##Matriz distnacias

    
#c) Para la tabla de datos EjemploEstudiantes.csv ejecute un Clustering Jer´arquico usando la distancia de Chebychev 
grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos.columns)

ward_res = ward(datos) 
plt.figure(figsize=(13,10))
dendrogram(ward_res,labels= datos.index.tolist())

# Agrega cortes con 2 y 3 clústeres con agregación de Ward
ax = plt.gca()
limites = ax.get_xbound()
ax.plot(limites, [7.25, 7.25], '--', c='k')
ax.plot(limites, [4, 4], '--', c='k')
ax.text(limites[1], 7.25, ' dos clústeres', va='center', fontdict={'size': 15})
ax.text(limites[1], 4, ' tres clústeres', va='center', fontdict={'size': 15})
plt.xlabel("Orden en el eje X")
plt.ylabel("Distancia o Agregación")

grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)    
plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos.columns)

grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
# El siguiente print es para ver en qué cluster quedó cada individuo
print(grupos)
centros = np.array(pd.concat([centroide(0, datos, grupos), 
                              centroide(1, datos, grupos),
                              centroide(2, datos, grupos)]))
print(centros)
plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)