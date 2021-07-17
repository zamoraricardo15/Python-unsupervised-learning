# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:43:07 2020

@author: rzamoram
"""

#Ricardo Zamora Mennigke
#Tarea 4
#Métodos NO Supervisados con Python
#k-medias

##1. [40 puntos] En este ejercicio vamos a usar la tabla de datos SpotifyTop2018 40 V2.csv. 

###a) Cargue la tabla de datos SpotifyTop2018 40 V2.csv
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("SpotifyTop2018_40_V2.csv", decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)
print(datos.shape)

import matplotlib.pyplot as plt
from   sklearn.decomposition import PCA
from   sklearn.datasets import make_blobs
from   sklearn.cluster import KMeans
import numpy as np
from   math import pi

import mglearn
mglearn.plots.plot_kmeans_algorithm() #Prueba de que mglearn funciona

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


###b) Ejecute el m´etodo k−medias para k = 3. Modificaremos los atributos de la clase KMeans(...) como sigue:
    #Usando  max iter = 1000 y n init = 100.
    

# Ejecuta k-medias con 3 clusters
kmedias = KMeans(n_clusters=3, max_iter=1000, n_init=100)  # Declara la instancia de clase
kmedias.fit(datos)
print(kmedias.predict(datos))
centros = np.array(kmedias.cluster_centers_)
print(centros) 




#c) Interprete los resultados del ejercicio anterior usando gr´aficos de barras y gr´aficos tipo Radar. Compare respecto a los resultados obtenidos en la tarea anterior en la que us´o

plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos.columns)

plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos.columns)


#d) Grafique usando colores sobre las dos primeras componentes del plano principal en el An´alisis en Componentes Principales los cl´usteres obtenidos seg´un k-medias (usando k =

pca = PCA(n_components=2)
componentes = pca.fit_transform(datos)
print(componentes)
print(datos.shape)
print(componentes.shape)
plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(datos))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.title('3 Cluster K-Medias')

#e) Usando 50 ejecuciones del m´etodo k−medias grafique el “Codo de Jambu” para este ejemplo. ¿Se estabiliza en alg´un momento la inercia inter–clases?

Nc = range(1, 40)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(datos).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')

Nc = range(1, 12)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(datos).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')


#2. [40 puntos] En este ejercicio vamos a realizar k-medias para la tabla SAheart.csv

#a) Repita el ejercicio 1 usando k = 3 usando esta tabla de datos, usando solo las variables

import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 3")
os.getcwd()
ejemplo10 = pd.read_csv("SAheart.csv", delimiter = ';', decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)
datosnum = datos_dummy.iloc[:,0:7] ##Usando solo numericas del archivo de datos, se eliminan las categoricas
print(datosnum.head())

kmedias = KMeans(n_clusters=3, max_iter=2000, n_init=150)  # Declara la instancia de clase
kmedias.fit(datosnum)
print(kmedias.predict(datosnum))
centros = np.array(kmedias.cluster_centers_)
print(centros) 

plt.figure(1, figsize = (12, 8))
bar_plot(centros, datosnum.columns)

plt.figure(1, figsize = (10, 10))
radar_plot(centros, datosnum.columns)

pca = PCA(n_components=2)
componentes = pca.fit_transform(datosnum)
print(componentes)
print(datosnum.shape)
print(componentes.shape)
plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(datosnum))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.title('3 Cluster K-Medias')


Nc = range(1, 50)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(datosnum).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')


#b) Repita los ejercicios anteriores pero esta vez incluya las variables categ´oricas usando c´odigos disyuntivos completos. ¿Son mejores los resultados?
        
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


kmedias = KMeans(n_clusters=3, max_iter=2000, n_init=150)  # Declara la instancia de clase
kmedias.fit(datos_dummy)
print(kmedias.predict(datos_dummy))
centros = np.array(kmedias.cluster_centers_)
print(centros) 

plt.figure(1, figsize = (12, 8))
bar_plot(centros, datos_dummy.columns)

plt.figure(1, figsize = (10, 10))
radar_plot(centros, datos_dummy.columns)

pca = PCA(n_components=2)
componentes = pca.fit_transform(datos_dummy)
print(componentes)
print(datos_dummy.shape)
print(componentes.shape)
plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(datos_dummy))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.title('3 Cluster K-Medias')


Nc = range(1, 50)
kmediasList = [KMeans(n_clusters=i) for i in Nc]
varianza = [kmediasList[i].fit(datos_dummy).inertia_ for i in range(len(kmediasList))]
plt.plot(Nc,varianza,'o-')
plt.xlabel('Número de clústeres')
plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
plt.title('Codo de Jambu')


#3. [20 puntos] Programe la jerarqu´ıa de clases que se muestra en el siguiente gr´afico

datos_est = pd.read_csv('EjemploEstudiantes.csv',delimiter=';',decimal=",", index_col=0)
#datos = mi_datos(datos_est)
import matplotlib.pyplot as plt
from prince import PCA
import pandas as pd
import numpy as np
import scipy.linalg as la
import os
from   math import pi
from   sklearn.datasets import make_blobs
# Import the dendrogram function and the ward, single, complete, average, linkage and fcluster clustering function from SciPy
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
from scipy.spatial.distance import pdist
import mglearn
import scipy.stats

#############################################################################################################################
class Exploratorio():  ###Base de la jerarquia
    def __init__(self, datos = pd.DataFrame()):  ##convierte a dataframe
        self.__num_filas = datos.shape[0]
        self.__num_columnas = datos.shape[1]
        self.__datos = datos
    @property
    def num_filas(self):  ##procesa numero de filas
        return self.__num_filas
    @property
    def num_columnas(self):  ##procesa numero de columnas
        return self.__num_columnas
    @property
    def datos(self):  ##para siguientes clases
        return self.__datos  
    def maximo(self):  ##estima numero maximo de los datos
        max = self.datos.iloc[0,0]
        for i in range(self.num_filas):
            for j in range(self.num_columnas):
                if self.datos.iloc[i,j] > max:
                    max = self.datos.iloc[i,j]
        return max
    def valores(self):    ##estima valores maximo, minimo, ceros y pares
        min = self.datos.iloc[0,0]
        max = self.datos.iloc[0,0]
        total_ceros = 0
        total_pares = 0
        for i in range(self.num_filas):
            for j in range(self.num_columnas):
                if self.datos.iloc[i,j] > max:
                    max = self.datos.iloc[i,j]
                if self.datos.iloc[i,j] < min:
                    min = self.datos.iloc[i,j]
                if self.datos.iloc[i,j] == 0:
                    total_ceros = total_ceros+1
                if self.datos.iloc[i,j] % 2 == 0:
                    total_pares = total_pares+1
        return {'Maximo' : max, 'Minimo' : min, 'Total_Ceros' : total_ceros, 'Pares' : total_pares}
    def estadisticas(self,nc): ##indica las estadisticas basicas
        media = np.mean(self.datos.iloc[:,nc])
        mediana = np.median(self.datos.iloc[:,nc])
        deviacion = np.std(self.datos.iloc[:,nc])
        varianza = np.var(self.datos.iloc[:,nc])
        maximo = np.max(self.datos.iloc[:,nc])
        minimo = np.min(self.datos.iloc[:,nc])
        return {'Variable' : self.datos.columns.values[nc],
                'Media' : media,
                'Mediana' : mediana,
                'DesEst' : deviacion,
                'Varianza' : varianza,
                'Maximo' : maximo,
                'Minimo' : minimo}
    def plot(self,datos): ##genera el plots descriptivos
        datos = np.random.randn(1000)
        plt.hist(datos) #Histograma
        boxplots = datos.boxplot(return_type='axes') ##box plots
        print(boxplots)
        densidad = datos[datos.columns[:1]].plot(kind='density')
        print(densidad) #Densidad
        datos = np.random.randn(1000)
        plt.style.use('seaborn-white') ##Distribucion
        shapiro_resultados = scipy.stats.shapiro(datos)
        print(shapiro_resultados) ##Test de normalidad
        p_value = shapiro_resultados[1] ##Shapiro-Wilk
        print(p_value)
        alpha = 0.05 ##indica un alpha de 0,05 para las estimaciones
        if p_value > alpha:
        	print('Sí sigue la curva Normal (No se rechaza H0)')
        else:
        	print('No sigue la curva Normal (Se rechaza H0)') ##Condiciones de normalidad
        ks_resultados = scipy.stats.kstest(datos, cdatos='norm')  ###Ktest
        print(ks_resultados)
        p_value = ks_resultados[1]
        print(p_value)
        alpha = 0.05
        if p_value > alpha:
        	print('Sí sigue la curva Normal (No se rechaza H0)')
        else:
        	print('No sigue la curva Normal (Se rechaza H0)')
        
####################################################################################################################
class mi_ACP(Exploratorio): ##Estimar ACP
    def __init__(self, datos , n_componentes = 5): #inicializa
        self.__datos = datos
        self.__n_componentes = n_componentes
        self.__datos_centrados_reducidos = (self.datos - self.datos.mean(axis=0))/self.datos.std(axis=0, ddof = 0)
        self.__matriz_correlaciones = self.__datos_centrados_reducidos.corr()
        self.__valores_propios  = la.eig(self.__matriz_correlaciones)[0].real
        self.__vectores_propios = la.eig(self.__matriz_correlaciones)[1].real
    
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos): ##para asignar clases
        self.__datos = datos
        
    @property
    def n_componentes(self):
        return self.__n_componentes
    @n_componentes.setter
    def n_componentes(self, n_componentes): ##componentes
        self.__n_componentes= n_componentes
    
    @property
    def datos_centrados_reducidos(self): ##determinar centrar
        return self.__datos_centrados_reducidos
    @datos_centrados_reducidos.setter
    def datos_centrados_reducidos(self, datos_centrados_reducidos):
        self.__datos = datos_centrados_reducidos
    
    @property
    def matriz_correlaciones(self):
        return self.__matriz_correlaciones
    @matriz_correlaciones.setter
    def matriz_correlaciones(self, matriz_correlaciones): ##determinar matrices 
        self.__matriz_correlaciones = matriz_correlaciones
    
    @property
    def valores_propios(self):
      return self.__valores_propios
    
    @property
    def vectores_propios(self):
      return self.__vectores_propios
    def datos(self):  
        return self.__datos 
    # Métodos
    
    def coordenadas_ind(self):
      r = np.matmul(self.datos_centrados_reducidos.values, self.vectores_propios)
      r = pd.DataFrame(r, index = self.datos.index)
      r = r.iloc[:, 0:self.n_componentes]
      return r
      
    def cos2_ind(self):
      r = self.coordenadas_ind() ** 2 
      r = r.div(r.sum(axis=1).iloc[0], axis='columns')
      r.index = self.datos.index
      return r.iloc[:, 0:self.n_componentes]
      
    def correlacion_var(self): ##estima correlaciones 
        r = np.sqrt(self.valores_propios) * self.vectores_propios
        r = pd.DataFrame(r, index = self.datos.columns)
        r = r.iloc[:, np.arange(0, self.n_componentes)]
        return r    
        
    def cos2_var(self): 
        return((self.correlacion_var()) ** 2)
        
    def var_explicada(self):
        r = [x/self.datos.shape[1] * 100 for x in self.valores_propios]
        return(r)
    
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'): ##genera plot plano principal
        x = self.coordenadas_ind()[ejes[0]].values
        y = self.coordenadas_ind()[ejes[1]].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada()[ejes[0]], 2)
        inercia_y = round(self.var_explicada()[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind().index):
                plt.annotate(txt, (x[i], y[i]))
                
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'): ##genera plot circulo correlacion
        cor = self.correlacion_var().iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada()[ejes[0]], 2)
        inercia_y = round(self.var_explicada()[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var().index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
                         
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):    ##Genera plano sobreposicion plano-circulo
        x = self.coordenadas_ind()[ejes[0]].values
        y = self.coordenadas_ind()[ejes[1]].values
        cor = self.correlacion_var().iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var().iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada()[ejes[0]], 2)
        inercia_y = round(self.var_explicada()[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind().index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var().index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')


class Clusters(Exploratorio):  ##genera los clusters a partir de jerarquia y k-medias
    def __init__(self, datos , n_componentes = 5):
        self.__datos = datos
        self.__n_componentes = n_componentes
        self.__datos_centrados_reducidos = (self.datos - self.datos.mean(axis=0))/self.datos.std(axis=0, ddof = 0)
        self.__matriz_correlaciones = self.__datos_centrados_reducidos.corr()
        self.__valores_propios  = la.eig(self.__matriz_correlaciones)[0].real
        self.__vectores_propios = la.eig(self.__matriz_correlaciones)[1].real
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
        
    @property
    def n_componentes(self):
        return self.__n_componentes
    @n_componentes.setter
    def n_componentes(self, n_componentes):
        self.__n_componentes= n_componentes
    
    @property
    def datos(self):  
        return self.__datos 
    def Clusters(self):  
        return self.__Clusters
    #Función para calcular los centroides de cada cluster
    def centroide(self, num_cluster, datos, clusters):
      ind = clusters == num_cluster
      return(pd.DataFrame(datos[ind].mean()).T)
    #Función para graficar los gráficos de Barras para la interpretación de clústeres
    def bar_plot(self, centros, labels, cluster = None, var = None): ##genera grafico de barras para jerarquico y k-medias
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
    #Función para graficar los gráficos tipo Radar para la interpretación de clústeres
    def radar_plot(self, centros, labels):
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
        
        
#####################################################################################################################       
class Jerarquico(Clusters): ##genera el clustering jerarquico y pasa estimacion a clustering class
    def __init__(self, datos , n_componentes = 5):
        self.__datos = datos
        self.__n_componentes = n_componentes
        self.__datos_centrados_reducidos = (self.datos - self.datos.mean(axis=0))/self.datos.std(axis=0, ddof = 0)
        self.__matriz_correlaciones = self.__datos_centrados_reducidos.corr()
        self.__valores_propios  = la.eig(self.__matriz_correlaciones)[0].real
        self.__vectores_propios = la.eig(self.__matriz_correlaciones)[1].real
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
        
    @property
    def n_componentes(self):
        return self.__n_componentes
    @n_componentes.setter
    def n_componentes(self, n_componentes):
        self.__n_componentes= n_componentes
    @property
    def Clusters(self):  
        return self.__Clusters
    def agregation(self, datos, centroide): ##estima funcion que su usa en clustering
        ward_res = ward(datos)         #Ward
        single_res = single(datos)     #Salto mínimo
        complete_res = complete(datos) #Salto Máximo
        average_res = average(datos)   #Promedio
        dendrogram(average_res,labels= datos.index.tolist())
        plt.figure(figsize=(13,10))
        dendrogram(complete_res,labels= datos.index.tolist())
        plt.figure(figsize=(13,10))
        dendrogram(single_res,labels= datos.index.tolist())
        plt.figure(figsize=(13,10))
        dendrogram(ward_res,labels= datos.index.tolist())
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
        centros = np.array(pd.concat([centroide(0, datos, grupos), centroide(1, datos, grupos), centroide(2, datos, grupos)]))
        print(centros)    
        plt.figure(1, figsize = (12, 8))
        bar_plot(centros, datos.columns)
        grupos = fcluster(linkage(pdist(datos), method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
        grupos = grupos-1 # Se resta 1 para que los clústeres se enumeren de 0 a (K-1), como usualmente lo hace Python
        # El siguiente print es para ver en qué cluster quedó cada individuo
        print(grupos)
        centros = np.array(pd.concat([centroide(0, datos, grupos), centroide(1, datos, grupos), centroide(2, datos, grupos)]))
        print(centros)
        plt.figure(1, figsize = (10, 10))
        radar_plot(centros, datos.columns)
    
    
    
################################################################################################################################    
class KMedias(Clusters):   ##Estima k-media para clase clustering
    def __init__(self, datos , n_componentes = 5):
        self.__datos = datos
        self.__n_componentes = n_componentes
        self.__datos_centrados_reducidos = (self.datos - self.datos.mean(axis=0))/self.datos.std(axis=0, ddof = 0)
        self.__matriz_correlaciones = self.__datos_centrados_reducidos.corr()
        self.__valores_propios  = la.eig(self.__matriz_correlaciones)[0].real
        self.__vectores_propios = la.eig(self.__matriz_correlaciones)[1].real
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
        
    @property
    def n_componentes(self):
        return self.__n_componentes
    @n_componentes.setter
    def n_componentes(self, n_componentes):
        self.__n_componentes= n_componentes
    @property
    def Clusters(self):  
        return self.__Clusters
    def estimar(self, datos, centroide):
        # Ejecuta k-medias con 3 clusters
        kmedias = KMeans(n_clusters=3)  # Declara la instancia de clase
        kmedias.fit(datos)
        print(kmedias.predict(datos))
        centros = np.array(kmedias.cluster_centers_)
        print(centros) 
        plt.figure(1, figsize = (12, 8))
        bar_plot(centros, datos.columns)
        plt.figure(1, figsize = (10, 10))
        radar_plot(centros, datos.columns)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(datos)
        print(componentes)
        print(datos.shape)
        print(componentes.shape)
        plt.scatter(componentes[:, 0], componentes[:, 1],c=kmedias.predict(datos))
        plt.xlabel('componente 1')
        plt.ylabel('componente 2')
        plt.title('3 Cluster K-Medias')
    def jambu(self, datos): ##estima codo de jambu
        Nc = range(1, 20)
        kmediasList = [KMeans(n_clusters=i) for i in Nc]
        varianza = [kmediasList[i].fit(datos).inertia_ for i in range(len(kmediasList))]
        plt.plot(Nc,varianza,'o-')
        plt.xlabel('Número de clústeres')
        plt.ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
        plt.title('Codo de Jambu')
##########################################################################################################################
    
prueba = Exploratorio(datos)
prueba.ACP() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    