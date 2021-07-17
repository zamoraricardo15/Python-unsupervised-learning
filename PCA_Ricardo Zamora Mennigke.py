# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:26:43 2020

@author: rzamoram
"""

#Tarea 1
#Ricardo Zamora Mennigke

#Ejercicio 1
##En este ejercicio vamos a usar la tabla de datos SpotifyTop2018 40 V2.csv


##a) Calcule el resumen numerico, interprete los resultados para dos variables.
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("SpotifyTop2018_40_V2.csv", decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

print(datos.dropna().describe())
print(datos.describe())
print(datos.mean(numeric_only=True))
print(datos.median(numeric_only=True))
print(datos.std(numeric_only=True))
print(datos.max(numeric_only=True))
###En este caso analizando danceability y energy, en este caso al ser ambas variables de 0 a 1, con un 0,719 y 0,662, 
###indican que en promedio las canciones mas reproducidas en Spotify del 2018, son aptas y tienen intensidad y actividad 
###la desviacion estandar de 0,151 y 0,138, indica una desviacion de estas dos variables no tan amplia. El maximo de 0,922 y 0,909
###indica que no existe cancion en 10 es decir, completamente apta para bailar, ni medida de intensidad y actividad total


##b) Realice el test de normalidad para una variable e interprete el resultado.
import scipy.stats
#import numpy as np
shapiro_resultados = scipy.stats.shapiro(datos.energy)
print(shapiro_resultados)
p_value = shapiro_resultados[1]
print(p_value)
# interpretación
alpha = 0.05
if p_value > alpha:
	print('Sí sigue la curva Normal (No se rechaza H0)')
else:
	print('No sigue la curva Normal (Se rechaza H0)')

from statsmodels.graphics.gofplots import qqplot
#from matplotlib import pyplot
qqplot(datos.energy, line='s')
    
ks_resultados = scipy.stats.kstest(datos.energy, cdf='norm')
print(ks_resultados)
p_value = ks_resultados[1]
print(p_value)
# interpretación
alpha = 0.05
if p_value > alpha:
	print('Sí sigue la curva Normal (No se rechaza H0)')
else:
	print('No sigue la curva Normal (Se rechaza H0)')
###Aqui resulta interesante notar que el test de Shapiro-Wilk no rechaza la hipotesis nula y el de Kolmogrov-Smirnov. Luego de una breve
###Investigacion resulta importante notar que las estimaciones de muestra mas pequenas generan mejores estimaciones en Shapiro por lo que resulta mejor 
###tomar en cuenta esta prueba, ya que la prueba de Kolmogrov tiende a castigar mas en la estimacion de muestras pequenas, por lo que,
###es se toma la Shapiro y se asume que la distribucion de la variable de energia es normal

    

##c) Realice un grafico de dispersion e interprete dos similitudes en el grafico.
import seaborn as sns
#import matplotlib.pyplot as plt

#sns.pairplot(datos, hue='chd', size=2.5)
sns.pairplot(datos, size=2.5)
sns.pairplot(datos, hue='energy', size=2.5)
###Se denota en el grafico de dispersion que tiem_signature y dinstrumentalness muestran una agrupacion en linea recta respecto a otras variables, 
###esto indicaria que son variables categoricas o dicotomicas sin importar el valor de los demas. Cabe senalar seguidamente que entre ninguna de 
###las variables se denota visualmente una agrupacion perfecta entre variables que esten correclacionadas es decir existe mucha dispersion
###en los graficos no existe tendencia entre variables
###Al meter la comparacion respecto a la variable energia tampoco se nota tendencia con otras variables 


##d) Para dos variables identifique los datos atıpicos, si los hay.
boxplots = datos.boxplot(return_type='axes')
boxplots = datos[['danceability','energy']].boxplot(return_type='axes')
###El boxplot con todas las variables resulta complicado de interpretar, especialmente al haber una variable con datos muy dispersos
###Al delimiatrlo a dos variables(danceability y energia) se muestra que existen dos datos atipicos bajos en danceability en energia no hay


##e) Calcule la matriz de correlaciones
corr = datos.corr()
print(corr)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
###la matriz de correlaciones indica el nivel de relacion que existe entre las variables de un problema. Entre mas correlacionadas estan
###mas similar resulta su comportamiento dentro del problema. Aqui debe notarse que entre mayor sea su nivel de correlacion mas parecido
###sera su comportamiento. Entre mas negativo sea resulta mas opuesto su comportamiento. La gran correlacion puede indicar tambien 
###redundancia o agrupacion de variables. En este caso parece que la mayoria de las variables estan inversamente correlacionadas. 
###Y el nivel de correlacion entre variables es bastante bajo.

    
##f ) Efectue un ACP y de una interpretacion 
import matplotlib.pyplot as plt
from prince import PCA

class ACP:
    def __init__(self, datos, n_componentes = 5): 
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
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
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
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
                

acp = ACP(datos,n_componentes=3)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal()
acp.plot_circulo()
acp.plot_sobreposicion()

acp.plot_plano_principal(ejes = [0, 2])
acp.plot_circulo(ejes = [0, 2])
acp.plot_sobreposicion(ejes = [0, 2])
###En este caso se genera un ACP de 3 componentes. El plano principal de todas las variables muestra que existe una gran dispersion de los datos,
###pero a su vez el componente 0 solo explica 21,03% de la variabilidad del modelo y el componente 1 solo un 17,85% adicional. Es decir,
###Es decir, el modelo no llega a explicar gran parte de la variabilidad en ambos componentes.

###Asi mismo el circulo de correlacion, muestra la correlacion entre variables de forma visual. En el ACP, la correlacion muestra una dificil 
###agrupacion entre variables. Parece haber una relacion entre energia y loudness, podria considerarse agrupar instrumentalness y duration_ms, 
###asi como tempo y liveness. Finalmente los restantes en un tercer factor.

###El plano de sobre-posicion muestra un plano visual bastante curioso se denota que gran parte de las canciones estan orientadas hacia la derecha del plano,
###contrario a las variables descriptivas, lo que indica una relacion mas inversa.

###Ahora bien en el plano del ajuste de componentes 1 y 3, la estimacion, explica aun menos variabilidad, el componente 0 aun 21,03%, y
###y el componente 2 un 14,14%. La dispersion de las variables en el circulo de correlacion se vuelve aun mas compleja. 

###Viendo el grafico de sobreposicion, tomamos primero In my Feelings, este se encuentra destacando y relacionado mas con liveness e
###pero poco relacionado con otras variables. Candy Paint se relaciona mas con tempo y instrumentalness, Humble, In my Mind y Havana son
###canciones mas relacionadas con el promedio de canciones, y se puede relacionar en terminos medios con las siguientes variables.

    

#Ejercicio 2
##En este ejercicio vamos a usar los datos TablaAffairs.csv

##a) Calcule el resumen numerico, interprete los resultados para una variable.
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("TablaAffairs.csv", delimiter = ';', decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

print(datos.dropna().describe())
print(datos.describe())
print(datos.mean(numeric_only=True))
print(datos.median(numeric_only=True))
print(datos.std(numeric_only=True))
print(datos.max(numeric_only=True))
###En este caso analizando edad, en este caso es una variable ordinal, promedio de edad muestreado de 32,49 anos, hay 601 casos 
###muestreados. Existe una desviacion estandar de 9,29 anos, lo cual indica en general una amplia variedad de edades, ya que se asume casado
###encima de 18 anos
### El minima de 17,5 anos y 57 anos maximo.


##b) Calcule la matriz de correlaciones
import matplotlib.pyplot as plt
import numpy as np
corr = datos.corr()
print(corr)
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
###la matriz de correlaciones indica el nivel de relacion que existe entre las variables de un problema. Entre mas correlacionadas estan
###mas similar resulta su comportamiento dentro del problema. Aqui debe notarse que entre mayor sea su nivel de correlacion mas parecido
###sera su comportamiento. Entre mas negativo sea resulta mas opuesto su comportamiento. La gran correlacion puede indicar tambien 
###redundancia o agrupacion de variables. En este caso parece que las variables estan positivamente correlacionadas todas entre por
###ejemplo parece haber una fuerte correlacion entre anos casado y edad.


##c) Usando solo las variables numericas efectue un ACP 
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)

acp = ACP(datos_dummy,n_componentes=4)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal()
acp.plot_circulo()
acp.plot_sobreposicion()
###En este caso se genera un ACP de 4 componentes. El plano principal de todas las variables muestra que existe una gran dispersion de los datos,
###pero a su vez el componente 0 solo explica 21,81% de la variabilidad del modelo y el componente 1 solo un 17,55% adicional. Es decir,
###Es decir, el modelo no llega a explicar gran parte de la variabilidad en ambos componentes.

###Asi mismo el circulo de correlacion, muestra la correlacion entre variables de forma visual. En el ACP, la correlacion muestra una dificil 
###agrupacion entre variables en cuatro puntos del plano, al menos de forma visual. Parece haber una relacion entre por ejemplo genero_female semuestra como un solo componente
###educacion, genero_male y ocupacion otro, Valoracion_muyfeliz y hijos_no, un tercer componente. El cuarto componente se desprende del resto.

###El plano de sobre-posicion muestra un plano visual con datos muy dispersos que cabe destacar que se acumulan hacia los cuatro componentes. Cabe senalar que se ven bastante separados los cuatro clusteres




##d) Ahora convierta las variables Genero e Hijos en Codigo Disyuntivo Completo y repita el ACP
# Recodificando la variable usando texto y luego se convierten a variables Dummy
def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod

datos["Genero"] = recodificar(datos["Genero"], {'male':1,'female':2})
datos["Hijos"] = recodificar(datos["Hijos"], {'no':1,'yes':2})
print(datos.head())
print(datos.dtypes)
# Conviertiendo la variables en Dummy
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)

acp = ACP(datos_dummy,n_componentes=4)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal()
acp.plot_circulo()
acp.plot_sobreposicion()
###En este caso, convirtiendo las variables Genero e Hijos en Codigo Disyuntivo Completo, se pierde interpretabilidad. El modelo explica
###aun menos variabilidad, y visualmente se vuelve mas complejo definir agrupacion de datos.


#Ejercicio 3
##En este ejercicio vamos a realizar un ACP para la tabla SAheart.csv
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("SAheart.csv", delimiter = ';', decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

##a) Efectue un ACP usando solo las variables numericas
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)

acp = ACP(datos_dummy,n_componentes=4)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal()
acp.plot_circulo()
acp.plot_sobreposicion()
###En este caso se genera un ACP de 3 componentes, ya que por la tendencia de los datos se puede notar perfectamente que los datos se 
###agrupan en tres clusters en forma de ovoide. El plano principal de todas las variables muestra que existe una gran dispersion de los datos,
###pero a su vez el componente 0 solo explica 32,09% de la variabilidad del modelo y el componente 1 solo un 15,8% adicional, mejor que en los ejercicios anteriores. Es decir,
###Es decir, el modelo no llega a explicar gran parte de la variabilidad en ambos componentes.

###Asi mismo el circulo de correlacion, muestra la correlacion entre variables de forma visual. En el ACP, la correlacion muestra una dificil 
###agrupacion entre variables en tres puntos del plano, al menos de forma visual. Parece haber una relacion entre obesity, age, ldl, tobacco y adiposity.
###Otro, parece surgir de famhist_Absent y chd_No. Los restantes en otra agrupacion caracteristica.

###El plano de sobre-posicion muestra un plano visual con datos dispersos en tres clusters que cabe destacar que se acumulan hacia los varaiables en tres componentes caracteristicos. 





##b) Efectue un ACP usando las variables num´ericas y las variables categoricas
def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod

datos["famhist"] = recodificar(datos["famhist"], {'Present':1,'Absent':2})
print(datos.head())
print(datos.dtypes)
# Conviertiendo la variables en Dummy
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)

acp = ACP(datos_dummy,n_componentes=3)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal()
acp.plot_circulo()
acp.plot_sobreposicion()
###En este caso se genera un ACP de 2 componentes, ya que por la tendencia de los datos se puede notar perfectamente que los datos se 
###agrupan en dos clusters en forma de ovoide. El plano principal de todas las variables muestra que existe una gran dispersion de los datos,
###pero a su vez el componente 0 solo explica 32,86% de la variabilidad del modelo y el componente 1 solo un 15,84% adicional, mejor que en el ejercicio anterior. Es decir,
###Es decir, el modelo no llega a explicar gran parte de la variabilidad en ambos componentes, pero en este caso resulta aunque no
###significativa la mejor en estimacion de variabilidad, si la estimacion con dos componentes se ve optimizada, asi como por consiguiente la interpretacion de resultados.

###Asi mismo el circulo de correlacion, muestra la correlacion entre variables de forma visual. En el ACP, la correlacion muestra una dificil 
###agrupacion entre variables en dos puntos del plano, al menos de forma visual. Existe relacion inversa entre variables en un componente por ejemplo. (chd_si y chd_no)

###El plano de sobre-posicion muestra un plano visual con datos dispersos en dos clusters 



#Ejercicio 4
##Programe una clase derivada (que herede) de la clase class ACP
##a) Que sobrecargue el constructor de la clase init para seleccionar variables
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("SpotifyTop2018_40_V2.csv", decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)

import matplotlib.pyplot as plt
from prince import PCA

class ACP:
    def __init__(self, datos, n_componentes = 5): 
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
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
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
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')


class ACP_sobrecaragado(ACP):
    def __init__(self, datos, n_componentes = 3,colum_eliminar=[]): 
        self.__datos = datos
        self.__colum_eliminar = colum_eliminar
        for col in self.__colum_eliminar:
            del self.__datos[col]
        self.__modelo = PCA(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations(datos)
        self.__coordenadas_ind = self.__modelo.row_coordinates(datos)
        self.__contribucion_ind = self.__modelo.row_contributions(datos)
        self.__cos2_ind = self.__modelo.row_cosine_similarities(datos)
        self.__var_explicada = [x * 100 for x in self.__modelo.explained_inertia_]
    @property
    def vector(self):
       return self.__vector
    @vector.setter
    def vector(self, nuevo):
       self.__vector = nuevo
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, nuevo):
        self.__datos = nuevo

##b) Que sobrecargue los metodos plot plano principal y plot sobreposicion 
class ACP_princ_sobre_sobracaragado(ACP):
    def plot_plano_principal_sobre(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('seaborn-whitegrid')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
    def plot_sobreposicion_sobre(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('seaborn-whitegrid')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')

##c) Verifique la nueva clase programada con los datos del ejercicio 1.
acp = ACP(datos,n_componentes=3)
print(acp.coordenadas_ind)
print(acp.cos2_ind)
print(acp.correlacion_var)               

acp.plot_plano_principal_sobre()
acp.plot_circulo()
acp.plot_sobreposicion_sobre()




