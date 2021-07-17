import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la

class mi_ACP:
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
    def datos_centrados_reducidos(self):
        return self.__datos_centrados_reducidos
    @datos_centrados_reducidos.setter
    def datos_centrados_reducidos(self, datos_centrados_reducidos):
        self.__datos = datos_centrados_reducidos
    
    @property
    def matriz_correlaciones(self):
        return self.__matriz_correlaciones
    @matriz_correlaciones.setter
    def matriz_correlaciones(self, matriz_correlaciones):
        self.__matriz_correlaciones = matriz_correlaciones
    
    @property
    def valores_propios(self):
      return self.__valores_propios
    
    @property
    def vectores_propios(self):
      return self.__vectores_propios
        
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
      
    def correlacion_var(self):
        r = np.sqrt(self.valores_propios) * self.vectores_propios
        r = pd.DataFrame(r, index = self.datos.columns)
        r = r.iloc[:, np.arange(0, self.n_componentes)]
        return r    
        
    def cos2_var(self):
        return((self.correlacion_var()) ** 2)
        
    def var_explicada(self):
        r = [x/self.datos.shape[1] * 100 for x in self.valores_propios]
        return(r)
    
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
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
                
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
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
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
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
