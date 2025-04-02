import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

#import arff
import pandas as pd

#def column_separater(df, df_col_as_np, lon_min, name):
#    for array in df_col_as_np:
#        df_aux = pd.DataFrame()
#        df_ax = pd.DataFrame()
#        for ele in range(lon_min):
#            columna = f'{name}_{ele}'
#            value = array[ele]
#            df_ax[columna] = value
#            df_aux = pd.concat([df_aux, df_ax], axis=1)
#        df = pd.concat([df, df_aux], axis=0, ignore_index=True)
#    return df

""" df_maker2(directory)
funcion que forma y devuelve un DataFrame con 4 columnas, a saber: [phi, k, S, arrested], a partir de los archivos .dat en un directorio
toma como apoyo las funciones 'arrested_or_not' y 'phi_value'

# Argumentos:
* `directory`: directorio donde se encuentran los archivos .dat de la base de datos

#Returns:
* `data_df`: DataFrame con 4 columnas ordenadas

"""

def df_maker2(directory):
    directory = Path(directory)
    data = {'phi':[0], 'k':[0], 'S':[0], 'vis':[0]}
    data_df = pd.DataFrame(data)

    files = directory.iterdir()
    for file in files:
        name = file.name

        with file.open('r') as f:
            data = np.loadtxt(f)
            f.close()
        
        dic_auxiliar = {'k':[data[0]], 'S':[data[1]]}
        df_aux = pd.DataFrame(dic_auxiliar)

        ph = phi_value(name)
        phi = float(ph)

        #vs = vis_value(name)
        vs = float(vis_value2(name))
        #vis_i = vs
        vis_i = np.log10(vs)
        #vis_i = 1 / float(vs)

        df_aux['phi'] = phi
        df_aux['vis'] = vis_i

        data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df

def tem_value(name):
    list = name.split('_')
    tem = list[3]
    return tem

def df_maker_WCA(directory):
    directory = Path(directory)
    data = {'phi':[0], 'T':[0], 'k':[0], 'S':[0], 'vis':[0]}
    data_df = pd.DataFrame(data)

    files = directory.iterdir()
    for file in files:
        name = file.name

        with file.open('r') as f:
            data = np.loadtxt(f)
            f.close()
        
        dic_auxiliar = {'k':[data[0]], 'S':[data[1]]}
        df_aux = pd.DataFrame(dic_auxiliar)

        y = tem_value(name)
        ph = phi_value(name)
        phi = float(ph)
        tem = float(y)

         #vs = vis_value(name)
        vs = float(vis_value2(name))
        #vis_i = vs
        vis_i = np.log10(vs)
        #vis_i = 1 / float(vs)

        df_aux['phi'] = phi
        df_aux['T'] = tem
        df_aux['vis'] = vis_i

        data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df
""" phi_value(name)
funcion que devuelve el valor de phi a partir de un nombre de algun archivo
archivo -> corresponde a los elementos de la base de datos que se genera con un archivo en julia

# Argumentos:
* `name`: nombre del archivo

# Returns:
* `phi`: float64 que se encuentra en el nombre del archivo
"""

def phi_value(name):
    list = name.split('_')
    phi = list[1]
    return phi


""" arrested_or_not(name)
funcion que devuelve 'yes' or 'no' si se encuentra 'true' en un nombre de un archivo
archivo -> corresponde a los elementos de la base de datos que se genera con un archivo en julia
los archivos deben contener 'true' or 'false'

# Argumentos:
* `name`: nombre del archivo

# Returns:
* `y`: objeto tipo str 'yes' o 'no'

"""

def arrested_or_not(name):
    if 'true' in name:
        y = 'yes'
    else:
        y = 'no'
    return y

""" df_separater(columnas, array)
funcion que toma un array numpy y lo converte en un DataFrame con los elementos separados en columnas segun el
input columnas

# Argumentos:
* `columnas`: es una lista donde se encuentran las columnas que tendra el nuevo DF
* `array`: es el array que se va a separar, el array puede ser un array exterior con elementos array

# Returns:
* `df_out`: es el DataFrame que se crea con las columnas segun el argumento columnas y cuyos datos son los del
array
"""

def df_separater2(columnas, array, df):
    df_aux = pd.DataFrame(array, columns=columnas)
    df_out = pd.concat([df_aux, df], axis=1)
    return df_out

""" redimens(elem, array)
funcion que toma un array con elemetons tipo array para redimensionar al mismo numero de entradas

# Argumentos:
* `elem`: es el numero de entradas objetivo para cada array
* `array`: array tipo numpy con elementos tipo array cuya dimensionalidad no es igual para todos

# Returns:
* `n_object`: array tipo numpy con elementos tipo array cuya dimensionalidad es la misma para todos

"""

def redimens(elem, array):# duvuelve un objeto numpy
    new_array = []
    n_object = []
    # elem es el numero de elementos minimo
    for e in array:

        for i in range(elem):
            new_array.append(e[i])
        new_array = np.array(new_array)
        n_object.append(new_array)
        new_array = []
    
    n_object = np.array(n_object)
    
    return n_object

""" listador_columnas(elem, name_ext)
funcion que crea una lista con el numero de elementos necesarias segun la variable elem

# Argumentos:
* `elem`: el numero de elementos que contendra la lista
* `name_ext`: nombre base de los elementos (el proposito de la lista es contener los nombres de columnas)

# Returns:
* `columnas`: lista que sus elementos seran los nombres de columnas de algun dataframe
"""

def listador_columnas(elem, name_ext):
    columnas = []
    for i in range(elem):
        name = f'{name_ext}_{i}'
        columnas.append(name)
        name = name_ext

    return columnas

""" min_element_2(numpy_array)
funcion que encuentra el numero minimo de elementos en los elementos array de un array externo

# Argumentos:
* `numpy_array`: array numpy externo el cual sus elemetos son arrays

# Returns:
* `min_long`: numero minimo de elementos
"""

def min_element_2(numpy_array):
    lista = []
    for i in range(len(numpy_array)):
        elem = len(numpy_array[i])
        lista.append(elem)

    min_long = min(lista)
    return min_long


def vis_value(name):
    list = name.split('_')
    if float(list[1]) < 0.58:
        
        vis = list[3]
    else:
        vis = np.inf
    return vis

def vis_value2(name):
    lista = name.split('_')
    vis = lista[5]
    return vis

def number_or_inf(vs):
    if vs == 'Inf':
        vis = np.inf
    else:
        vis = float(vs)
    return vis

def df_separater(columnas, array):
    df_out = pd.DataFrame(array, columns=columnas)
    
    return df_out

def df_separater2(columnas, array, df1, df2):
    df_aux = pd.DataFrame(array, columns=columnas)
    df_prev = pd.concat([df_aux, df1], axis=1)
    df_out = pd.concat([df_prev, df2], axis=1)
    return df_out

#win
#directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBases_WCA_visc\\DataBase_WCA_vis_T5')
#directory = Path('/home/jasanchez/MEGA/NESCGLE_scripts/DateBase/DataBases_WCA_visc/DataBase_WCA_vis_T5')

#directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_SW_vis\\DataBase_SW_vis_T1p8')
directory = Path('/home/jasanchez/MEGA/NESCGLE_scripts/DateBase/DataBase_SW_vis/DataBase_SW_vis_T1p8')

df_data = df_maker_WCA(directory)
df_data.drop(0, inplace=True)
df_copy = df_data.copy()

condicion = df_copy[df_copy['vis'] > 8.0].index

df_copy = df_copy.drop(condicion)

data = df_copy.drop(['k', 'vis', 'T', 'phi'], axis=1)

Y_real = df_copy['vis'].copy()
dataphi = df_copy['phi'].copy()
dataT = df_copy['T'].copy()

dataphi_np = dataphi.to_numpy()
dataT_np = dataT.to_numpy()
dataphi_S = pd.Series(dataphi_np, name='phi')
dataT_S = pd.Series(dataT_np, name='T')

columnas = listador_columnas(472, 'S')

data_np = data['S'].to_numpy()
Rdata = redimens(472, data_np)
print(f'Ahora el tamagno es: {Rdata.shape}')
#df_pred = df_separater(columnas, Rdata)
df_pred = df_separater2(columnas, Rdata, dataphi_S, dataT_S)

import joblib

#lasso = joblib.load('log_viscosidades_senza_scaler.joblib')
#ridge = joblib.load('log_rdg_viscosidades_senza_scaler.joblib')

lasso = joblib.load('log_viscosidades_senza_noscaler.joblib')
ridge = joblib.load('log_rdg_viscosidades_senza_noscaler.joblib')
#lassoWCA_senza = joblib.load('log_viscosidadesWCA_noscaler_senza.joblib')
lassoWCA = joblib.load('log_viscosidadesWCA_imagene.joblib')
ridgeWCA = joblib.load('log_viscosidadesWCA_imagene_r.joblib')

lassoSW = joblib.load('log_viscosidadesSW_las.joblib')
ridgeSW = joblib.load('log_viscosidadesSWr.joblib')

lassoSW2 = joblib.load('vis_SW_scaler.joblib')
ridgeSW2 = joblib.load('vis_SWrig_scaler')
scaler = joblib.load('scaler_SW.joblib')

train_score = ridgeSW.score(df_pred, Y_real)
print(train_score)
predicciones = ridgeSW.predict(df_pred)

plt.rc('text', usetex = True)

plt.plot(df_copy['phi'][::10], Y_real[::10], 'o', markersize=1, label=r'Datos Teoricos')
plt.plot(df_copy['phi'][::10], predicciones[::10], 'o', markersize = 1, c='red', label=r'ML LASSO')
plt.xlabel(r'$\phi$', size=15)
plt.ylabel(r'$log_{10}(\eta)$', size=15)
#plt.ylim(1, 1000)
#plt.yscale('log')
plt.legend()
plt.show()

train_score_rdg = ridgeWCA.score(df_pred, Y_real)
print(train_score_rdg)
predicciones_rdg = ridgeWCA.predict(df_pred)