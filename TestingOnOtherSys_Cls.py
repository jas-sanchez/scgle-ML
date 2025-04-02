import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import arff
import pandas as pd
from sklearn.model_selection import train_test_split

def tem_value(name):
    list = name.split('_')
    tem = list[3]
    return tem

def tem_value_asym(name):
    list = name.split('_')
    tem = list[4]
    return tem

def df_maker_WCA_SW(directory):
    directory = Path(directory)
    data = {'phi':[0], 'T':[0], 'k':[0], 'S':[0]}
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

        df_aux['phi'] = phi
        df_aux['T'] = tem

        data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df

def phi_value(name):
    list = name.split('_')
    phi = list[1]
    return phi

def phi_value_asym(name):
    list = name.split('_')
    phi = list[2]
    return phi

def min_element_2(numpy_array):
    lista = []
    for i in range(len(numpy_array)):
        elem = len(numpy_array[i])
        lista.append(elem)

    min_long = min(lista)
    return min_long

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

def listador_columnas(elem, name_ext):
    columnas = []
    for i in range(elem):
        name = f'{name_ext}_{i}'
        columnas.append(name)
        name = name_ext

    return columnas

def df_separater(columnas, array):
    df_out = pd.DataFrame(array, columns=columnas)
    return df_out

def df_maker_asym(directory):
    data = {'phi':[0], 'T':[0], 'k':[0], 'S':[0]}
    data_df = pd.DataFrame(data)

    files = directory.iterdir()
    for file in files:
        name = file.name

        with file.open('r') as f:
            data = np.loadtxt(f)
            f.close()
        
        dic_auxiliar = {'k':[data[:, 0]], 'S':[data[:, 1]]}
        df_aux = pd.DataFrame(dic_auxiliar)

        y = tem_value_asym(name)
        ph = phi_value_asym(name)
        phi = float(ph)
        tem = float(y)

        df_aux['phi'] = phi
        df_aux['T'] = tem

        data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df

# directorio donde se encuentran los datos
#directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_WCA_v4') # en windows
#directory = Path("/home/jasanchez/MEGA/NESCGLE_scripts/DateBase/DataBase_WCA_v4/") # en linux

# para el pozo cuadrado
directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_SW_2')

df_data = df_maker_WCA_SW(directory)

df_data.drop(0, inplace=True)
df_copy = df_data.copy()

S_data_np = df_copy['S'].to_numpy()
print(len(S_data_np))
lon_minima = min_element_2(S_data_np)
print(lon_minima)
phi_data = df_copy['phi']
T_data = df_copy['T']

RS_data_np = redimens(440, S_data_np)

columnas = listador_columnas(440, 'S')

df_to_model = df_separater(columnas, RS_data_np)

from joblib import load

# modelo de svm
#clf_model = load('classification_svm_r.joblib')
 # se encarga de cargar el modelo que fue guardado con Training_Cls.py

predicciones = clf_model.predict(df_to_model)
predicciones_l = predicciones.tolist()
phi = phi_data.tolist()
temp = T_data.tolist()
print(len(predicciones_l))

plt.rc('text', usetex = True)

for i in range(len(predicciones_l)):
    if predicciones_l[i] == 'no':
        plt.scatter(phi[i], temp[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_l[i] == 'yes':
        plt.scatter(phi[i], temp[i], marker='o', c='red', linewidths = 0.5)

plt.xlabel(r'$\phi$')
plt.ylabel(r'$T$')
plt.xlim(0.55, 0.9)
plt.ylim(0, 10)
plt.show()