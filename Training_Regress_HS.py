import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

#import arff
import pandas as pd
from sklearn.model_selection import train_test_split

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
    vis = lista[3]
    return vis

def number_or_inf(vs):
    if vs == 'Inf':
        vis = np.inf
    else:
        vis = float(vs)
    return vis

def tem_value(name):
    list = name.split('_')
    tem = list[3]
    return tem

def df_separater(columnas, array):
    df_out = pd.DataFrame(array, columns=columnas)
    
    return df_out

def df_maker3(directory):
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

        vs = vis_value2(name)
        vis_i = float(vs)
        vis_ii = np.log10(vis_i)

        df_aux['phi'] = phi
        df_aux['vis'] = vis_ii

        data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df

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

# directorio donde se encuentran los datos
#directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_HS_vs') # en windows
#directory = Path("/home/jasanchez/Documentos/DataBase_v2/") # en linux

#directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_HS_vs_v2') # para datos entre 0.5 y 0.68

#directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_HS_vs_v4')
directory = Path('/home/jasanchez/MEGA/NESCGLE_scripts/DateBase/DataBase_HS_vs_v4')

df_data = df_maker2(directory)

df_data.drop(0, inplace=True)
df_copy = df_data.copy()

train_set, test_set = train_test_split(df_copy, test_size=0.8, random_state=42)
print('Longitud train test: ', len(train_set))
print('Longitud test set: ', len(test_set))
print()
val_test, test_set = train_test_split(test_set, test_size=0.5, random_state=42)
print('Longitud val test: ', len(val_test))
print('Longitud test set: ', len(test_set))

X_train = train_set.drop(['k', 'vis'], axis=1)
X_val = val_test.drop(['k', 'vis'], axis=1)
X_test = test_set.drop(['k', 'vis'], axis=1)

Y_train = train_set['vis'].copy()
Y_val = val_test['vis'].copy()
Y_test = test_set['vis'].copy()

columnas = listador_columnas(472, 'S')

X_train_Snp = X_train['S'].to_numpy()
X_val_Snp = X_val['S'].to_numpy()
X_test_Snp = X_test['S'].to_numpy()
df_phi_train = X_train['phi']
df_phi_val = X_val['phi']
df_phi_test = X_test['phi']

RX_train_sn = redimens(472, X_train_Snp)
print(f'Ahora el tamagno es: {RX_train_sn.shape}')
RX_val_sn = redimens(472, X_val_Snp)
RX_test_sn = redimens(472, X_test_Snp)

X_train_phinp = df_phi_train.to_numpy()
X_val_phinp = df_phi_val.to_numpy()
X_test_phinp = df_phi_test.to_numpy()

X_train_phior = pd.Series(X_train_phinp, name='phi')
X_val_phior = pd.Series(X_val_phinp, name='phi')
X_test_phior = pd.Series(X_test_phinp, name='phi')

df_train = df_separater(columnas, RX_train_sn)
#df_val = df_separater2(columnas, RX_val_sn, X_val_phior)
df_val = df_separater(columnas, RX_val_sn)
#df_test = df_separater2(columnas, RX_test_sn, X_test_phior)
df_test = df_separater(columnas, RX_test_sn)

Y_train_np = Y_train.to_numpy()
Y_val_np = Y_val.to_numpy()
Y_test_np = Y_test.to_numpy()

Y_train_or = pd.Series(Y_train_np, name='vis')
Y_val_or = pd.Series(Y_val_np, name='vis')
Y_test_or = pd.Series(Y_test_np, name='vis')

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.001)

lasso.fit(df_train, Y_train_or)
train_score = lasso.score(df_train, Y_train_or)
print(train_score)
predicciones_train = lasso.predict(df_train)
val_score = lasso.score(df_val, Y_val_or)
print(val_score)
predicciones_val = lasso.predict(df_val)
test_score = lasso.score(df_test, Y_test_or)
print(test_score)
predicciones_test = lasso.predict(df_test)
plt.rc('text', usetex = True)

plt.plot(X_test_phior[::4], Y_test_or[::4], 'o', markersize=2, label=r'Datos Teoricos')
plt.plot(X_test_phior[::4], predicciones_test[::4], 'o', markersize = 2, c='red', label=r'ML lasso')
#plt.ylim(1, 1000)
#plt.yscale('log')
plt.ylabel(r'$log_{10}(\eta)$', size = 15)
plt.xlabel(r'$\phi$', size = 15)
plt.legend()
plt.show()

from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.001)
ridge.fit(df_train, Y_train_or)
test_score_rdg = ridge.score(df_test, Y_test_or)
print(test_score_rdg)
predicciones_test_rdg = ridge.predict(df_test)
plt.rc('text', usetex = True)

plt.plot(X_test_phior[::4], Y_test_or[::4], 'o', markersize=2, label=r'Datos teóricos')
plt.plot(X_test_phior[::4], predicciones_test_rdg[::4], 'o', markersize = 2, c='red', label=r'ML ridge')
#plt.plot(X_test_phior[::4], predicciones_test[::4], 'o', markersize = 2, c='green', label=r'ML lasso')
#plt.ylim(1, 1000)
#plt.yscale('log')
plt.ylabel(r'$log_{10}(\eta)$', size = 15)
plt.xlabel(r'$\phi$', size = 15)
plt.legend()
plt.show()

from joblib import dump

#dump(lasso, 'log_viscosidades_senza_noscaler.joblib')
#dump(ridge, 'log_rdg_viscosidades_senza_noscaler.joblib')