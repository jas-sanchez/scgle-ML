import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import arff
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
    data = {'phi':[0], 'k':[0], 'S':[0], 'arrested':[0]}
    data_df = pd.DataFrame(data)

    files = directory.iterdir()
    for file in files:
        name = file.name

        with file.open('r') as f:
            data = np.loadtxt(f)
            f.close()
        
        dic_auxiliar = {'k':[data[0]], 'S':[data[1]]}
        df_aux = pd.DataFrame(dic_auxiliar)

        y = arrested_or_not(name)
        ph = phi_value(name)
        phi = float(ph)

        df_aux['phi'] = phi
        df_aux['arrested'] = y

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

def df_separater(columnas, array):
    df_out = pd.DataFrame(array, columns=columnas)
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


# directorio donde se encuentran los datos
directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_v5') # en windows
#directory = Path("/home/jasanchez/Documentos/DataBase_v2/") # en linux
df_data = df_maker2(directory)
df_data.drop(0, inplace=True)
df_copy = df_data.copy()

train_set, test_set = train_test_split(df_copy, test_size=0.85, random_state=42, stratify=df_copy["arrested"])
print('Longitud train test: ', len(train_set))
print('Longitud test set: ', len(test_set))
print()
val_test, test_set = train_test_split(test_set, test_size=0.5, random_state=42, stratify=test_set['arrested'])
print('Longitud val test: ', len(val_test))
print('Longitud test set: ', len(test_set))

X_train = train_set.drop('arrested', axis=1)
Y_train = train_set['arrested'].copy()
X_val = val_test.drop('arrested', axis=1)
Y_val = val_test['arrested'].copy()
X_test = test_set.drop('arrested', axis=1)
Y_test = test_set['arrested'].copy()

X_train_set_np = X_train['S'].to_numpy()

X_train_set_np = X_train['S'].to_numpy()
lon_minima= min_element_2(X_train_set_np)
print('La longitud minima de los elementos en X_train es : ', lon_minima)
X_val_set_np = X_val['S'].to_numpy()
lon_min_val = min_element_2(X_val_set_np)
print('La longitud minima de los elementos en X_val es : ', lon_min_val)
X_test_set_np = X_test['S'].to_numpy()
lon_min_test = min_element_2(X_test_set_np)
print('La longitud minima de los elementos en X_set es : ', lon_min_test)
print(X_train_set_np)

RX_train_set_np = redimens(440, X_train_set_np)
print('La forma de X_train es: ', RX_train_set_np.shape)
RX_val_set_np = redimens(440, X_val_set_np)
print('La forma de X_val es: ', RX_val_set_np.shape)
RX_test_set_np = redimens(440, X_test_set_np)
print('La forma de X_test es: ', RX_test_set_np.shape)

columnas = listador_columnas(440, 'S')

df_train = df_separater(columnas, RX_train_set_np)
df_val = df_separater(columnas, RX_val_set_np)
df_test = df_separater(columnas, RX_test_set_np)

Y_train_np = Y_train.to_numpy()
Y_val_np = Y_val.to_numpy()
Y_test_np = Y_test.to_numpy()

Y_train_or = pd.Series(Y_train_np, name='arrested')
Y_val_or = pd.Series(Y_val_np, name='arrested')
Y_test_or = pd.Series(Y_test_np, name='arrested')


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='newton-cg', max_iter=1000)

clf.fit(df_train, Y_train_or)

y_pred = clf.predict(df_val)

print("F1 score:", f1_score(Y_val_or, y_pred, pos_label='no'))
print("accuracy: ", accuracy_score(Y_val_or, y_pred))

y_pred_test = clf.predict(df_test)

test_phi = X_test['phi'].to_numpy()
print(len(test_phi))
print(len(y_pred_test))

from joblib import dump

#dump(clf, 'classification_2.joblib')

from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf', gamma=0.02, C=30)
svm_clf.fit(df_train, Y_train_or)

y_pred_svm = svm_clf.predict(df_val)
print("F1 score:", f1_score(Y_val_or, y_pred_svm, pos_label='no'))

y_pred_svm = svm_clf.predict(df_test)
print('F1 score: ', f1_score(Y_test_or, y_pred_svm, pos_label='no'))

#dump(svm_clf, 'classification_svm_r.joblib')   estos joblib guardan los modelos