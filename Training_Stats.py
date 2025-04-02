#from Lisis_ML import data_manager_v2 as dm
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

def df_maker_stats(directory):
    directory = Path(directory)
    data = {'phi':[0], 'mean':[0], 'std':[0], 'skewness':[0], 'kurtosis':[0], 'min':[0], 'max':[0], 'arrested':[0]}
    data_df = pd.DataFrame(data)

    files = directory.iterdir()
    for file in files:
        name = file.name

        with file.open('r') as f:
            data = np.loadtxt(f)
            f.close()
    
        mean = np.mean(data[1])
        std = np.std(data[1])
        skew = pd.Series(data[1]).skew()
        kurt = pd.Series(data[1]).kurt()

        y = arrested_or_not(name)
        ph = phi_value(name)
        phi = float(ph)

        dic_auxiliar = {'min':min(data[0]), 'max':max(data[1])}
        df_aux = pd.DataFrame(dic_auxiliar, index=[0])

        df_aux['phi'] = phi
        df_aux['arrested'] = y
        df_aux['mean'] = mean
        df_aux['std'] = std
        df_aux['skewness'] = skew
        df_aux['kurtosis'] = kurt

        data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df

directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_v5') # en windows
#arch_test = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_v5\\phi_0.272_Arrest_false_.dat')directory = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_v5') # en windows
#arch_test = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_v5\\phi_0.272_Arrest_false_.dat')

file = arch_test
with file.open('r') as f:
            data = np.loadtxt(f)
            f.close()

print(type(data[1]))
mean = np.mean(data[1])
print(mean)
print(type(pd.Series(data[1])))
skew = pd.Series(data[1]).skew()
print(skew)

df_data = df_maker_stats(directory)
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

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='newton-cg', max_iter=1000)

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_val)

print("F1 score:", f1_score(Y_val, y_pred, pos_label='no'))
print("accuracy: ", accuracy_score(Y_val, y_pred))
from joblib import dump

#ump(clf, 'class_stats.joblib')

from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf', gamma=0.02, C=20)
svm_clf.fit(X_train, Y_train)

y_pred_svm = svm_clf.predict(X_test)
print("F1 score:", f1_score(Y_test, y_pred_svm, pos_label='no'))

#ump(svm_clf, 'class_stats_svm.joblib')

from joblib import load

#lf = load('class_stats.joblib')
#lf_svm = load('class_stats_svm.joblib')

predicciones_clf = clf.predict(df_copy)
file = np.loadtxt('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\diagrama_de_arresto_WCA.dat')

lista_y = []
for i in range(len(file)):
    exp = file[i][1]
    
    lista_y.append(exp)
    

lista_x = []
for i in range(len(file)):
    el = file[i][0]
    lista_x.append(el)

    plt.rc('text', usetex = True)

for i in range(len(predicciones_clf)-1):
    if predicciones_clf[i] == 'no':
        plt.scatter(phi_data.iloc[i], tem_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_clf[i] == 'yes':
        plt.scatter(phi_data.iloc[i], tem_data.iloc[i], marker='o', c='red', linewidths = 0.5)
#plt.plot(lista_x2, lista_y2, c='green')
plt.plot(lista_x, lista_y, c='black')
#plt.plot(lista_x3, lista_y3, c='green')
#plt.plot(lista_x_55, lista_y_55, c='black')
#plt.plot(lista_x_5, lista_y_5, c='yellow', lw=2)
#plt.plot(lista_x_45, lista_y_45, c='orange', lw=2)

plt.xlabel(r'$\phi$')
plt.ylabel(r'$T$')
plt.xlim(0.55, 0.9)
plt.ylim(0, 10)
plt.show()

predicciones_svm = clf_svm.predict(df_copy)

plt.rc('text', usetex = True)
for i in range(len(predicciones_svm)-1):
    if predicciones_svm[i] == 'no':
        plt.scatter(phi_data.iloc[i], tem_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        plt.scatter(phi_data.iloc[i], tem_data.iloc[i], marker='o', c='red', linewidths = 0.5)
#plt.plot(lista_x2, lista_y2, c='green')
plt.plot(lista_x, lista_y, c='black')
#plt.plot(lista_x3, lista_y3, c='green')
#plt.plot(lista_x_55, lista_y_55, c='black')
#plt.plot(lista_x_5, lista_y_5, c='yellow', lw=2)
#plt.plot(lista_x_45, lista_y_45, c='orange', lw=2)

plt.xlabel(r'$\phi$')
plt.ylabel(r'$T$')
plt.xlim(0.55, 0.9)
plt.ylim(0, 10)
plt.show()

# extraemos las caracteristicas en variables separadas
phi_data = df_copy['phi'].copy()
mean_data = df_copy['mean'].copy()
std_data = df_copy['std'].copy()
skew_data = df_copy['skewness'].copy()
kurt_data = df_copy['kurtosis'].copy()
min_data = df_copy['min'].copy()
max_data = df_copy['max'].copy()

plt.rc('text', usetex = True)
for i in range(0, len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        plt.scatter(std_data.iloc[i], mean_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        plt.scatter(std_data.iloc[i], mean_data.iloc[i], marker='o', c='red', linewidths = 0.5)
#plt.plot(lista_x2, lista_y2, c='green')
#plt.plot(lista_x, lista_y, c='black')
#plt.plot(lista_x3, lista_y3, c='green')
#plt.plot(lista_x_55, lista_y_55, c='black')
#plt.plot(lista_x_5, lista_y_5, c='yellow', lw=2)
#plt.plot(lista_x_45, lista_y_45, c='orange', lw=2)

plt.xlabel(r'$std$')
plt.ylabel(r'$mean$')
#plt.xlim(0.55, 0.9)
#plt.ylim(0, 10)
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(15, 10))
# Grafica de phi contra el phi
for i in range(0,len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[0, 0].scatter(phi_data.iloc[i], phi_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[0, 0].scatter(phi_data.iloc[i], phi_data.iloc[i], marker='o', c='red', linewidths = 0.5)

# Grafica de phi contra el mean
for i in range(0,len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[0, 1].scatter(phi_data.iloc[i], mean_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[0, 1].scatter(phi_data.iloc[i], mean_data.iloc[i], marker='o', c='red', linewidths = 0.5)

# Grafica de phi contra el std
for i in range(0, len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[1, 0].scatter(phi_data.iloc[i], std_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[1, 0].scatter(phi_data.iloc[i], std_data.iloc[i], marker='o', c='red', linewidths = 0.5)

# Grafica de phi contra el skew
for i in range(0, len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[1, 1].scatter(phi_data.iloc[i], skew_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[1, 1].scatter(phi_data.iloc[i], skew_data.iloc[i], marker='o', c='red', linewidths = 0.5)

# Grafica de phi contra el kurt
for i in range(0, len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[2, 0].scatter(phi_data.iloc[i], kurt_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[2, 0].scatter(phi_data.iloc[i], kurt_data.iloc[i], marker='o', c='red', linewidths = 0.5)

# Grafica de phi contra el min
for i in range(0, len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[2, 1].scatter(phi_data.iloc[i], min_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[2, 1].scatter(phi_data.iloc[i], min_data.iloc[i], marker='o', c='red', linewidths = 0.5)

# Grafica de phi contra el max
for i in range(0, len(predicciones_svm), 5):
    if predicciones_svm[i] == 'no':
        axes[3, 0].scatter(phi_data.iloc[i], max_data.iloc[i], marker='*', c='blue', linewidths= 0.5)
    elif predicciones_svm[i] == 'yes':
        axes[3, 0].scatter(phi_data.iloc[i], max_data.iloc[i], marker='o', c='red', linewidths = 0.5)


axes[0, 0].set_xlabel('phi')
axes[0, 0].set_ylabel('phi')
axes[0, 1].set_xlabel('phi')
axes[0, 1].set_ylabel('mean')
axes[1, 0].set_xlabel('phi')
axes[1, 0].set_ylabel('std')
axes[1, 1].set_xlabel('phi')
axes[1, 1].set_ylabel('skew')
axes[2, 0].set_xlabel('phi')
axes[2, 0].set_ylabel('kurt')
axes[2, 1].set_xlabel('phi')
axes[2, 1].set_ylabel('min')
axes[3, 0].set_xlabel('phi')
axes[3, 0].set_ylabel('max')

#plt.xlim(0.55, 0.9)
#plt.ylim(0, 10)
plt.tight_layout()
plt.show()