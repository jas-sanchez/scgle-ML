from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Para encontrar los cortes en G' y G'' 

def a_greater_b(gpr, gbpr, index_stop, crosses):
    bucle = True
    for i1 in range(index_stop, len(gpr)):
            if gpr[i1] > gbpr[i1]:
                if i1 == (len(gpr) -1):
                     bucle = False
                continue
            elif gpr[i1] <= gbpr[i1]:
                #print('Ha ocurrido un cruce')
                crosses += 1
                index_stop = i1
                break
    return crosses, index_stop, bucle

def a_lesser_b(gpr, gbpr, index_stop, crosses):
    bucle = True
    for i2 in range(index_stop, len(gpr)):
            if gpr[i2] < gbpr[i2]:
                if i2 == (len(gpr)-1):
                     bucle = False
                continue
            elif gpr[i2] >= gbpr[i2]:
                #print("Ha ocurrido un cruce")
                crosses += 1
                index_stop = i2
                break
    
    return crosses, index_stop, bucle

def cruces_g_gg(gpr, gbpr):
    crosses = 0
    index_stop = 0
    bucle = True

    while bucle:
        if gpr[index_stop] > gbpr[index_stop]:
            crosses, index_stop, bucle = a_greater_b(gpr, gbpr, index_stop, crosses)
        else:
            crosses, index_stop, bucle = a_lesser_b(gpr, gbpr, index_stop, crosses)

    return crosses

def calculo_de_tan(omega, gpr, gbpr):
    list_cocientes = []
    for i in range(len(omega)):
        cociente = gpr[i] / gbpr[i]
        list_cocientes.append(cociente)

    list_cocientes = np.array(list_cocientes)

    return list_cocientes

def cruce_menor(tand, lst_crxs, indx): # lst_crxs es una lista que inicialmente esta vacia
    lista_cruces = lst_crxs
    bucle = True
    for i1 in range(indx, len(tand)):
        if tand[i1] < 1.0:
            if i1 == (len(tand) - 1):
                bucle = False
                indx_lst = i1
            continue
        elif tand[i1] >= 1.0:
            lista_cruces.append(1)
            indx_lst = i1
            break
    
    return lista_cruces, indx_lst, bucle

def cruce_mayor(tand, lst_crxs, indx):
    lista_cruces = lst_crxs
    bucle = True
    for i2 in range(indx, len(tand)):
        if tand[i2] > 1.0:
            if i2 == (len(tand) -1):
                indx_lst = i2
                bucle = False
            continue
        elif tand[i2] <= 1.0:
            lista_cruces.append(1)
            indx_lst = i2
            break
    return lista_cruces, indx_lst, bucle

def cruces_de_uno(tand):
    bucle = True
    indx_last = 0
    lista_cruces = []

    while bucle:
        if tand[indx_last] < 1.0:
            lista_cruces, indx_last, bucle = cruce_menor(tand, lista_cruces, indx_last)
        else:
            lista_cruces, indx_last, bucle = cruce_mayor(tand, lista_cruces, indx_last)

    return lista_cruces

def listador_columnas(elem, name_ext):
    columnas = []
    for i in range(elem):
        name = f'{name_ext}_{i}'
        columnas.append(name)
        name = name_ext

    return columnas

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

def df_maker_Gs(directory):
    directory = Path(directory)
    # para todos los casos phi_i = 0.1
    data = {'omega':[0], 'Gp':[0], 'Gbp':[0], 'tand':[0], 'x_1':[0], 'x_Gs':[0], 'phi_f':[0], 'T_f':[0]}
    data_df = pd.DataFrame(data)

    for ele in directory.iterdir():
        if ele.is_dir():
            for elem in ele.iterdir():
                dir_archivo = elem.joinpath('u_999_G_WCA.csv')
                three_colm_aux = pd.read_csv(dir_archivo, delimiter='\t', names=['omega', 'Gp', 'Gbp'])

                omega = three_colm_aux['omega'].to_numpy()
                gp = three_colm_aux['Gp'].to_numpy()
                gbp = three_colm_aux['Gbp'].to_numpy()

                dic_auxiliar = {'omega':[omega], 'Gp':[gp], 'Gbp':[gbp]}
                df_aux = pd.DataFrame(dic_auxiliar)

                tand = calculo_de_tan(omega, gp, gbp)
                df_aux['tand'] = [tand]

                lista_cruces = cruces_de_uno(tand)
                df_aux['x_1'] = len(lista_cruces)

                cruces_g_gg_num = cruces_g_gg(gp, gbp)
                df_aux['x_Gs'] = cruces_g_gg_num

                phi_local = str(dir_archivo.parents[0].name)
                T_f_local = str(dir_archivo.parents[1].name)
                phi_aux = float(phi_local[4:].replace('p', '.'))
                T_f_aux = float(T_f_local[3:6].replace('p', '.'))
                df_aux['phi_f'] = phi_aux
                df_aux['T_f'] = T_f_aux

                data_df = pd.concat([data_df, df_aux], ignore_index=True)

    return data_df

def df_separater2(columnas, array, df1, df2):
    df_aux = pd.DataFrame(array, columns=columnas)
    df_prev1 = pd.concat([df_aux, df1], axis=1)
    df_out = pd.concat([df_prev1, df2], axis=1)
    #df_prev2 = pd.concat([df_prev1, df2], axis=1)
    #df_prev3 = pd.concat([df_prev2, df3], axis=1)
    #df_out = pd.concat([df_prev3, df4], axis=1)

    return df_out

# Debemos tener accecibles a G' y G'' 

directory = 'C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DataBase\\WCA'
df_main = df_maker_Gs(directory)
df_main.drop(0, inplace=True)
df_copy = df_main.copy()

train_set, test_set = train_test_split(df_copy, test_size=0.4, random_state=42)
print('Longitud train test: ', len(train_set))
print('Longitud test set: ', len(test_set))
print()
val_test, test_set = train_test_split(test_set, test_size=0.5, random_state=42)
print('Longitud val test: ', len(val_test))
print('Longitud test set: ', len(test_set))

X_train = train_set.drop(['omega', 'Gp', 'Gbp'], axis=1)
X_val = val_test.drop(['omega', 'Gp', 'Gbp'], axis=1)
X_test = test_set.drop(['omega', 'Gp', 'Gbp'], axis=1)

columnas = listador_columnas(86, 'tand')

X_train_tannp = X_train['tand'].to_numpy()
X_val_tannp = X_val['tand'].to_numpy()
X_test_tannp = X_test['tand'].to_numpy()
df_x_1_train = X_train['x_1']
df_x_1_val = X_val['x_1']
df_x_1_test = X_test['x_1']
df_x_Gs_train = X_train['x_Gs']
df_x_Gs_val = X_val['x_Gs']
df_x_Gs_test = X_test['x_Gs']

df_phi_f_train = X_train['phi_f']
df_phi_f_val = X_val['phi_f']
df_phi_f_test = X_test['phi_f']

df_T_f_train = X_train['T_f']
df_T_f_val = X_val['T_f']
df_T_f_test = X_test['T_f']
print(f'Ahora el tamagno es: {X_train_tannp.shape}')

RX_train_tan = redimens(86, X_train_tannp)
print(f'Ahora el tamagno es: {RX_train_tan.shape}')
RX_val_tan = redimens(86, X_val_tannp)
RX_test_tan = redimens(86, X_test_tannp)
len(RX_train_tan)

X_train_x_1np = df_x_1_train.to_numpy()
X_val_x_1np = df_x_1_val.to_numpy()
X_test_x_1np = df_x_1_test.to_numpy()

X_train_x_Gsnp = df_x_Gs_train.to_numpy()
X_val_x_Gsnp = df_x_Gs_val.to_numpy()
X_test_x_Gsnp = df_x_Gs_test.to_numpy()

X_train_phi_fnp = df_phi_f_train.to_numpy()
X_val_phi_fnp = df_phi_f_val.to_numpy()
X_test_phi_fnp = df_phi_f_test.to_numpy()

X_train_T_fnp = df_T_f_train.to_numpy()
X_val_T_fnp = df_T_f_val.to_numpy()
X_test_T_fnp = df_T_f_test.to_numpy()

X_train_x_1or = pd.Series(X_train_x_1np, name='x_1')
X_val_x_1or = pd.Series(X_val_x_1np, name='x_1')
X_test_x_1or = pd.Series(X_test_x_1np, name='x_1')
X_train_x_Gsor = pd.Series(X_train_x_Gsnp, name='x_Gs')
X_val_x_Gsor = pd.Series(X_val_x_Gsnp, name='x_Gs')
X_test_x_Gsor = pd.Series(X_test_x_Gsnp, name='x_Gs')

X_train_phi_for = pd.Series(X_train_phi_fnp, name='phi_f')
X_val_phi_for = pd.Series(X_val_phi_fnp, name='phi_f')
X_test_phi_for = pd.Series(X_test_phi_fnp, name='phi_f')

X_train_T_for = pd.Series(X_train_T_fnp, name='T_f')
X_val_T_for = pd.Series(X_val_T_fnp, name='T_f')
X_test_T_for = pd.Series(X_test_T_fnp, name='T_f')

#df_train = df_separater2(columnas, RX_train_tan, X_train_x_1or, X_train_x_Gsor, X_train_phi_for, X_train_T_for)
df_train = df_separater2(columnas, RX_train_tan, X_train_x_1or, X_train_x_Gsor)

#df_val = df_separater2(columnas, RX_val_tan, X_val_x_1or, X_val_x_Gsor, X_val_phi_for, X_val_T_for)
df_val = df_separater2(columnas, RX_val_tan, X_val_x_1or, X_val_x_Gsor)

#df_test = df_separater2(columnas, RX_test_tan, X_test_x_1or, X_test_x_Gsor, X_test_phi_for, X_test_T_for)
df_test = df_separater2(columnas, RX_test_tan, X_test_x_1or, X_test_x_Gsor)

scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_val_scaled = scaler.transform(df_val)
df_test_scaled = scaler.transform(df_test)

pca = PCA()

pca.fit(df_train_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.xlabel('dimensions')

plt.plot(cumsum)

pca = PCA(n_components=9)
df_train_reduced = pca.fit_transform(df_train_scaled)
df_val_reduced = pca.transform(df_val_scaled)
df_test_reduced = pca.transform(df_test_scaled)

k = 3
kmeans = KMeans(n_clusters=k)

y_pred = kmeans.fit_predict(df_train_reduced)
y_pred_val = kmeans.predict(df_val_reduced)
y_pred_test = kmeans.predict(df_test_reduced)

plt.rc('text', usetex = True)

for i in range(len(y_pred)):
    if y_pred[i] == 0:
        plt.plot(X_train_phi_for[i], X_train_T_for[i], marker='*', c='blue')
    elif y_pred[i] == 1:
        plt.plot(X_train_phi_for[i], X_train_T_for[i], marker='*', c='green')
    elif y_pred[i] == 2:
        plt.plot(X_train_phi_for[i], X_train_T_for[i], marker='*', c='red')

for i in range(len(y_pred_val)):
    if y_pred_val[i] == 0:
        plt.plot(X_val_phi_for[i], X_val_T_for[i], marker='*', c='blue')
    elif y_pred_val[i] == 1:
        plt.plot(X_val_phi_for[i], X_val_T_for[i], marker='*', c='green')
    elif y_pred_val[i] == 2:
        plt.plot(X_val_phi_for[i], X_val_T_for[i], marker='*', c='red')

for i in range(len(y_pred_test)):
    if y_pred_test[i] == 0:
        plt.plot(X_test_phi_for[i], X_test_T_for[i], marker='*', c='blue')
    elif y_pred_test[i] == 1:
        plt.plot(X_test_phi_for[i], X_test_T_for[i], marker='*', c='green')
    elif y_pred_test[i] == 2:
        plt.plot(X_test_phi_for[i], X_test_T_for[i], marker='*', c='red')

plt.xlabel(r'$\phi$')
plt.ylabel(r'$T$')
plt.show()

pca_3 = PCA(n_components=3)

df_train_reduced_3 = pca_3.fit_transform(df_train_reduced)
df_val_reduced_3 = pca_3.transform(df_val_reduced)
df_test_reduced_3 = pca_3.transform(df_test_reduced)

plt.rc('text', usetex = True)

for i in range(len(y_pred)):
    if y_pred[i] == 0:
        plt.scatter(df_train_reduced_3[i][0], df_train_reduced_3[i][1], df_train_reduced_3[i][2], marker='*', c='blue')
    elif y_pred[i] == 1:
        plt.plot(df_train_reduced_3[i][0], df_train_reduced_3[i][1], df_train_reduced_3[i][2], marker='*', c='red')
    elif y_pred[i] == 2:
        plt.plot(df_train_reduced_3[i][0], df_train_reduced_3[i][2], df_train_reduced_3[i][2], marker='*', c='green')

for i in range(len(y_pred_val)):
    if y_pred_val[i] == 0:
        plt.scatter(df_val_reduced_3[i][0], df_val_reduced_3[i][1], df_val_reduced_3[i][2], marker='*', c='blue')
    elif y_pred_val[i] == 1:
        plt.plot(df_val_reduced_3[i][0], df_val_reduced_3[i][1], df_val_reduced_3[i][2], marker='*', c='red')
    elif y_pred_val[i] == 2:
        plt.plot(df_val_reduced_3[i][0], df_val_reduced_3[i][2], df_val_reduced_3[i][2], marker='*', c='green')

for i in range(len(y_pred_test)):
    if y_pred_test[i] == 0:
        plt.scatter(df_test_reduced_3[i][0], df_test_reduced_3[i][1], df_test_reduced_3[i][2], marker='*', c='blue')
    elif y_pred_test[i] == 1:
        plt.plot(df_test_reduced_3[i][0], df_test_reduced_3[i][1], df_test_reduced_3[i][2], marker='*', c='red')
    elif y_pred_test[i] == 2:
        plt.plot(df_test_reduced_3[i][0], df_test_reduced_3[i][2], df_test_reduced_3[i][2], marker='*', c='green')

#plt.xlabel(r'$\phi$')
#plt.ylabel(r'$T$')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(y_pred)):
    if y_pred[i] == 0:
        ax.scatter(df_train_reduced_3[i][0], df_train_reduced_3[i][1], df_train_reduced_3[i][2], marker='*', c='blue', linewidths= 0.5)
    elif y_pred[i] == 1:
        ax.scatter(df_train_reduced_3[i][0], df_train_reduced_3[i][1], df_train_reduced_3[i][2], marker='*', c='red', linewidths= 0.5)
    elif y_pred[i] == 2:
        ax.scatter(df_train_reduced_3[i][0], df_train_reduced_3[i][1], df_train_reduced_3[i][2], marker='*', c='green', linewidths= 0.5)

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_label('x_3')

plt.show()