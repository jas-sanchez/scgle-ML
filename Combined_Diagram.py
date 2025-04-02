import joblib
from pathlib import Path
import pandas as pd
import numpy as np

def phi_value(name):
    list = name.split('_')
    phi = list[1]
    return phi

def vis_value2(name):
    lista = name.split('_')
    vis = lista[3]
    return vis

def tem_value(name):
    list = name.split('_')
    tem = list[3]
    return tem

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

def df_separater(columnas, array):
    df_out = pd.DataFrame(array, columns=columnas)
    
    return df_out

def df_separater2(columnas, array, df1, df2):
    df_aux = pd.DataFrame(array, columns=columnas)
    df_prev = pd.concat([df_aux, df1], axis=1)
    df_out = pd.concat([df_prev, df2], axis=1)
    return df_out

def df_maker2(archivo):

    name = archivo.name

    with archivo.open('r') as f:
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

    return df_aux

def df_maker_WCA(archivo):

    name = archivo.name

    with archivo.open('r') as f:
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

    return df_aux

# funciones

def s_de_k(archivo, modelo):
    if modelo == 'WCA':
        df_full = df_maker_WCA(archivo)

        data = df_full.drop(['k', 'vis', 'phi', 'T'], axis=1)
        dataphi = df_full['phi'].copy()
        dataT = df_full['T'].copy()
        dataphi_np = dataphi.to_numpy()
        dataT_np = dataT.to_numpy()
        dataphi_S = pd.Series(dataphi_np, name='phi')
        dataT_S = pd.Series(dataT_np, name='T')
        columnas = listador_columnas(472, 'S')
        data_np = data['S'].to_numpy()
        Rdata = redimens(472, data_np)
        #df_pred = df_separater(columnas, Rdata)
        df_pred = df_separater2(columnas, Rdata, dataphi_S, dataT_S)
    else:
        df_full = df_maker2(archivo)

        data = df_full.drop(['k', 'vis', 'phi'], axis=1)
        columnas = listador_columnas(472, 'S')
        data_np = data['S'].to_numpy()
        Rdata = redimens(472, data_np)
        df_pred = df_separater(columnas, Rdata)

    return df_pred

def flux_development(df_S, modelo):
    global clf_model
    global ridge
    global lasso
    global lassoWCA
    global ridgeWCA

    if modelo == "WCA":
        df_arr = df_S.drop(['phi', 'T'], axis=1)
        prediccion = clf_model.predict(df_arr)
        prediccion_l = prediccion.tolist()
        if prediccion_l[0] == 'no':
            print('Se trata de un estado no arrestado')
            pred_vis_lasso = ridgeWCA.predict(df_S)
            #pred_vis_ridge = ridge.predict(df_S)
            #print()
            #print('El valor de la viscosidad es: ')
            pred_vis_lasso_l = pred_vis_lasso.tolist()
            #pred_vis_ridge_l = pred_vis_ridge.tolist()
            print()
            print(f'El valor de viscosidad calculado con lasso es: \n{10**(pred_vis_lasso_l[0])}') # estos valores son log(eta)
            #print(f'En cambio calculado con ridge es: \n{10**(pred_vis_ridge_l[0])}\nListo :)')
            viscos = 10**(pred_vis_lasso_l[0])
            return viscos
        else:
            print('Se trata de un estado arrestado :0.')
            viscos = 1*10**(15)
            return viscos
    else:
        prediccion = clf_model.predict(df_S)
        prediccion_l = prediccion.tolist()
        if prediccion_l[0] == 'no':
            print('Se trata de un estado no arrestado')
            pred_vis_lasso = lasso.predict(df_S)
            pred_vis_ridge = ridge.predict(df_S)
            print()
            print('El valor de la viscosidad es: ')
            pred_vis_lasso_l = pred_vis_lasso.tolist()
            pred_vis_ridge_l = pred_vis_ridge.tolist()
            print()
            print(f'El valor de viscosidad calculado con lasso es: \n{10**(pred_vis_lasso_l[0])}') # estos valores son log(eta)
            print(f'En cambio calculado con ridge es: \n{10**(pred_vis_ridge_l[0])}\nListo :)')
        else:
            print('Se trata de un estado arrestado :0.')

            # vamos a importar modelos ya entrenados. En el caso particular y debido a que para calcular viscosidades, los modelos solo funcionan
# con aquellos datos del sistema fisico que se uso para el entrenamiento, en este caso usaremos el modelo con el sistema de HS, despues 
# se puede hacer otros modelos para otros sistemas como WCA para obtener sus viscosidades.

lasso = joblib.load('log_viscosidades_senza_noscaler.joblib')
ridge = joblib.load('log_rdg_viscosidades_senza_noscaler.joblib')
clf_model = joblib.load('classification_svm_r.joblib')
lassoWCA = joblib.load('log_viscosidadesWCA_imagene.joblib')
#lassoWCA_senza = joblib.load('log_viscosidadesWCA_noscaler_senza.joblib')
ridgeWCA = joblib.load('log_viscosidadesWCA_imagene_r.joblib')

# la lectura de datos o de los datos que se quiere preguntar sobre ellos

# considero que se debe tomar en cuenta el sistema, por ahora phi no es necesario, solo el factor de estructura
# si se tiene archivos de datos con S(k):

##### carpeta con los archivos de la imagen e
directorio = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBases_WCA_visc\\DataBase_WCA_vis_e')
###### 

archivo = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBase_HS_vs_v4\\phi_0.47328_vis_11.10722953011908_.dat')
#archivo = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DateBase\\DataBases_WCA_visc\\DataBase_WCA_vis_T1\\phi_0.64_T_1.0_vs_10.851307607066715_.dat')
if "T" in archivo.name:
    model = 'WCA'
else:
    model = "HS"

    files = directorio.iterdir()
phis = []
temps = []
viss = []

for file in files:
    
    if 'T' in file.name:
        model = 'WCA'
    else:
        model = 'HS'

    df_obj = s_de_k(file, model)

    vis = flux_development(df_obj, model)
    viss.append(vis)

    ph = phi_value(file.name)
    phi = float(ph)
    phis.append(phi)

    y = tem_value(file.name)
    tem = float(y)
    temps.append(tem)

    print(f'El valor de phi: {phi}')
    print(f'El valor de T: {tem}')
    print(f'Los valores teoricos son: {file.name}')

df_aux = pd.DataFrame({
    'phis':phis,
    'temperaturas':temps,
    'vis':viss
})

df_aux.to_csv('imagen_e.csv', sep='\t', index=False)