import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
    

def calculo_de_tan(omega, gpr, gbpr):
    list_cocientes = []
    for i in range(len(omega)):
        cociente = gpr[i] / gbpr[i]
        list_cocientes.append(cociente)

    list_cocientes = np.array(list_cocientes)

    return list_cocientes

def derivada_log(x, y): # x y y son arreglos numpy
    #log_x = np.log10(x)
    #log_y = np.log10(y)

    #deri_logx_y = np.gradient(log_y, log_x)

    #dy_dx = (y / x) * deri_logx_y
    dy_dx = np.gradient(y, x)
    return dy_dx

plt.rc('text', usetex = True)

main_dir = Path('C:\\Users\\josea\\MEGA\\NESCGLE_scripts\\DataBase\\WCA')

clasificacion = []
phi = []
T_f = []
cruces = []

for ele in main_dir.iterdir():
    if ele.is_dir():
        for elem in ele.iterdir():
            dir_archivo = elem.joinpath('u_999_G_WCA.csv')
            datas = pd.read_csv(dir_archivo, delimiter='\t', names=['omega', 'Gprima', 'Gbiprima'])

            omega = datas['omega'].to_numpy()
            gpr = datas['Gprima'].to_numpy()
            gbpr = datas['Gbiprima'].to_numpy()

            tand = calculo_de_tan(omega, gpr, gbpr)

            dtand_dom = derivada_log(omega, tand)

            phi_local = str(dir_archivo.parents[0].name)
            T_f_local = str(dir_archivo.parents[1].name)
            phi_aux = float(phi_local[4:].replace('p', '.'))
            T_f_aux = float(T_f_local[3:6].replace('p', '.'))

            phi.append(phi_aux)
            T_f.append(T_f_aux)

            lista_cruces = cruces_de_uno(tand)
            cruces_g_gg_num = cruces_g_gg(gpr, gbpr)

            cruces.append(cruces_g_gg_num)

            if (len(lista_cruces) == 0) and (cruces_g_gg_num == 0):
                clasificacion.append('I')
            elif len(lista_cruces) == 1 and (cruces_g_gg_num == 1):
                clasificacion.append('III')
            elif len(lista_cruces) == 2 and (cruces_g_gg_num == 2):
                clasificacion.append('II')
            else:
                clasificacion.append('IDK')

print(clasificacion)
print(cruces)

for i in range(len(clasificacion)):
    if clasificacion[i] == 'I':
        plt.scatter(phi[i], T_f[i], marker='*', c='blue', linewidths= 0.5)
    elif clasificacion[i] == 'II':
        plt.scatter(phi[i], T_f[i], marker='o', c='red', linewidths = 0.5)
    elif clasificacion[i] == 'III':
        plt.scatter(phi[i], T_f[i], marker='o', c='green', linewidths = 0.5)
    else:
        plt.scatter(phi[i], T_f[i], marker='*', c='black', linewidths=0.5)

plt.xlabel(r'$\phi$')
plt.ylabel(r'$T$')
#plt.xlim(0, 0.45)
#plt.ylim(0, 1.8)
plt.show()
