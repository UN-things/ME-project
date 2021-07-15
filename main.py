import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cv2
from consolemenu import *
from consolemenu.format import *
from consolemenu.items import *
from k_means import KMeans
from sklearn.preprocessing import StandardScaler

# Reduce la dimensión de la imagen de 3 dimensiones a 2
def process_image(route_image):
    img_orig = cv2.imread(route_image)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = img_orig / 255
    img = img_orig.reshape(-1, 3)
    return img_orig, img


def print_img(img_orig, img):
    figure, axis = plt.subplots(2, figsize=(10,10))
    plt.axis('off')
    axis[0].imshow(img_orig)
    axis[0].set_title("Imagen Original")
    axis[1].imshow(img)
    axis[1].set_title("Imagen clusterizada")
    for a in axis:
        a.axis('off')
    plt.show()
    plt.close()


def run_k_means(k, init_method):
    img_orig, img = process_image('data/bone_scan.jpg')
    model = KMeans(k=k, init_method=init_method)
    # cluster_means => k x 3(RGB)
    # img_with_clusters => 1000 filas-pixeles X 3 (RGB) + 1 (clusters (0-k-1))
    cluster_means, img_with_clusters = model.fit(img)
    # Matriz de ceros de N_pixeles x 3(RGB)
    compressed_img = np.zeros(img.shape)

    # Cada pixel pertenece a un cluster, entonces solamente hay k colores
    # lo que hace el for es cambiar el RGB a cada pixel segun el grupo al que
    # pertence
    for i, cluster in enumerate(img_with_clusters[:, -1]):
        compressed_img[i, :] = cluster_means[int(cluster)]

    # Rearma la imagen
    compressed_img_reshaped = compressed_img.reshape(img_orig.shape)
    print_img(img_orig, compressed_img_reshaped)


def get_menu_format():
    menu_format = MenuFormatBuilder().set_border_style_type(MenuBorderStyleType
                                                            .HEAVY_BORDER) \
        .set_prompt("SELECT>") \
        .set_title_align('center') \
        .set_subtitle_align('center') \
        .set_left_margin(4) \
        .set_right_margin(2)
    return menu_format


def get_method_info(method):
    methods = {
        "Forgy": " Forgy Description",
        "Macqueen": "Macqueen Description",
        "Min Max": "Min max Description",
        "Var Part": "Var Part Description"
    }
    return methods[method]


def image_submenu(method):
    mn = {
        "title": "Selección de k",
        "item_1": "2",
        "item_2": "3",
        "item_3": "4"
    }
    menu = ConsoleMenu(title=mn["title"], prologue_text=get_method_info(method),
                       formatter=get_menu_format())

    k_2_item = FunctionItem(mn["item_1"], run_k_means, [
                            int(mn["item_1"]), method])
    k_3_item = FunctionItem(mn["item_2"], run_k_means, [
                            int(mn["item_2"]), method])
    k_4_item = FunctionItem(mn["item_3"], run_k_means, [
                            int(mn["item_3"]), method])
    # Seleccionar la cantidad e clusters = k
    menu.append_item(k_2_item)
    menu.append_item(k_3_item)
    menu.append_item(k_4_item)
    return menu


def image_menu():
    mn = {
        "title": "Clusterización de una imagen",
        "prologue": "A continuación se realizará la clusterización de una radiografía, esto dividirá los pixeles de la imagen en k grupos, lo que nos permitiría detectar formaciones anormales en esta. Por favor seleccione uno de los 4 métodos de inicialización de centroides.",
        "item_1": "Forgy",
        "item_2": "Macqueen",
        "item_3": "Min Max",
        "item_4": "Var Part"
    }

    menu = ConsoleMenu(title=mn["title"], prologue_text=mn["prologue"],
                       formatter=get_menu_format())
    forgy_item = SubmenuItem(mn["item_1"],
                             image_submenu(mn["item_1"]), menu)
    macqueen_item = SubmenuItem(mn["item_2"],
                                image_submenu(mn["item_2"]), menu)
    min_max_item = SubmenuItem(mn["item_3"],
                               image_submenu(mn["item_3"]), menu)
    var_part_item = SubmenuItem(
        mn["item_4"], image_submenu(mn["item_4"]), menu)
    # seleccionar el método de inicialización
    menu.append_item(forgy_item)
    menu.append_item(macqueen_item)
    menu.append_item(min_max_item)
    menu.append_item(var_part_item)
    return menu

def k_means_dataset(df, k):
    model = KMeans(max_iter = 500, tolerance = 0.001, k = k, init_method = "Min Max")
    (clusters, data) = model.fit(df)

    plt.close()
    plt.figure(figsize=(12,12))
    plt.subplot(321)
    for i, cluster_mean in enumerate(clusters):
        data_cluster_i = data[ data[:, -1] == i ]
        plt.scatter(data_cluster_i[:, 2], data_cluster_i[:, 0], label = 'Cluster ' + str(i))
        plt.plot(cluster_mean[0], cluster_mean[1], label = 'Centroid ' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1)
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=16)
    plt.title('Partidos ganados', fontsize=16)

    plt.subplot(322)
    for i, cluster_mean in enumerate(clusters):
        data_cluster_i = data[ data[:, -1] == i ]
        plt.scatter(data_cluster_i[:, 2], data_cluster_i[:, 1], label = 'Cluster ' + str(i))
        plt.plot(cluster_mean[0], cluster_mean[1], label = 'Centroid ' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1)
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=16)
    plt.title('Eficiencia defensiva', fontsize=16)

    plt.subplot(323)
    for i, cluster_mean in enumerate(clusters):
        data_cluster_i = data[ data[:, -1] == i ]
        plt.scatter(data_cluster_i[:, 2], data_cluster_i[:, 3], label = 'Cluster ' + str(i))
        plt.plot(cluster_mean[0], cluster_mean[1], label = 'Centroid ' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1)
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=16)
    plt.title('Lanzamientos efectivos', fontsize=16)

    plt.subplot(324)
    for i, cluster_mean in enumerate(clusters):
        data_cluster_i = data[ data[:, -1] == i ]
        plt.scatter(data_cluster_i[:, 2], data_cluster_i[:, 4], label = 'Cluster ' + str(i))
        plt.plot(cluster_mean[0], cluster_mean[1], label = 'Centroid ' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1)
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=16)
    plt.title('Porcentaje de tiros de 2 puntos hechos', fontsize=16)


    plt.subplot(325)
    for i, cluster_mean in enumerate(clusters):
        data_cluster_i = data[ data[:, -1] == i ]
        plt.scatter(data_cluster_i[:, 2], data_cluster_i[:, 5], label = 'Cluster ' + str(i))
        plt.plot(cluster_mean[0], cluster_mean[1], label = 'Centroid ' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1)
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=16)
    plt.title('Triunfos por encima de la burbuja', fontsize=16)

    plt.show()

def process_data(df):
    temp = df[['W','ADJOE','BARTHAG','EFG_O','2P_O','WAB']]
    scaler = StandardScaler()
    df_scale = scaler.fit_transform(temp)
    return df_scale


def print_selected_variables(datos):
    plt.figure(figsize=(12,10))

    plt.subplot(321)
    plt.scatter(y=datos['BARTHAG'], x=datos['W'],alpha=0.5, edgecolor='k')
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=12)
    plt.title('Partidos Ganados', fontsize=12)

    plt.subplot(322)
    plt.scatter(y=datos['BARTHAG'], x=datos['ADJOE'],alpha=0.5, edgecolor='k')
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=12)
    plt.title('Eficiencia Defensiva', fontsize=12)


    plt.subplot(323)
    plt.scatter(y=datos['BARTHAG'], x=datos['EFG_O'],alpha=0.5, edgecolor='k')
    plt.yticks(fontsize=12)
    plt.ylabel('Prob Ganar', fontsize=12)
    plt.title('Lanzamientos efectivos', fontsize=12)

    plt.subplot(324)
    plt.scatter(y=datos['BARTHAG'], x=datos['2P_O'],alpha=0.5, edgecolor='k')
    plt.ylabel('Prob Ganar', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('% rotaciónPorcentaje de tiros de 2 puntos hechos', fontsize=12)


    plt.subplot(325)
    plt.scatter(y=datos['BARTHAG'], x=datos['WAB'],alpha=0.5, edgecolor='k')
    plt.ylabel('Prob ganar', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Triunfos por encima de la burbuja', fontsize=12)

    plt.show()


def print_cmap(df):
    alpha = ['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB']
    corr = df.corr()
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticklabels(['']+alpha)
    ax.set_yticklabels(['']+alpha)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.show()

def get_dataframe():
    df = pd.read_csv('data/basketball_19.csv')
    df = df.drop(['TEAM','CONF','POSTSEASON','SEED'],axis=1)
    return df

def dataset_submenu_3(df):
    mn = {
        "title": "Clusterización del dataset",
        "prologue": "Para realizar la clusterización de los datos se eliminaron las columnas que no fueron seleccionadas en el paso anterior y se estandarizaron las variables. Seleccione la cantidad de grupos que desea formar.",
        "item_1": "2",
        "item_2": "3",
        "item_3": "4"
    }
    new_df = process_data(df)

    menu = ConsoleMenu(title=mn["title"], prologue_text=mn["prologue"],
                       formatter=get_menu_format())

    k_2_item = FunctionItem(mn["item_1"], k_means_dataset, [new_df, int(mn["item_1"])])
    k_3_item = FunctionItem(mn["item_2"], k_means_dataset, [new_df, int(mn["item_2"])])
    k_4_item = FunctionItem(mn["item_3"], k_means_dataset, [new_df, int(mn["item_3"])])
    # Seleccionar la cantidad e clusters = k
    menu.append_item(k_2_item)
    menu.append_item(k_3_item)
    menu.append_item(k_4_item)
    return menu

def dataset_submenu_2(df):
    mn = {
        "title": "Variables seleccionadas",
        "prologue": "Se escogieron 5 variables que están fuertemente relacionadas con el BARTHAG (probabilidad de vencer a un equipo), y son W (Partidos ganados), ADJOE (Eficiencia Defensiva), EFG_O (Lanzamientos efectivos), 2P_O (Porcentaje de tiros de 2 puntos hechos) y WAB (Triunfos por encima de la burbuja).",
        "item_1": "Ver graficas de las variables selecionadas",
        "item_2": "Continuar"
    }

    menu = ConsoleMenu(title=mn["title"], prologue_text=mn["prologue"],
                       formatter=get_menu_format())
    item_1 = FunctionItem(mn["item_1"], print_selected_variables, [df])
    item_2 = SubmenuItem(mn["item_2"], dataset_submenu_3(df), menu)

    menu.append_item(item_1)
    menu.append_item(item_2)
    return menu

def dataset_submenu_1(df):
    mn = {
        "title": "Exploración de datos",
        "prologue": "Para empezar el análisis hay que hacer una exploración inicial de los datos. Nuestra variable objetivo es el BARTHAG (probabilidad de vencer a un equipo), por lo que elegiremos variables que estén fuertemente relacionadas con este, para ello utilizaremos un mapa de calor que podrá ver seleccionando la primera opción del menú.",
        "item_1": "Ver mapa de calor",
        "item_2": "Continuar"
    }
    menu = ConsoleMenu(title=mn["title"], prologue_text=mn["prologue"],
                       formatter=get_menu_format())
    item_1 = FunctionItem(mn["item_1"], print_cmap, [df])
    item_2 = SubmenuItem(mn["item_2"], dataset_submenu_2(df), menu)

    menu.append_item(item_1)
    menu.append_item(item_2)
    return menu

def dataset_menu():
    mn = {
        "title": "Clusterización sobre un dataset de baloncesto",
        "prologue": " Se tienen los datos de desempeño de los equipos de baloncesto del torneo NCAA March Madness que contiene las estadísticas de juego de 353 equipos de la liga. El objetivo es encontrar patrones en el desempeño de los equipos y generar recomendaciones de umbrales en las estadísticas para que un equipo esté en el grupo de desempeño superior.",
        "item_1": "Exploración de datos"
    }

    df = get_dataframe()

    menu = ConsoleMenu(title=mn["title"], prologue_text=mn["prologue"],
                       formatter=get_menu_format())
    item = SubmenuItem(mn["item_1"], dataset_submenu_1(df), menu)

    menu.append_item(item)
    return menu

def principal_menu():
    mn = {
        "title": "Clusterización usando K-Means",
        "prologue": "A continuación podrá seleccionar 1 de 3 escenarios en los que se utiliza el algoritmo de K-Means para dividir un conjunto de datos     en grupos con características similares.",
        "item_1": "Clusterización de una imágen.",
        "item_2": "Clusterización de un conjutno de datos.",
        "item_3": "Comparación de los 4 tipos de inicialización de centroides."
    }

    menu = ConsoleMenu(title=mn["title"], prologue_text=mn["prologue"],
                       formatter=get_menu_format())
    img_item = SubmenuItem(mn["item_1"], image_menu(), menu)
    df_item = SubmenuItem(mn["item_2"], dataset_menu(), menu)

    menu.append_item(img_item)
    menu.append_item(df_item)
    menu.show()


def main():
    principal_menu()


if __name__ == '__main__':
    main()
