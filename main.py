import matplotlib.pyplot as plt
import numpy as np

import cv2
from consolemenu import *
from consolemenu.format import *
from consolemenu.items import *
from k_means import KMeans


def process_image(route_image):
    img_orig = cv2.imread(route_image)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = img_orig / 255
    img = img_orig.reshape(-1, 3)
    return img_orig, img


def print_img(img_orig, img):
    figure, axis = plt.subplots(2)
    axis[0].imshow(img_orig)
    axis[1].imshow(img)
    # plt.axis('off')
    plt.show()


def run_k_means(k, init_method):
    img_orig, img = process_image('data/bone_scan.jpg')
    model = KMeans(k=k, init_method=init_method)
    cluster_means, img_with_clusters = model.fit(img)
    compressed_img = np.zeros(img.shape)

    # Assigning each pixel color to its corresponding cluster centroid
    for i, cluster in enumerate(img_with_clusters[:, -1]):
        compressed_img[i, :] = cluster_means[int(cluster)]

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

    k_2_item = FunctionItem(mn["item_1"], run_k_means, [int(mn["item_1"]), method])
    k_3_item = FunctionItem(mn["item_2"], run_k_means, [int(mn["item_2"]), method])
    k_4_item = FunctionItem(mn["item_3"], run_k_means, [int(mn["item_3"]), method])

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
    menu.append_item(forgy_item)
    menu.append_item(macqueen_item)
    menu.append_item(min_max_item)
    menu.append_item(var_part_item)
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
    menu.append_item(img_item)
    menu.show()


def main():
    principal_menu()


if __name__ == '__main__':
    main()
