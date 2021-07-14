from consolemenu import *
from consolemenu.format import *
from consolemenu.items import *


def hello():
    print("Hello World")


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
        "title": "Selección de k"
        "item_1": "2",
        "item_2": "3",
        "item_3": "4"
    }
    menu = ConsoleMenu(title=mn["title"], prologue_text=get_method_info(method),
                       formatter=get_menu_format())

    k_2_item = FunctionItem(mn["item_1"], hello())
    k_3_item = FunctionItem(mn["item_2"], hello())
    k_4_item = FunctionItem(mn["item_3"], hello())

    menu.append_item(k_2_item)
    menu.append_item(k_3_item)
    menu.append_item(k_4_item)


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
    forgy_item = SubmenuItem(mn["item_1"], image_submenu(mn["item_1"]), menu)
    macqueen_item = SubmenuItem(
        mn["item_2"], image_submenu(mn["item_2"]), menu)
    min_max_item = SubmenuItem(mn["item_3"], image_submenu(mn["item_3"]), menu)
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
