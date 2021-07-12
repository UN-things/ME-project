import numpy as np
import matplotlib.pyplot as plt
import cv2

import my_menu as mm
from k_means import KMeans

def print_menu_1():
    menu = mm.OptionMenu("Seleccione el método de inicialización de centroides.",
                         title_color="magenta")
    menu.add_option("Forgy", color="cyan")
    menu.add_option("Macqueen", color="cyan")
    menu.add_option("Min Max", color="cyan")
    menu.add_option("Var Part", color="cyan")
    menu.add_option("Ejecutar todos", color="cyan")
    menu.add_option("Salir", color="cyan")

    return menu.run()

def print_menu_2():
    menu = mm.InputMenu("Ingrese la cantidad de grupos que desea formar.",
                        title_color="magenta")
    menu.add_input("k", color="cyan")
    return menu.run()

def process_image(route_image):
    img_orig = cv2.imread(route_image)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = img_orig / 255
    img = img_orig.reshape(-1, 3)
    return img_orig, img

def print_img(img_orig, img):
    plt.close()
    plt.axis('off')
    plt.imshow(img_orig)
    plt.show()

    plt.close()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def run_k_means(k, init_method):
    img_orig, img = process_image('data/bone_scan.jpg')
    model = KMeans(k=k, init_method=init_method)
    cluster_means, img_with_clusters = model.fit(img)
    compressed_img = np.zeros(img.shape)

    ## Assigning each pixel color to its corresponding cluster centroid
    for i, cluster in enumerate(img_with_clusters[:, -1]):
        compressed_img[i, :] = cluster_means[ int(cluster) ]

    compressed_img_reshaped = compressed_img.reshape(img_orig.shape)
    print_img(img_orig, compressed_img_reshaped)

def main():
    result = print_menu_1()
    if(result == "Forgy"):
        init_method = "forgy"
    elif(result == "Macqueen"):
        init_method = "macqueen"
    elif(result == "Min Max"):
        init_method = "min_max"
    elif(result == "Var Part"):
        init_method = "var_part"
    elif(result == "Ejecutar todos"):
        pass
    else:
        return 0
    temp = print_menu_2()
    k = int(temp["k"])
    run_k_means(k, init_method)

if __name__ == "__main__":
    main()
