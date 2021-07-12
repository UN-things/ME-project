import my_menu as mm
from k_means import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import io
import time

wolf_img_orig = cv2.imread('data/wolf.jpg')

wolf_img_orig = cv2.cvtColor(wolf_img_orig, cv2.COLOR_BGR2RGB)
wolf_img_shape = wolf_img_orig.shape
# Convert 3d image array to 2d array
wolf_img_orig = wolf_img_orig / 255
wolf_img = wolf_img_orig.reshape(-1, 3)

model = KMeans(k=12, init_method = 'var_part') # When var_part is used, the algorithm is a lot faster.
cluster_means, image_data_with_clusters = model.fit(wolf_img)

compressed_image = np.zeros(wolf_img.shape)

## Assigning each pixel color to its corresponding cluster centroid
for i, cluster in enumerate(image_data_with_clusters[:, -1]):
    compressed_image[i, :] = cluster_means[ int(cluster) ]

compressed_image_reshaped = compressed_image.reshape(wolf_img_shape)

plt.close()
plt.axis('off')
plt.imshow(wolf_img_orig)

plt.close()
plt.axis('off')
plt.imshow(compressed_image_reshaped)

plt.show()