from cv2 import blur
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import PCA_utils
from scipy.ndimage import gaussian_filter


"""
Exercise 2.3: Gaussian smoothing

Apply Gaussian smoothing function with different sigmas to the images. 
Perform again the steps in task 2.2 (Cross validation). 
Describe and analyze the results for one of the smoothing methods depending on the amount of smoothing.
"""

#Loading data and removing first column due to index in csv file
data_csv = pd.read_csv("data_proc.csv", header=None)
data = pd.DataFrame.to_numpy(data_csv)
data = np.delete(data, 0, 1)
np.random.shuffle(data)

#Extracting random images from the data and visualizing thease
random_indices = np.random.choice(data.shape[0], size=3, replace=False)
images = data[random_indices, :]
#PCA_utils.visualize_images(images, 1, 3, "Original data")


test = images.reshape(3,23,23)
print(test.shape)


#Creating gaussian filter, smoothing images and visualizing images
# blurred_images = np.array([[]])
# for i in range(images.shape[0]):
#     np.append(blurred_images, gaussian_filter(images[i], sigma=1), axis=0)

# print(blurred_images)
# print(blurred_images.shape)

#PCA_utils.visualize_images(blurred_images, 1, 3, "Blurred images with sigma=1")

