#!/usr/local pyhton3

from operator import sub
import pandas as pd
import numpy as np
import PCA_utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from math import sqrt

## Exercise 2.4: Reconstruction using PCA
# 1. This task is about reconstructing data using PCA. First select some images from the dataset and plot them.
# 2. Plot the first 10 eigenvectors/loadingvectors as images. Can you describe what you see?
# 3. Plot a reconstruction of the images you displayed in 2.4.1 using all PC’s. This can be done by multiplying the loadings with the scores and adding the removed centering.
# 4. Now re-recreate using 80 % of variance, 90 % and 95 % . Can you describe what you see? How much have you reduced the data size?
# 5. The last exercise is to compare the outcomes between two different ciphers. For instance, two different ciphers, (e.g. datapoints that represent a ’0’ and a ’1’), compare the 10 first scores and see if you can spot a difference. Try also to calculate the mean for all instances of these ciphers and compare the first 10 scores. Can you spot a pattern when comparing with the loadings.

NUM_OF_DP_2_VIS = 9

data_csv = pd.read_csv("data_proc.csv", header=None)
data = pd.DataFrame.to_numpy(data_csv)
data = np.delete(data, 0, 1)

# 1. This task is about reconstructing data using PCA. First select some images from the dataset and plot them.
# random_indices = np.random.choice(data.shape[0], size=NUM_OF_DP_2_VIS, replace=False)
# random_images = data[random_indices, :]
# PCA_utils.visualize_images(random_images, sqrt(NUM_OF_DP_2_VIS), sqrt(NUM_OF_DP_2_VIS), "Original data")

# 2. Plot the first 10 eigenvectors/loadingvectors as images. Can you describe what you see?
pca_decomp = PCA()
pca_decomp.fit(data)

pcas = pca_decomp.components_[:NUM_OF_DP_2_VIS, :]
sub_titles = []
for i in range(1, NUM_OF_DP_2_VIS + 1):
    sub_titles.append("pca " + str(i))


# print 9 pcas
# PCA_utils.visualize_generic(pcas, sqrt(NUM_OF_DP_2_VIS), sqrt(NUM_OF_DP_2_VIS), "PCAs", sub_titles)
# print the 10'th pca
# PCA_utils.visualize_generic(pca_decomp.components_[9], 1,1, "PCAs", "pca 10")

# description: shits cool yo

# 3. Plot a reconstruction of the images you displayed in 2.4.1 using all 
# PC’s. This can be done by multiplying the loadings with the scores and 
# adding the removed centering.

# loadings = eigenvectors = components
# scores = 
Z = pca_decomp.score_samples(data)
