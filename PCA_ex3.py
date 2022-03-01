import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import PCA_utils
from scipy import misc


"""
Exercise 2.3: Gaussian smoothing

Apply Gaussian smoothing function with different sigmas to the images. 
Perform again the steps in task 2.2 (Cross validation). 
Describe and analyze the results for one of the smoothing methods depending on the amount of smoothing.
"""


data_csv = pd.read_csv("data_proc.csv", header=None)
data = pd.DataFrame.to_numpy(data_csv)
data = np.delete(data, 0, 1)
np.random.shuffle(data)

