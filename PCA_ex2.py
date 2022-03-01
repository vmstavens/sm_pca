import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats

import PCA_utils

"""
# Exercise 2.2: Normalization

1.  Perform one of the two normalizations suggested in the lecture (min-max 
    normalization and z-standardization) for the best parameter setting found
    under 2.1.3 and apply kNN with 10 fold cross-validation 
    (10 runs, 90% training and 10% test set).
2.  Apply the normalization before and after PCA independently and compare the 
    results. Analyze the results.
"""

data_csv = pd.read_csv("data_proc.csv", header=None)
data = pd.DataFrame.to_numpy(data_csv)
data = np.delete(data, 0, 1)
np.random.shuffle(data)

# Best performance was found using
# K = 1
# PCA variance coverage = 80 %
# Using mixed data set

data_nB = data  #Data with student id, label, img

data_nB[:,2:] = stats.zscore(data[:,2:], axis=0)


# nA = normalize After pca
# nB = normalize Before pca
pca_nA = PCA()
pca_nB = PCA()

# Fit the PCAs using the entire data set
pca_nA.fit(data[:,2:])
pca_nB.fit(data_nB[:,2:])


## Calculate cummulative variance ratios
pca_nA_varRatio = pca_nA.explained_variance_ / np.sum(pca_nA.explained_variance_)
pca_nB_varRatio = pca_nB.explained_variance_ / np.sum(pca_nB.explained_variance_)

pca_nA_varRatioCum = np.ndarray(pca_nA_varRatio.shape)
for i in range(len(pca_nA_varRatio)):
    if i == 0:
        pca_nA_varRatioCum[i] = pca_nA_varRatio[i]
    else:
        pca_nA_varRatioCum[i] = pca_nA_varRatioCum[i-1] + pca_nA_varRatio[i]

pca_nB_varRatioCum = np.ndarray(pca_nB_varRatio.shape)
for i in range(len(pca_nB_varRatio)):
    if i == 0:
        pca_nB_varRatioCum[i] = pca_nB_varRatio[i]
    else:
        pca_nB_varRatioCum[i] = pca_nB_varRatioCum[i-1] + pca_nB_varRatio[i]


# Perform KNN with cross validation
K = [1]
pct = 0.8

acc_nA = []
acc_nB = []
cT_nA = []
cT_nB = []

for i in range(10):
    print('i =', i+1, '/ 10')
    # Split and tranform pre-normalized data
    d_train_nB, l_train_nB, d_test_nB, l_test_nB = PCA_utils.split_data_crossValidation(data_nB, i)
    d_train_nB = pca_nB.transform(d_train_nB)
    d_test_nB = pca_nB.transform(d_test_nB)

    # Split, transform and normalize original data
    d_train_nA, l_train_nA, d_test_nA, l_test_nA = PCA_utils.split_data_crossValidation(data, i)
    d_train_nA = pca_nA.transform(d_train_nA)
    d_test_nA = pca_nA.transform(d_test_nA)
    d_train_nA = stats.zscore(d_train_nA, axis=0)
    d_test_nA = stats.zscore(d_test_nA, axis=0)

    # Perform KNN for test slice i
    pct_idx = np.min(np.argwhere(pca_nB_varRatioCum > pct))
    a,_,_,_,t = PCA_utils.knnParamSearch(d_train_nB[:,:pct_idx], l_train_nB, d_test_nB[:,:pct_idx], l_test_nB, K, metrics=['euclidean'])
    acc_nB.append(a)
    cT_nB.append(t)

    pct_idx = np.min(np.argwhere(pca_nA_varRatioCum > pct))
    a,_,_,_,t = PCA_utils.knnParamSearch(d_train_nA[:,:pct_idx], l_train_nA, d_test_nA[:,:pct_idx], l_test_nA, K, metrics=['euclidean'])
    acc_nA.append(a)
    cT_nA.append(t)

print("Pre-normalized data:\nacc: ", acc_nB, "\ncT: ", cT_nB)
print("Post-normalized data:\nacc: ", acc_nA, "\ncT: ", cT_nA)

# Plotting accuracy and computation time for both runs
# acc_nB = [[0.7274242424242424], [0.7328787878787879], [0.7406060606060606], [0.7334848484848485], [0.7218181818181818], [0.7296969696969697], [0.7436363636363637], [0.7343939393939394], [0.7396969696969697], [0.7293939393939394]] 
# cT_nB = [[8.738563299179077], [7.949604749679565], [7.541474342346191], [7.5706024169921875], [7.589169263839722], [7.458940744400024], [7.633926153182983], [7.774741172790527], [7.677222013473511], [7.411119699478149]]
# acc_nA = [[0.675], [0.6845454545454546], [0.6840909090909091], [0.6657575757575758], [0.676969696969697], [0.6693939393939394], [0.6763636363636364], [0.6783333333333333], [0.6798484848484848], [0.6668181818181819]] 
# cT_nA =  [[7.945237636566162], [7.59130072593689], [7.426719903945923], [7.615141153335571], [7.46854043006897], [7.6106202602386475], [7.648665428161621], [7.764485836029053], [7.922472715377808], [7.417284727096558]]

acc_nB_mean = [np.mean(acc_nB)]*10
acc_nA_mean = [np.mean(acc_nA)]*10
cT_nB_mean = [np.mean(cT_nB)]*10
cT_nA_mean = [np.mean(cT_nA)]*10


fig, axs = plt.subplots(1,2)

axs[0].plot(range(1,11), acc_nB, linewidth=2.0, color='blue', marker='.', markersize=12, label="Pre-normalized")
axs[0].plot(range(1,11), acc_nA, linewidth=2.0, color='orange', marker='.', markersize=12, label="Post-normalized")
axs[0].plot(range(1,11), acc_nB_mean, linewidth=1.5, color='blue', marker = ' ')
axs[0].plot(range(1,11), acc_nA_mean, linewidth=1.5, color='orange', marker = ' ')
axs[0].set(title="Accuracy of crossvalidation", xlabel="test section", ylabel="Accuracy")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(1,11), cT_nB, linewidth=2.0, color='blue', marker='.', markersize=12, label="Pre-normalized")
axs[1].plot(range(1,11), cT_nA, linewidth=2.0, color='orange', marker='.', markersize=12, label="Post-normalized")
axs[1].plot(range(1,11), cT_nB_mean, linewidth=1.5, color='blue', marker = ' ')
axs[1].plot(range(1,11), cT_nA_mean, linewidth=1.5, color='orange', marker = ' ')
axs[1].set(title="Computation time of crossvalidation", xlabel="test section", ylabel="Time[s]")
axs[1].legend()
axs[1].grid(True)

plt.show()

