from email import header
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import PCA_utils

"""
# Exercise 2.1: Principal Component Analysis (PCA)

Perform a PCA on the data for the "all persons in" and "disjunct" data set.

1.  Show the standard deviation ( From prcomp Eigenvalues ), the proportion of
    variance and the cumulative sum of variance of the principal components. 
    (In the report the first 10-20 principal components, should be sufficient 
    to illustrate the tendencies.)
2.  Show the performance of selecting enough principal components to represent 
    80%, 90%, 95%, 99% of the accumulated variance. For each test vary "k" in 
    kNN, try 3 reasonable values.
3.  Measure run times for the prediction step of the kNN-classifier with PCA 
    based dimensionality reduction. How does the feature vector dimensionality 
    effect performance?
4.  Interpret the results.
"""

PERFORM_KNN = False

data_csv = pd.read_csv("data_proc.csv", header=None)
data = pd.DataFrame.to_numpy(data_csv)
data = np.delete(data, 0, 1)
np.random.shuffle(data)

#####################################
# 2.1.1
#####################################
# Fitting the PCA
d_train_all, l_train_all, d_test_all, l_test_all = PCA_utils.split_data_all(data, 0.8)
d_train_dis, l_train_dis, d_test_dis, l_test_dis = PCA_utils.split_data_disjunct(data, 0.8)

pca_all = PCA()
pca_all.fit(d_train_all)

pca_dis = PCA()
pca_dis.fit(d_train_dis)

# Preparing data
pca_all_varRatio = pca_all.explained_variance_ / np.sum(pca_all.explained_variance_)
pca_dis_varRatio = pca_dis.explained_variance_ / np.sum(pca_dis.explained_variance_)

pca_all_varRatioCum = np.ndarray(pca_all_varRatio.shape)
for i in range(len(pca_all_varRatio)):
    if i == 0:
        pca_all_varRatioCum[i] = pca_all_varRatio[i]
    else:
        pca_all_varRatioCum[i] = pca_all_varRatioCum[i-1] + pca_all_varRatio[i]

pca_dis_varRatioCum = np.ndarray(pca_dis_varRatio.shape)
for i in range(len(pca_dis_varRatio)):
    if i == 0:
        pca_dis_varRatioCum[i] = pca_dis_varRatio[i]
    else:
        pca_dis_varRatioCum[i] = pca_dis_varRatioCum[i-1] + pca_dis_varRatio[i]

# Printing variance
sum = 0
print('PCA#\t Var\t\t Pct\t\t CumVar\t\t CumPct')
for i in range(200):
    sum += pca_all.explained_variance_[i]
    print(i, '\t', "{:.2f}".format(pca_all.explained_variance_[i]), '\t', "{:.5f}".format(pca_all_varRatio[i]), '\t', "{:.2f}".format(sum), '\t', "{:.5}".format(pca_all_varRatioCum[i]))

# Plotting
plt.plot(range(len(pca_all.explained_variance_)), pca_all_varRatioCum, linewidth=1.0, marker=".", markersize=1, label="All students")
plt.plot(range(len(pca_dis.explained_variance_)), pca_dis_varRatioCum, linewidth=2.0, marker=".", markersize=1, label="Split students")

plt.legend()
plt.title("Cummulative variance ratios")
plt.ylabel("Variance %")
plt.xlabel("PCA#")
plt.grid(True)
plt.show()

#####################################
# 2.1.2-3
#####################################
# Transform data into PCA space
d_train_all_pca = pca_all.transform(d_train_all)
d_test_all_pca = pca_all.transform(d_test_all)

d_train_dis_pca = pca_dis.transform(d_train_dis)
d_test_dis_pca = pca_dis.transform(d_test_dis)

# Running KNN for different PCA# and different K
ks = [1, 3, 5]
pcts = [0.8, 0.9, 0.95, 0.99]

if PERFORM_KNN:
    accuracy_all  = np.ndarray((len(pcts), len(ks)))
    comp_time_all = np.ndarray((len(pcts), len(ks)))
    print("K =", ks)

    print("Running for mixed students, evaluating on test data")
    for i, pct in enumerate(pcts):
        # argwhere returns array of indicies where the statement is true. min returns the smallest element i.e. the threshold index
        pct_idx = np.min(np.argwhere(pca_all_varRatioCum > pct))

        print("{:.0f}".format(pct*100), "%, p =", pct_idx)
        a, _, _, _, t = PCA_utils.knnParamSearch(d_train_all_pca[:,:pct_idx], l_train_all, 
                                                d_test_all_pca[:,:pct_idx], l_test_all, 
                                                ks, metrics=['euclidean'])
        accuracy_all[i,:] = np.array(a)
        comp_time_all[i,:] = np.array(t)

    print('\n')

    accuracy_dis  = np.ndarray((len(pcts), len(ks)))
    comp_time_dis = np.ndarray((len(pcts), len(ks)))

    print("Running for split students, evaluating on test data")
    for i, pct in enumerate(pcts):
        pct_idx = np.min(np.argwhere(pca_dis_varRatioCum > pct))
        
        print("{:.0f}".format(pct*100), "%, p =", pct_idx, "...")
        a, _, _, _, t = PCA_utils.knnParamSearch(d_train_dis_pca[:,:pct_idx], l_train_dis, 
                                                d_test_dis_pca[:,:pct_idx], l_test_dis, 
                                                ks, metrics=['euclidean'])
        accuracy_dis[i,:] = np.array(a)
        comp_time_dis[i,:] = np.array(t)

    print('\n')

    accuracy_all_train  = np.ndarray((len(pcts), len(ks)))
    comp_time_all_train = np.ndarray((len(pcts), len(ks)))
    print("K =", ks)

    print("Running for mixed students, evaluating on training data")
    for i, pct in enumerate(pcts):
        # argwhere returns array of indicies where the statement is true. min returns the smallest element i.e. the threshold index
        pct_idx = np.min(np.argwhere(pca_all_varRatioCum > pct))

        print("{:.0f}".format(pct*100), "%, p =", pct_idx)
        a, _, _, _, t = PCA_utils.knnParamSearch(d_train_all_pca[:,:pct_idx], l_train_all, 
                                                d_train_all_pca[:,:pct_idx], l_train_all, 
                                                ks, metrics=['euclidean'])
        accuracy_all_train[i,:] = np.array(a)
        comp_time_all_train[i,:] = np.array(t)

    print('\n')

    accuracy_dis_train  = np.ndarray((len(pcts), len(ks)))
    comp_time_dis_train = np.ndarray((len(pcts), len(ks)))

    print("Running for split students, evaluating on training data")
    for i, pct in enumerate(pcts):
        pct_idx = np.min(np.argwhere(pca_dis_varRatioCum > pct))
        
        print("{:.0f}".format(pct*100), "%, p =", pct_idx, "...")
        a, _, _, _, t = PCA_utils.knnParamSearch(d_train_dis_pca[:,:pct_idx], l_train_dis, 
                                                d_train_dis_pca[:,:pct_idx], l_train_dis, 
                                                ks, metrics=['euclidean'])
        accuracy_dis_train[i,:] = np.array(a)
        comp_time_dis_train[i,:] = np.array(t)

    print('\n')

    np.savetxt("acc_all.csv", accuracy_all, delimiter=",")
    np.savetxt("acc_dis.csv", accuracy_dis, delimiter=",")
    np.savetxt("c_time_all.csv", comp_time_all, delimiter=",")
    np.savetxt("c_time_dis.csv", comp_time_dis, delimiter=",")
    np.savetxt("acc_all_train.csv", accuracy_all_train, delimiter=",")
    np.savetxt("c_time_all_train.csv", comp_time_all_train, delimiter=",")
    np.savetxt("acc_dis_train.csv", accuracy_dis_train, delimiter=",")
    np.savetxt("c_time_dis_train.csv", comp_time_dis_train, delimiter=",")
    
    

else:
    accuracy_all = pd.read_csv("acc_all.csv", header=None)
    accuracy_all = pd.DataFrame.to_numpy(accuracy_all)
    accuracy_dis = pd.read_csv("acc_dis.csv", header=None)
    accuracy_dis = pd.DataFrame.to_numpy(accuracy_dis)
    accuracy_all_train = pd.read_csv("acc_all_train.csv", header=None)
    accuracy_all_train = pd.DataFrame.to_numpy(accuracy_all_train)
    accuracy_dis_train = pd.read_csv("acc_dis_train.csv", header=None)
    accuracy_dis_train = pd.DataFrame.to_numpy(accuracy_dis_train)
    comp_time_all = pd.read_csv("c_time_all.csv", header=None)
    comp_time_all = pd.DataFrame.to_numpy(comp_time_all)
    comp_time_dis = pd.read_csv("c_time_dis.csv", header=None)
    comp_time_dis = pd.DataFrame.to_numpy(comp_time_dis)
    comp_time_all_train = pd.read_csv("c_time_all_train.csv", header=None)
    comp_time_all_train = pd.DataFrame.to_numpy(comp_time_all_train)
    comp_time_dis_train = pd.read_csv("c_time_dis_train.csv", header=None)
    comp_time_dis_train = pd.DataFrame.to_numpy(comp_time_dis_train)


print("rows = PCA%, cols = K")
print("accuracy_all:\n", accuracy_all)
print("accuracy_dis:\n", accuracy_dis)
print("comp_time_all:\n", comp_time_all)
print("comp_time_dis:\n", comp_time_dis)


# Plot KNN results
ac_max = np.max((np.max(accuracy_all), np.max(accuracy_dis), np.max(accuracy_all_train), np.max(accuracy_dis_train)))
ac_min = np.min((np.min(accuracy_all), np.min(accuracy_dis), np.min(accuracy_all_train), np.min(accuracy_dis_train))) 
ct_max = np.max((np.max(comp_time_all), np.max(comp_time_dis), np.max(comp_time_all_train), np.max(comp_time_dis_train))) + 1
ct_min = np.min((np.min(comp_time_all), np.min(comp_time_dis), np.min(comp_time_all_train), np.min(comp_time_dis_train))) - 1

fig, axs = plt.subplots(2, 2)

axs[0,0].plot(ks, accuracy_all[0], linewidth=2.0, marker=".", markersize=12,label="PCA 80% var test")
axs[0,0].plot(ks, accuracy_all[1], linewidth=2.0, marker=".", markersize=12,label="PCA 90% var test")
axs[0,0].plot(ks, accuracy_all[2], linewidth=2.0, marker=".", markersize=12,label="PCA 95% var test")
axs[0,0].plot(ks, accuracy_all[3], linewidth=2.0, marker=".", markersize=12,label="PCA 99% var test")
axs[0,0].plot(ks, accuracy_all_train[0], linewidth=2.0, marker="^", markersize=12,label="PCA 80% var train")
axs[0,0].plot(ks, accuracy_all_train[1], linewidth=2.0, marker="^", markersize=12,label="PCA 90% var train")
axs[0,0].plot(ks, accuracy_all_train[2], linewidth=2.0, marker="^", markersize=12,label="PCA 95% var train")
axs[0,0].plot(ks, accuracy_all_train[3], linewidth=2.0, marker="^", markersize=12,label="PCA 99% var train")
axs[0,0].set(title="Accuracy for mixed data set", xlabel="K", ylabel="Accuracy", ylim=(ac_min, ac_max))
axs[0,0].legend()
axs[0,0].grid(True)


axs[0,1].plot(ks, accuracy_dis[0], linewidth=2.0, marker=".", markersize=12,label="PCA 80% var test")
axs[0,1].plot(ks, accuracy_dis[1], linewidth=2.0, marker=".", markersize=12,label="PCA 90% var test")
axs[0,1].plot(ks, accuracy_dis[2], linewidth=2.0, marker=".", markersize=12,label="PCA 95% var test")
axs[0,1].plot(ks, accuracy_dis[3], linewidth=2.0, marker=".", markersize=12,label="PCA 99% var test")
axs[0,1].plot(ks, accuracy_dis_train[0], linewidth=2.0, marker="^", markersize=12,label="PCA 80% var train")
axs[0,1].plot(ks, accuracy_dis_train[1], linewidth=2.0, marker="^", markersize=12,label="PCA 90% var train")
axs[0,1].plot(ks, accuracy_dis_train[2], linewidth=2.0, marker="^", markersize=12,label="PCA 95% var train")
axs[0,1].plot(ks, accuracy_dis_train[3], linewidth=2.0, marker="^", markersize=12,label="PCA 99% var train")
axs[0,1].set(title="Accuracy for disjunct data set", xlabel="K", ylabel="Accuracy", ylim=(ac_min, ac_max))
axs[0,1].legend()
axs[0,1].grid(True)

axs[1,0].plot(ks, comp_time_all[0], linewidth=2.0, marker=".", markersize=12,label="PCA 80% var test")
axs[1,0].plot(ks, comp_time_all[1], linewidth=2.0, marker=".", markersize=12,label="PCA 90% var test")
axs[1,0].plot(ks, comp_time_all[2], linewidth=2.0, marker=".", markersize=12,label="PCA 95% var test")
axs[1,0].plot(ks, comp_time_all[3], linewidth=2.0, marker=".", markersize=12,label="PCA 99% var test")
axs[1,0].plot(ks, comp_time_all_train[0], linewidth=2.0, marker="^", markersize=12,label="PCA 80% var train")
axs[1,0].plot(ks, comp_time_all_train[1], linewidth=2.0, marker="^", markersize=12,label="PCA 90% var train")
axs[1,0].plot(ks, comp_time_all_train[2], linewidth=2.0, marker="^", markersize=12,label="PCA 95% var train")
axs[1,0].plot(ks, comp_time_all_train[3], linewidth=2.0, marker="^", markersize=12,label="PCA 99% var train")
axs[1,0].set(title="Computation time for mixed data set", xlabel="K", ylabel="Computation time [s]", ylim=(ct_min, ct_max))
axs[1,0].legend()
axs[1,0].grid(True)

axs[1,1].plot(ks, comp_time_dis[0], linewidth=2.0, marker=".", markersize=12,label="PCA 80% var test")
axs[1,1].plot(ks, comp_time_dis[1], linewidth=2.0, marker=".", markersize=12,label="PCA 90% var test")
axs[1,1].plot(ks, comp_time_dis[2], linewidth=2.0, marker=".", markersize=12,label="PCA 95% var test")
axs[1,1].plot(ks, comp_time_dis[3], linewidth=2.0, marker=".", markersize=12,label="PCA 99% var test")
axs[1,1].plot(ks, comp_time_dis_train[0], linewidth=2.0, marker="^", markersize=12,label="PCA 80% var train")
axs[1,1].plot(ks, comp_time_dis_train[1], linewidth=2.0, marker="^", markersize=12,label="PCA 90% var train")
axs[1,1].plot(ks, comp_time_dis_train[2], linewidth=2.0, marker="^", markersize=12,label="PCA 95% var train")
axs[1,1].plot(ks, comp_time_dis_train[3], linewidth=2.0, marker="^", markersize=12,label="PCA 99% var train")
axs[1,1].set(title="Computation time for disjunct data set", xlabel="K", ylabel="Computation time [s]", ylim=(ct_min, ct_max))
axs[1,1].legend()
axs[1,1].grid(True)

plt.show()



