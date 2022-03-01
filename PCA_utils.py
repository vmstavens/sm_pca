import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import time

def visualize_data(data_pts : np.ndarray, resx, resy, rows, cols):
    
    if len(data_pts) != rows * cols:
        print("Mismatch between datapoints and plot dimensions")
        return

    fig = plt.figure(figsize=(10,10))
    for i in range(len(data_pts)):
        ax = fig.add_subplot(rows, cols, i+1)
        title = "Ground truth: " + str(data_pts[i,1]) + " | student ID: " + str(data_pts[i,0])
        ax.set_title(title, color='blue', fontweight='bold')
        plt.imshow(data_pts[i,2:].reshape(resx,resy), cmap='gray')
        plt.tight_layout()

    plt.show()
    return 


def visualize_data_2(data_pts : np.ndarray, resx, resy, rows, cols):
    
    if len(data_pts) != rows * cols:
        print("Mismatch between datapoints and plot dimensions")
        return

    fig = plt.figure(figsize=(10,10))
    for i in range(len(data_pts)):
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(data_pts[i].reshape(resx,resy), cmap='gray')
        plt.tight_layout()

    plt.show()
    return 


def knn_classify(d_train, l_train, d_test, l_test, ks):
    n_test = len(d_test)

    predictions = np.ndarray((len(d_test),len(ks)))
    accuracy = np.ndarray((len(ks),))
    comp_time = np.ndarray((len(ks),))

    for i, k in enumerate(ks):
        KNNC = neighbors.KNeighborsClassifier(k, weights='uniform', algorithm='brute', metric='euclidean')
        KNNC.fit(d_train, l_train)
        t_start = time.time()
        predictions[:,i] = KNNC.predict(d_test)
        t_end = time.time()
        accuracy[i] = sum(predictions[:,i] == l_test)/n_test
        comp_time[i] = t_end - t_start
    
    return accuracy, predictions, comp_time

def knnParamSearch(data_train, labels_train, data_test, labels_test, ks, metrics, algorithm='brute'):
  n_test = len(data_test)
  results             = []
  methods             = []
  predictions         = []
  confusion_matrices  = []
  comp_times          = []
  for met in metrics:
    for k in ks:
      #We set the weights to uniform to obtain uniform importance of the k neighbors. Alternativly set this to "distance", to weight closer neighbors higher. 
      KNNC = neighbors.KNeighborsClassifier(k, weights='uniform', algorithm=algorithm, metric=met)
      KNNC.fit(data_train, labels_train)
      start = time.time()
      prediction = KNNC.predict(data_test)
      end = time.time()
      predictions.append(prediction)
      results.append(sum(prediction == labels_test)/n_test)
      methods.append([k, met])
      confusion_matrices.append(confusion_matrix(labels_test, prediction))
      comp_time = end - start
      comp_times.append(comp_time)

  return results, methods, predictions, confusion_matrices, comp_times

def split_data_all(data, train_pct):
    n = int(len(data) * train_pct)
    
    d_train = data[0:n,:]
    d_test = data[n:len(data)]

    l_train = d_train[:,1]
    l_test = d_test[:,1]

    d_train = np.delete(d_train,[0,1], 1)
    d_test = np.delete(d_test, [0,1], 1)

    return d_train, l_train, d_test, l_test


def split_data_disjunct(data, train_pct):

    sid_max = np.max(data[:,0])
    sid_n = int(train_pct * sid_max)

    d_train = data[data[:,0] <= sid_n]
    d_test = data[data[:,0] > sid_n]

    l_train = d_train[:,1]
    l_test = d_test[:,1]

    d_train = np.delete(d_train,[0,1], 1)
    d_test = np.delete(d_test, [0,1], 1)

    return d_train, l_train, d_test, l_test


def split_data_crossValidation(data, test_idx, splt_cnt=10):
    labels = data[:,1]
    data = np.delete(data, [0,1], 1)

    test_len = int(len(data) / splt_cnt)
    fst_idx = test_idx * test_len
    lst_idx = fst_idx + test_len

    d_test = data[fst_idx:lst_idx] 
    d_train = np.delete(data, range(fst_idx, lst_idx), 0)

    l_test = labels[fst_idx:lst_idx] 
    l_train = np.delete(labels, range(fst_idx, lst_idx))

    return d_train, l_train, d_test, l_test


def zeroCenter(dataset):
  mean = dataset.mean()
  std = dataset.std()
  dataset = dataset - mean
  dataset = dataset/std
  return dataset
