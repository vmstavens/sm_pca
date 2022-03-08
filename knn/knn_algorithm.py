import numpy as np
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
import time
from typing import List, Tuple

def knnParamSearch(data_train:np.ndarray, labels_train:np.ndarray, data_test:np.ndarray, labels_test:np.ndarray, ks:List[int], metrics:List[str], algorithm:str='brute') -> Tuple[list, list, list, list, list]:
  """
  data_train, data_test : np.ndarray -> train and test data without student id and labels
  labes_train, labels_test : np.ndarray -> labels for the training and test data
  ks : List[int] -> list of K's to run the algorithm on.
  metrics : List[str] -> list of distance metrics, e.g. 'euclidean', 'manhattan' etc.
  algorithm : str = 'brute' -> what algorithm to use.
  """
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