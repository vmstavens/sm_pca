from typing import Tuple
import numpy as np


def split_data_all(data:np.ndarray, train_pct:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    data : np.ndarray -> image data incl. student id (col 0) and labels (col 1)
    train_pct : float -> [0,1] indicating how much is training data. E.g. 0.8 -> 80% train, 20% test
    """
    n = int(len(data) * train_pct)
    
    d_train = data[0:n,:]
    d_test = data[n:len(data)]

    l_train = d_train[:,1]
    l_test = d_test[:,1]

    d_train = np.delete(d_train,[0,1], 1)
    d_test = np.delete(d_test, [0,1], 1)

    return d_train, l_train, d_test, l_test


def split_data_disjunct(data:np.ndarray, train_pct:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    data : np.ndarray -> image data incl. student id (col 0) and labels (col 1)
    train_pct : float -> [0,1] indicating how much is training data. E.g. 0.8 -> roughly 80% train, 20% test depending on the amount of students
    """
    sid_max = np.max(data[:,0])
    sid_n = int(train_pct * sid_max)

    d_train = data[data[:,0] <= sid_n]
    d_test = data[data[:,0] > sid_n]

    l_train = d_train[:,1]
    l_test = d_test[:,1]

    d_train = np.delete(d_train,[0,1], 1)
    d_test = np.delete(d_test, [0,1], 1)

    return d_train, l_train, d_test, l_test


def split_data_crossValidation(data:np.ndarray, test_idx:int, splt_cnt:int=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    data : np.ndarray -> image data incl. student id (col 0) and labels (col 1)
    train_idx : int   -> index of what section of the set will be test data. train_idx < splt_cnt
    split_cnt : int = 10 -> How many sections to split the data into
    """
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