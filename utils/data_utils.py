from typing import Tuple
import numpy as np
from typing import List


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

def hest():
    print("HEJ")


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



def get_data_student_cifers(data: np.array, student_id: int, cifers: List, instances: int):
    """
    data: np.ndarray -> image data incl. studnet id (col 0) and labels (col 1)
    student_id: int -> the student id of the student you want the data from 
    cifers: List -> A list of the cifers of interrest, to extract from the data of the student. EX: [0,4,7] will give you cifers 0, 4, 7
    instances: int -> how many instances of each cifer to return 
    """

    data_cifers = []
    data_student = data[data[:,0] == student_id]
    for c in cifers:
        data_cifers.extend(data_student[data_student[:,1] == c][0:instances])
    data_cifers = np.array(data_cifers)
    return data_cifers


