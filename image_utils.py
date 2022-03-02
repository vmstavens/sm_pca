import numpy as np
from scipy.ndimage import gaussian_filter
from constants import *

#Gaussian smoothing of data from statistical machine learning. 
#Extracts the label and student id, then converts the flattened image into a 2D 23x23 image and blurs the image
#The image is then converted back 2 1D and the student ID and label is again added to the row vector.
#input: 1 row of the data that contains studentID, label, Data
def blur_data(img, sigma):
    student_id = img[0]
    label = img[1]
    img = np.delete(img, 0, axis=0)
    img = np.delete(img, 0, axis=0)
    img = img.reshape(IMG_ROWS, IMG_COLS)
    gaussian_filter(img, sigma=sigma)
    img.reshape(IMG_ROWS * IMG_COLS)
    img = np.insert(img, 0, label, axis=0)
    img = np.insert(img, 0 , student_id, axis=0)
    return img
