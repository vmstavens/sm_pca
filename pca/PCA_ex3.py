from cv2 import blur
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import PCA_utils
from scipy.ndimage import gaussian_filter
from scipy import stats
import image_utils


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



#Extracting random images from the data and visualizing these
random_indices = np.random.choice(data.shape[0], size=3, replace=False)
random_images = data[random_indices, :]
PCA_utils.visualize_images(random_images, 1, 3, "Original data")


#Creating gaussian filter, smoothing images and visualizing images
blurred_images = []
for i in range(random_images.shape[0]):
    blurred_images.append(image_utils.blur_data(random_images[i], sigma=1)) 
blurred_images = np.array(blurred_images)
PCA_utils.visualize_images(blurred_images,1, 3, "Blurred images with sigma=1")


#Values for hyperparameters found in ex1 and with a sigma choosen arbitrary to 1:
sigma = 1
k = [1]
pct = 0.8

#Creating blurred data set
blurred_data = []
for i in range(data.shape[0]):
    blurred_data.append(image_utils.blur_data(data[i],sigma=sigma))
blurred_data = np.array(blurred_data)

#Visualizing the PCA data, the images are blurred before PCA is applied, in this example all PCs are used:
pca_visualization = PCA()
pca_vis_data = pca_visualization.fit_transform(blurred_data[:, 2:])
pca_vis_data = np.concatenate((blurred_data[:,0:2], pca_vis_data), axis=1)
PCA_utils.visualize_images(pca_vis_data[random_indices], 1, 3, "PCA data from blurred images")




#applying PCA to blurred dataset with 80% of the total variance
pca = PCA(n_components=0.8, svd_solver='full')
pca_data = pca.fit_transform(blurred_data[:, 2:])
#concatenating student-ID and labels back on the PCA dataset
pca_data = np.concatenate((blurred_data[:,0:2], pca_data), axis=1)




#Performing 10 fold cross validation on PCA data
accuracies = []
comp_time = []

for i in range(10):
    print("Sigma: ", sigma, "   I: ", i)
    data_train, label_train, data_test, label_test = PCA_utils.split_data_crossValidation(pca_data, i)
    acc,_,_,_,time = PCA_utils.knnParamSearch(data_train, label_train, data_test, label_test, k, metrics=['euclidean'])
    accuracies.append(acc)
    comp_time.append(time)

#Visualizing the results
mean_acc = [np.mean(accuracies)] * 10
mean_comp_time = [np.mean(comp_time)] * 10
i = np.arange(1,11)

fig, ax = plt.subplots(2,1)

ax[0].plot(i, accuracies, linewidth= 2.0, marker='.', markerfacecolor ='orange', color='blue', markersize = 12, label = "Accuracy")
ax[0].plot(i, mean_acc, linewidth=2.0, color='blue', label = "Mean Accuracy")
ax[0].set(title = "Accuracy", xlabel = "Test section", ylabel = "Accuracy")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(i, comp_time, linewidth= 2.0, marker='.', markerfacecolor ='blue', color='orange', markersize = 12, label = "Computation time")
ax[1].plot(i, mean_comp_time, linewidth = 2.0, color ='orange', label = "Mean computational time")
ax[1].set(title = "Computational time", xlabel = "Test section", ylabel = "Time[s]")
ax[1].grid(True)
ax[1].legend()
plt.show()

plt.savefig("blurred_pca.png")

