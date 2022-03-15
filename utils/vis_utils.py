import numpy as np
import matplotlib.pyplot as plt
from typing import List

def visualize_generic(data:np.ndarray, rows:int, cols:int, fig_title:str, sub_titles: List, img_name="img.png", save_img:bool = False) -> None:
    """
    data : np.ndarray -> list of images or a single image
    rows: int -> the number of rows in the image showing matrix, if only one image rows = 1
    cols: int -> the number of cols in the image showing matrix, if only one image cols = 1
    fig_title: str -> the title of the plot
    sub_titles: List -> list of sub plot titles, if only 1 image is given, this is ignored 
    """

    # print("data shape = ",data.shape)
    # print("test", int(np.sqrt(len(data[0]))))

    # in case of only one image is desired
    if rows == 1 and cols == 1:
        fig = plt.figure(figsize=(10, 10))
        img_rows = int(np.sqrt(len(data)))
        img_cols = img_rows
        plt.title(fig_title, color="blue", fontweight="bold")
        plt.imshow(data.reshape(img_rows, img_cols), cmap="gray")
        plt.tight_layout()
        plt.savefig(img_name)
        plt.show()
        return

    # multiple images
    # fig = plt.figure()
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(fig_title, fontsize=16, color="blue", fontweight="bold")
    number_of_img = data.shape[0]
    img_rows = int(np.sqrt(len(data[0])))
    img_cols = img_rows
    for i in range(1, number_of_img + 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.set_title(sub_titles[i-1], color="blue", fontweight="bold")
        plt.imshow(data[i-1].reshape(img_rows, img_cols), cmap="gray")
        plt.tight_layout()
    if save_img == True:
        plt.savefig(img_name)
    plt.show()