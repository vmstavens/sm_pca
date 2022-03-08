# Statistical Machinelearning Exercises

<details>
<summary><strong>General Information</strong></summary></br>

| Titles                                                      | Values                                                                         |
|-------------------------------------------------------------|--------------------------------------------------------------------------------|
| Course ID                                                   | T550031101                                                                     |
| Course ECTS                                                 | 5                                                                              |
| Group number                                                | 10                                                                             |
| Group members                                               | Joakim Stein, Thomas Agervig & Victor Staven                                   |
| Teacher                                                     | Norbert Krüger                                                                 |
| Instructor                                                  | Lakshadeep Naik                                                                |

</details>

## Overview

* [Overview](#overview)
* [Setup](#setup)
	+ [Download Data](#download-data)
	+ [Run Exercise Solutions](#run-exercise-solutions)
* [RepositoryStructure](#repository-structure)
* [Acknowledgments](#acknowledgments)

## Setup

### Download Data

The data applied in this project can be downloaded [here](https://drive.google.com/file/d/1Jk-uDbLiSbpBexvtzlzgX8-PFhHvpSzo/view?usp=sharing).

### Run Exercise Solutions

Each of the exercise solutions are Jupyter notebooks but can be converted to python scripts in the following manner
```bash
jupyter nbconvert --to python ex1.ipynb
```
Alternatively the notebooks can be opened and executed separately using texteditors like [Visual Studio Code](https://code.visualstudio.com/).

The Python version applied in this project is `Python 3.8.10`. You can check your python version in the following manner
```bash
python3 --version
```

### Dependencies

The python dependencies for this project 

- `numpy==1.22.2`
- `pandas==1.4.1`
- `matplotlib==3.1.2`
- `sklearn==0.0` 
- `cv2==4.5.5.62`
- `scipy==1.7.3`  

To install these the following command can be run using the `requirements.txt` file in the root folder
```bash
pip install -r requirements.txt
```

## Repository Structure

The structure of this project is build up of 7 modules
- 6 lecture modules 
- 1 utils module

Each of these have their own data (`data`) and image (`img`) directory, along with sub exercise solutions (`ex1.ipynb`,`ex2.ipynb`,`ex3.ipynb` and `ex4.ipynb`).

This can be seen in the structure below

<details>
<summary><strong>Structure<code></code></strong></summary></br>

```
.
├── bayes_classification
│   ├── data
│   ├── img
│   ├── constants.py
│   ├── ex1.ipynb
│   ├── ex2.ipynb
│   ├── ex3.ipynb
│   └── ex4.ipynb
├── clustering
│   ├── data
│   ├── ...
│   └── ex4.ipynb
├── decision_and_random_trees
│   ├── data
│   ├── ...
│   └── ex4.ipynb
├── knn
│   ├── data
│   ├── ...
│   └── ex4.ipynb
├── pca
│   ├── data
│   ├── ...
│   └── ex4.ipynb
├── svm_and_nn
│   ├── data
│   ├── ...
│   └── ex4.ipynb
├── utils
|   └── image_utils.py
└── README.md
```
</details>

## Acknowledgments

Thanks to Norbert Krüger and Lakshadeep Naik.

[semver]: http://semver.org/
[releases]: about:blank
[changelog]: CHANGELOG.md
[wiki]: about:blank

[ros]: http://wiki.ros.org/noetic
[gazebo]: http://gazebosim.org
[rosdep]: https://wiki.ros.org/rosdep
[vcstool]: https://github.com/dirk-thomas/vcstool
[catkin_tools]: https://catkin-tools.readthedocs.io
[export_fig]: https://se.mathworks.com/matlabcentral/fileexchange/23629-export_fig
[Ghostscript]: https://ghostscript.com/index.html

