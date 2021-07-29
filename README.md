# ManipulationLibrary

The e²SEC[2] framework can be used to express human manipulation actions in a simple and concise way. Therefore, a manipulation in translated to a 24 x n matrix, which represents spatial relation between objects. 

Introduction:
-------------
[![](https://img.shields.io/badge/docs-blue.svg)](https://manipulationlibrary.readthedocs.io/en/latest/)

ManipulationLibrary is a module for Python 3 to work with eSEC[1] matrices and extract e²SEC[2] matrices from manipulations. 

You can find examples on how to use this library in the "example" folder in GitHub. The example manipulation is taken from the MANAIC dataset [3].

	Disclaimer: This module is tested on Ubuntu 20.04.1 LTS with Python 3.8.5

If you have trouble with the "filter_cython_new" function compile it on your system
with the following command in the "cython_filter_new" folder: ``python3 setup.py build_ext --inplace``

If the "filter_cython_new" can't be imported, comment out line 28 instead of line 29 in "manipulation_analyser.py".

How to use:
-----------

In order to analyse existing eSEC matrices, the "eSEC_analyser.py" module is used. In the example folder a Jupyter Notebook script is provided which shows how to:
* import the module
* read eSEc matricers from PDF
* plot the importance of rows
* check if any eSEC matrix is same as another one
* Remove combinations of relations (rows)
* plot a dendrogram and dissimilarity matrix for all manipulations
* plot a confusion matrix, classification accuacy with corresponding bar plot for e²SEC matrices

The module "manipulation_analyser.py" is used to calculate e²SEC matrices using the MANIAC dataset. An example Jupyter Notebook along with point cloud data and corresponding labels is provided in the example folder. From this example manipulation the e²SEC matrices are extracted and saved along with the debug images.

Depends on external libraries:
------------------------------

* tabula

* progressbar

* pandas

* numpy

* matplotlib

* itertools

* scipy

* seaborn

* open3d

* OpenCV

* Cython

Installation of libraries:
--------------------------

To install these libraries use the following command:
	``pip3 install tabula matplotlib pandas numpy scipy seaborn open3d opencv-python cython progressbar2``

See the documentation at https://manipulationlibrary.readthedocs.io/en/latest/ for futher explanation of all functions.

###### [1] Fatemeh Ziaeetabar, Tomas Kulvicius, Minija Tamosiunaite, and Florentin Wörgötter (2018). Recognition and prediction of manipulation actions using enriched semantic event chains. Robotics and Autonomous Systems.
###### [2] Tobias Strübing, Fatemeh Ziaeetabar, and Florentin Wörgötter (2021). A summarized semantic structure to represent manipulation actions.
###### [3] Eren Erdal Aksoy, Minija Tamosiunaite, and Florentin Wörgötter. Model-free incremental learning of the semantics of manipulation actions. Robotics and Autonomous Systems.