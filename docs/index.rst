.. ManipulationLibrary documentation master file, created by Tobias Strübing
   sphinx-quickstart on Tue Jul 21 17:10:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ManipulationLibrary's documentation!
===============================================
A module for Python 3 to work with eSEC matrices and extract e$^2$SEC matrices from manipulations. 
n order to analyse existing eSEC matrices, the "eSEC_analyser.py" module is used. In the example folder a Jupyter Notebook script is provided which shows how to:
* import the module
* read eSEc matricers from PDF
* plot the importance of rows
* check if any eSEC matrix is same as another one
* Remove combinations of relations (rows)
* plot a dendrogram and dissimilarity matrix for all manipulations
* plot a confusion matrix, classification accuacy with corresponding bar plot for e$^2$SEC matrices

The module "manipulation_analyser.py" is used to calculate e²SEC matrices using the MANIAC dataset. 
An example Jupyter Notebook along with point cloud data and corresponding labels is provided in the 
example folder. From this example manipulation the e$^2$SEC matrices are extracted and saved along with 
the debug images.

	``Disclaimer: This module is tested on Ubuntu 20.04.1 LTS with Python 3.8.5``

If you have trouble with the "filter_cython_new" function compile it on your system
with the following command in the "cython_filter_new" folder: ``python3 setup.py build_ext --inplace``

If the "filter_cython_new" can't be imported, comment out line 28 instead of line 29 in "manipulation_analyser.py".

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
	``pip3 install tabula-py matplotlib pandas numpy scipy seaborn open3d opencv-python cython progressbar2``

Code
====

	
.. toctree::
   :maxdepth: 2
   
   codi

   codi2


   



