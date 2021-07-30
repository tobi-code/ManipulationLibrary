.. ManipulationLibrary documentation master file, created by Tobias Strübing
   sphinx-quickstart on Tue Jul 21 17:10:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ManipulationLibrary's documentation!
===============================================
The e²SEC[2] framework can be used to express human manipulation actions in a simple and concise way. 
Therefore, a manipulation in translated to a 24 x n matrix, which represents spatial relation between objects. 

Introduction:
-------------
ManipulationLibrary is a module for Python 3 to work with eSEC[1] matrices and extract e²SEC[2] matrices from manipulations. 
In order to analyse existing eSEC[1] matrices, the "eSEC_analyser.py" module is used. In the example folder a Jupyter Notebook script is provided which shows how to:

* import the module

* read eSEC[1] matricers from PDF

* plot the importance of rows

* check if any eSEC[1] matrix is same as another one

* Remove combinations of relations (rows)

* plot a dendrogram and dissimilarity matrix for all manipulations

* plot a confusion matrix, classification accuacy with corresponding bar plot for e²SEC[2] matrices

The module "manipulation_analyser.py" is used to calculate e²SEC[2] matrices using the MANIAC[3] dataset. 
An example Jupyter Notebook along with point cloud data and corresponding labels is provided in the 
example folder. From this example manipulation the e²SEC[2] matrices are extracted and saved along with 
the debug images. The example manipulation is taken from the MANAIC[3] dataset.

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

**********
References
**********

[1] Fatemeh Ziaeetabar, Tomas Kulvicius, Minija Tamosiunaite, and Florentin Wörgötter. Recognition and prediction of manipulation actions using enriched semantic event chains. Robotics and Autonomous Systems, 110:173–188, 2018.

[2] Tobias Strübing, Fatemeh Ziaeetabar, and Florentin Wörgötter. A summarized semantic structure to represent manipulation actions. Proceedings of the 16th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP. 370–379. 2021

[3] Eren Erdal Aksoy, Minija Tamosiunaite, and Florentin Wörgötter. Model-free incremental learning of the semantics of manipulation actions. Robotics and Autonomous Systems, 71:118–133, 2015.

Code
====

	
.. toctree::
   :maxdepth: 2
   
   codi

   codi2


   



