.. ManipulationLibrary documentation master file, created by Tobias Strübing
   sphinx-quickstart on Tue Jul 21 17:10:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ManipulationLibrary's documentation!
===============================================
A module for Python 3 to produce and work with eSEC matrices. 
The eSEC analyser is used to analyse existing eSEC matrices and is capable of:

* read a list of eSEC matrices

* find the importance of up to 5 rows

* remove rows 

* calculate dissimilarity matrices and dendrograms

* measure the group dissimilarity

The manipulation analyser is used calculate e²SEC matrices using the MANIAC dataset.

You can find examples on how to use this library in the "example" folder in GitHub.

	``Disclaimer: This module is tested on Ubuntu 20.04.1 LTS with Python 3.8.5``

If you have trouble with the "filter_cython" function you can try to compile it with the
following command in the "cython_filter" folder: ``python3 setup.py build_ext --inplace``

You can find a detailed documentation here: ``https://manipulationlibrary.readthedocs.io/en/latest/``

Depends on external librabies:
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

Installation of librabies:
--------------------------

To install these librabies use the following command:
	``pip3 install tabula-py matplotlib pandas numpy scipy seaborn open3d opencv-python cython progressbar2``

Code
====

	
.. toctree::
   :maxdepth: 2
   
   codi

   codi2


   



