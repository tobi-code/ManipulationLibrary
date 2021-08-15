from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy
import random

#*************************************************
# Availability: https://riptutorial.com/cython
#*************************************************
ext = Extension(name="filter_cython_new", sources=["filter_cython_new.pyx"])
setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])
