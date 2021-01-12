from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy
import random

# define an extension that will be cythonized and compiled
ext = Extension(name="filter_cython_new", sources=["filter_cython_new.pyx"])
setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])
 
setup(
    ext_modules=cythonize("filter_cython_new.pyx"),
    include_dirs=[numpy.get_include()]
)  
