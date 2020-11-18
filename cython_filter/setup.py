from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
ext = Extension(name="filter_cython", sources=["filter_cython.pyx"])
setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])
 
setup(
    ext_modules=cythonize("filter_cython.pyx"),
    include_dirs=[numpy.get_include()]
)  