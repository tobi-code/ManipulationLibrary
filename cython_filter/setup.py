from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="filter_cython", sources=["filter_cython.pyx"])
setup(ext_modules=cythonize(ext))
 
