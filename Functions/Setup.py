from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy

setup(
    include_dirs = [cython_gsl.get_include(), numpy.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("Functions_c",
                             ["Functions_c.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()])]
    )

#run this file using the command 'python Setup.py build_ext --inplace' this must be run from inside functions