from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [ 
    Extension("fastmodules",
    ["acm/estimators/galaxy_clustering/src/fastmodules.pyx"],
    libraries=["m"],
    extra_compile_args = ["-ffast-math"],
    include_dirs=[numpy.get_include()]),
    Extension("minkowski",
    ["acm/estimators/galaxy_clustering/src/minkowski.pyx"],
    libraries=["m"],
    extra_compile_args = ["-ffast-math"],
    include_dirs=[numpy.get_include()]),
]

setup(
  name = "acm",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
