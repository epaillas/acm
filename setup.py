from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

import subprocess
subprocess.call(["make"])

galaxy_clustering_src_name = "acm.estimators.galaxy_clustering.src"
galaxy_clustering_src_path = "acm/estimators/galaxy_clustering/src"

fastmodules = Extension(
    name = galaxy_clustering_src_name + ".fastmodules",
    sources = [galaxy_clustering_src_path + "/fastmodules.pyx"],
    libraries = ["m"],
    extra_compile_args = ["-ffast-math"],
    include_dirs = [numpy.get_include()],
)

minkowski = Extension(
    name = galaxy_clustering_src_name + ".minkowski",
    sources = [galaxy_clustering_src_path + "/minkowski.pyx"],
    libraries = ["m"],
    extra_compile_args = ['-O3', '-ffast-math'],
    include_dirs = [numpy.get_include()],
)

pydive = Extension(
    name = galaxy_clustering_src_name + ".pydive",
    sources = [galaxy_clustering_src_path + "/pydive.pyx"],
    include_dirs = [
        numpy.get_include(), 
        '/global/u1/d/dforero/lib/CGAL-5.4/include',
    ],
    libraries = ['m', 'gsl', 'gslcblas', 'gmp', 'mpfr'],
    language = 'c++',
    extra_compile_args = ['-fPIC', '-fopenmp'],
    extra_link_args = ['-fopenmp']
)

extensions = [
    fastmodules, 
    minkowski,
    # pydive, #include_dirs private
]

print(f"Building extensions: {[ext.name.split('.')[-1] for ext in extensions]}")

setup(
    ext_modules = cythonize(extensions),
)