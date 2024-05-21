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
    extra_compile_args = ['-O3','-ffast-math'],
    include_dirs=[numpy.get_include()]),
    Extension("pydive",
        sources=['acm/estimators/galaxy_clustering/src/pydive.pyx',
                  ],
        include_dirs=[numpy.get_include(), 
                      '/global/u1/d/dforero/lib/CGAL-5.4/include', 
                      ],
                      
        library_dirs=[
                      ],
        libraries=['m', 'gsl', 'gslcblas', 'gmp', 'mpfr'],
        language='c++',
        extra_compile_args=['-fPIC', '-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]




setup(
  name = "acm",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  install_requires=[
        'numpy',
        'kymatio',
    ],)
