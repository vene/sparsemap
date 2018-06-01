import os

import numpy

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ad3_dir = os.environ.get('AD3_DIR')
if not ad3_dir:
    print("Warning: please set the AD3_DIR environment variable to point"
          "to the path where you have downloaded the AD3 library.")
    exit(1)


ext_args = dict(
    # below is a hack so we can include ./ad3 as well as ../ad3
    libraries=['ad3'],
    extra_compile_args=["-std=c++11"],
    language="c++")


setup(name="sparsemap",
      version="0.1.dev0",
      author="Vlad Niculae",
      author_email="vlad@vene.ro",
      ext_modules=cythonize([
          Extension(
              "sparsemap._sparsemap",
              ["sparsemap/_sparsemap.pyx"],
              include_dirs=["../src", ad3_dir, os.path.join(ad3_dir, 'python'),
                            numpy.get_include()],
              library_dirs=[os.path.join(ad3_dir, 'ad3')],
              **ext_args),
          Extension(
              "sparsemap._factors",
              ["sparsemap/_factors.pyx",
               "../src/lapjv/lapjv.cpp",
               os.path.join(ad3_dir, 'examples', 'cpp', 'parsing', 'FactorTree.cpp')
               ],
              include_dirs=["../src", ad3_dir, os.path.join(ad3_dir, 'python'),
                            numpy.get_include()],
              library_dirs=[os.path.join(ad3_dir, 'ad3')],
              **ext_args),
      ])
)
