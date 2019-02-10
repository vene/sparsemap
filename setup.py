import os

import numpy

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

AD3_DIR = os.environ.get('AD3_DIR')
if not AD3_DIR:
    print("Warning: please set the AD3_DIR environment variable to point "
          "to the path where you have downloaded the AD3 library.")
    exit(1)

# PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))


ext_args = dict(
    # below is a hack so we can include ./ad3 as well as ../ad3
    libraries=['ad3'],
    extra_compile_args=["-std=c++11"],
    language="c++")


package_dir = {'sparsemap': 'python/sparsemap'}

setup(name="sparsemap",
      version="0.1.dev0",
      author="Vlad Niculae",
      author_email="vlad@vene.ro",
      package_dir=package_dir,
      packages=['sparsemap', 'sparsemap.layers_pt',
                'sparsemap.layers_pt.tests'],
      include_package_data=True,
      ext_modules=cythonize([
          Extension(
              "sparsemap._sparsemap",
              ["python/sparsemap/_sparsemap.pyx"],
              include_dirs=["src",
                            AD3_DIR,
                            os.path.join(AD3_DIR, 'python'),
                            numpy.get_include()],
              library_dirs=[os.path.join(AD3_DIR, 'ad3')],
              **ext_args),
          Extension(
              "sparsemap._factors",
              ["python/sparsemap/_factors.pyx",
               os.path.join("src", "lapjv", "lapjv.cpp"),
               os.path.join("src", "FactorTree.cpp"),
               ],
              include_dirs=["src",
                            AD3_DIR,
                            os.path.join(AD3_DIR, 'python'),
                            numpy.get_include()],
              library_dirs=[os.path.join(AD3_DIR, 'ad3')],
              **ext_args),
      ])
)
