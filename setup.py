from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(
    ext_modules=cythonize([
        "option_models_fast/Binomials.pyx",
        "option_models_fast/Trinomials.pyx"
    ]),
    include_dirs=[np.get_include()]
)

# run 
# python setup.py build_ext --inplace 
# in terminal