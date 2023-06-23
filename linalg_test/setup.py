# run from the command line via
# python setup.py install

from setuptools import setup
from torch.utils import cpp_extension

setup(name='custom_op_one',
      ext_modules=[cpp_extension.CppExtension('custom_op_one', ['custom_op_one.cpp'])],
      license='Apache License v2.0',
      cmdclass={'build_ext': cpp_extension.BuildExtension})
