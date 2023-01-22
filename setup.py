import sys
import setuptools
from setuptools import setup,Extension
from setuptools.command.build_ext import build_ext
from distutils.util import split_quoted


class CustomBuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        cpp_args = []
        if c == 'msvc':
            cpp_args.append('/std:c++20')
        elif c in {'unix','cygwin','mingw32'}:
            cpp_args.append('-std=c++20')

        for e in self.extensions:
            e.extra_compile_args = cpp_args

        super().build_extensions()


# If setup.py is run directly, it won't be obvious if the version of setuptools
# is too old.
try:
    from packaging.version import Version
    if (setuptools.__version__ != 'unknown'
        and Version(setuptools.__version__) < Version('61.0.0')):
        sys.exit('Requires setuptools version >= 61.\nCurrent version is %s.' % setuptools.__version__)
except:
    pass

setup(
    cmdclass={'build_ext' : CustomBuildExt},
    packages=[], # prevent automatic package discovery
    ext_modules=[Extension(
        'poly_ops',
        ['py/poly_ops.pyx'],
        include_dirs=['include'],
        language='c++',
        depends=[
            'include/poly_ops/poly_ops.hpp',
            'include/poly_ops/base.hpp'
            'include/poly_ops/offset.hpp',
            'include/poly_ops/strided_itr.hpp',
            'include/poly_ops/mini_flat_set.hpp'])],
    zip_safe=True)
