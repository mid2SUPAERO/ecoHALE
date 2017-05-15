from numpy.distutils.core import setup, Extension
import os
from subprocess import call


setup(name='openaerostruct',
    version='0.1',
    description='The Surrogate Model Toolbox (SMT)',
    author='Mohamed Amine Bouhlel',
    author_email='mbouhlel@umich.edu',
    license='BSD-3',
    packages=[
        'openaerostruct',
        'openaerostruct/geometry',
    ],
    install_requires=[],
    zip_safe=False,
    # ext_modules=ext,
)
