from numpy.distutils.core import setup, Extension
import os
from subprocess import call


# setup(name='openaerostruct',
#     version='0.4.1',
#     description='OpenAeroStruct',
#     author='John Jasa',
#     author_email='johnjasa@umich.edu',
#     license='BSD-3',
#     packages=[
#         'openaerostruct',
#         'openaerostruct/geometry',
#         'openaerostruct/structures',
#         'openaerostruct/aerodynamics',
#         'openaerostruct/functionals',
#         'openaerostruct/integration',
#         'openaerostruct/fortran',
#     ],
#     # TODO: fix this with the correct requires
#     install_requires=[],
#     zip_safe=False,
#     # ext_modules=ext,
# )

setup(name='openaerostruct',
    version='0.4.1',
    description='openaerostruct',
    author='John Jasa',
    author_email='johnjasa@umich.edu',
    license='BSD-3',
    packages=[
        'openaerostruct',
        'openaerostruct/geometry',
        'openaerostruct/structures',
        'openaerostruct/aerodynamics',
        'openaerostruct/aerostruct',
        'openaerostruct/common',
        'openaerostruct/utils',
    ],
    # TODO: fix this with the correct requires
    install_requires=[],
    zip_safe=False,
    # ext_modules=ext,
)
