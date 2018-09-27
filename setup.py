from numpy.distutils.core import setup


setup(name='openaerostruct',
    version='2.1.0',
    description='OpenAeroStruct',
    author='John Jasa',
    author_email='johnjasa@umich.edu',
    license='BSD-3',
    packages=[
        'openaerostruct',
        'openaerostruct/geometry',
        'openaerostruct/structures',
        'openaerostruct/aerodynamics',
        'openaerostruct/functionals',
        'openaerostruct/integration',
        'openaerostruct/common',
    ],
    # TODO: fix this with the correct requires
    install_requires=[],
    zip_safe=False,
    # ext_modules=ext,
    entry_points="""
    [console_scripts]
    plot_wing=openaerostruct.utils.plot_wing:disp_plot
    plot_wingbox=openaerostruct.utils.plot_wingbox:disp_plot
    """
)
