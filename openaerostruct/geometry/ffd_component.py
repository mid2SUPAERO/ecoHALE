""" Manipulate geometry mesh based on high-level design parameters. """

from __future__ import division, print_function
import numpy as np
from pygeo import *

from openaerostruct.geometry.utils import \
    rotate, scale_x, shear_x, shear_y, shear_z, \
    sweep, dihedral, stretch, taper

from openmdao.api import ExplicitComponent
from openaerostruct.structures.utils import radii

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class GeometryMesh(ExplicitComponent):
    """
    OpenMDAO component that performs mesh manipulation functions. It reads in
    the initial mesh from the surface dictionary and outputs the altered
    mesh based on the geometric design variables.

    Depending on the design variables selected or the supplied geometry information,
    only some of the follow parameters will actually be given to this component.
    If parameters are not active (they do not deform the mesh), then
    they will not be given to this component.

    Parameters
    ----------
    twist[ny] : numpy array
        1-D array of rotation angles for each wing slice in degrees.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Modified mesh based on the initial mesh in the surface dictionary and
        the geometric design variables.
    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        self.surface = surface = self.metadata['surface']

        filename = write_FFD_file(surface)

        self.DVGeo = DVGeometry(filename)
        self.DVGeo.writePlot3d('debug.fmt')
        pts = surface['mesh'].reshape(-1, 3)

        self.DVGeo.addPointSet(pts, 'surface')
        # Associate a 'reference axis' for large-scale manipulation
        self.DVGeo.addRefAxis('wing_axis', xFraction=0.25, alignIndex='i')
        # Define a global design variable function:
        def twist(val, geo):
           geo.rot_y['wing_axis'].coef[:] = val[:]
        # Now add this as a global variable:
        self.DVGeo.addGeoDVGlobal('twist', 0.0, twist, lower=-10, upper=10)
        # Now add local (shape) variables
        self.DVGeo.addGeoDVLocal('shape', lower=-0.5, upper=0.5, axis='z')

        nx, ny = surface['mesh'].shape[:2]

        self.add_input('twist', val=0.)
        self.add_input('shape', val=np.zeros((nx*ny)))

        self.add_output('mesh', val=surface['mesh'])

        self.approx_partials('*', '*')

    def compute(self, inputs, outputs):
        surface = self.surface

        dvs = self.DVGeo.getValues()

        nx, ny = surface['mesh'].shape[:2]
        half = nx * ny

        dvs['twist'] = inputs['twist']
        dvs['shape'][:half] = inputs['shape']
        dvs['shape'][half:] = -inputs['shape']

        self.DVGeo.setDesignVars(dvs)
        coords = self.DVGeo.update('surface')

        mesh = coords.copy()

        nx, ny = surface['mesh'].shape[:2]

        mesh = mesh.reshape((nx, ny, 3))
        outputs['mesh'] = mesh


def write_FFD_file(surface):

    mesh = surface['mesh']

    nx, ny = mesh.shape[:2]

    half_ffd = mesh.copy()

    # Delete some of the y direction
    # half_ffd = half_ffd[:, [0, -1], :]

    # Delete some of the x direction
    # half_ffd = half_ffd[[0, -1], :, :]

    xmin, xmax = np.min(mesh[:, :, 0]), np.max(mesh[:, :, 0])
    ymin, ymax = np.min(mesh[:, :, 1]), np.max(mesh[:, :, 1])
    zmin, zmax = np.min(mesh[:, :, 2]), np.max(mesh[:, :, 2])

    cushion = 1.

    print(mesh.shape)
    print(half_ffd.shape)
    half_ffd[0, :, 0] = xmin - cushion
    half_ffd[-1, :, 0] = xmax + cushion
    half_ffd[:, 0, 1] = ymin - cushion
    half_ffd[:, -1, 1] = ymax + cushion

    bottom_ffd = half_ffd.copy()
    bottom_ffd[:, :, 2] = zmin - cushion

    top_ffd = half_ffd.copy()
    top_ffd[:, :, 2] = zmax + cushion

    ffd = np.vstack((bottom_ffd, top_ffd))

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = ffd[:, :, 0].flatten()
    ys = ffd[:, :, 1].flatten()
    zs = ffd[:, :, 2].flatten()

    ax.scatter(xs, ys, zs, c='red')

    xs = mesh[:, :, 0].flatten()
    ys = mesh[:, :, 1].flatten()
    zs = mesh[:, :, 2].flatten()

    ax.scatter(xs, ys, zs, c='blue')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

    print(ffd)

    filename = surface['name'] + '_ffd.fmt'

    with open(filename, 'w') as f:
        f.write('1\n')
        f.write('{} {} {}\n'.format(nx, 2, ny))
        x = np.array_str(ffd[:, :, 0].flatten(order='F'))[1:-1] + '\n'
        y = np.array_str(ffd[:, :, 1].flatten(order='F'))[1:-1] + '\n'
        z = np.array_str(ffd[:, :, 2].flatten(order='F'))[1:-1] + '\n'

        f.write(x)
        f.write(y)
        f.write(z)

    return filename
