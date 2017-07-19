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
        self.mx, self.my = self.surface['mx'], self.surface['my']

        filename = write_FFD_file(surface, self.mx, self.my)

        self.DVGeo = DVGeometry(filename)
        self.DVGeo.writePlot3d('debug.fmt')
        pts = surface['mesh'].reshape(-1, 3)

        self.DVGeo.addPointSet(pts, 'surface')
        # Associate a 'reference axis' for large-scale manipulation
        self.DVGeo.addRefAxis('wing_axis', xFraction=0.25, alignIndex='i')
        # Define a global design variable function:
        def twist(val, geo):
           geo.rot_y['wing_axis'].coef[:] = val[:]

        # Now add local (shape) variables
        self.DVGeo.addGeoDVLocal('shape', lower=-0.5, upper=0.5, axis='z')

        coords = self.DVGeo.getLocalIndex(0)
        self.inds = coords[:, 0, :].flatten()
        self.inds2 = coords[:, 1, :].flatten()

        self.add_input('twist', val=0.)
        self.add_input('shape', val=np.zeros((self.mx*self.my)))

        self.add_output('mesh', val=surface['mesh'])

    def compute(self, inputs, outputs):
        surface = self.surface

        dvs = self.DVGeo.getValues()

        nx, ny = surface['mesh'].shape[:2]

        dvs['twist'] = inputs['twist']

        for i, ind in enumerate(self.inds):
            ind2 = self.inds2[i]
            dvs['shape'][ind] = inputs['shape'][i]
            dvs['shape'][ind2] = inputs['shape'][i]

        self.DVGeo.setDesignVars(dvs)
        coords = self.DVGeo.update('surface')

        mesh = coords.copy()
        mesh = mesh.reshape((nx, ny, 3))
        outputs['mesh'] = mesh

    def compute_partials(self, inputs, outputs, partials):
        self.DVGeo.computeTotalJacobian('surface')
        jac = self.DVGeo.JT['surface'].toarray().T
        my_jac = partials['mesh', 'shape']
        my_jac[:, :] = 0.

        for i, ind in enumerate(self.inds):
            ind2 = self.inds2[i]
            my_jac[:, i] += jac[:, ind]
            my_jac[:, i] += jac[:, ind2]

        partials['mesh', 'shape'] = my_jac

def view_mat(mat):
    """ Helper function used to visually examine matrices. """
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = np.sum(mat, axis=2)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

def plot_3d_points(half_mesh):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    right_mesh = half_mesh.copy()
    right_mesh[:, :, 1] *= -1
    mesh = np.hstack((half_mesh, right_mesh[:, ::-1, :]))

    xs = half_mesh[:, :, 0]
    ys = half_mesh[:, :, 1]
    zs = half_mesh[:, :, 2]
    ax.plot_wireframe(xs, ys, zs, color='k')

    xs = right_mesh[:, :, 0]
    ys = right_mesh[:, :, 1]
    zs = right_mesh[:, :, 2]
    ax.plot_wireframe(xs, ys, zs, color='k')

    ax.set_axis_off()

    ax.set_xlim([10, 60])
    ax.set_ylim([-25, 25])
    ax.set_zlim([-25, 25])

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()

def write_FFD_file(surface, mx, my):

    mesh = surface['mesh']

    xmin, xmax = np.min(mesh[:, :, 0]), np.max(mesh[:, :, 0])
    ymin, ymax = np.min(mesh[:, :, 1]), np.max(mesh[:, :, 1])
    zmin, zmax = np.min(mesh[:, :, 2]), np.max(mesh[:, :, 2])

    cushion = 1.

    xlins = np.linspace(xmin - cushion, xmax + cushion, mx)
    ylins = np.linspace(ymin - cushion, ymax + cushion, my)

    xv, yv = np.meshgrid(xlins, ylins)

    half_ffd = np.zeros((mx, my, 3))

    half_ffd[:, :, 0] = xv.T
    half_ffd[:, :, 1] = yv.T

    bottom_ffd = half_ffd.copy()
    bottom_ffd[:, :, 2] = zmin - cushion

    top_ffd = half_ffd.copy()
    top_ffd[:, :, 2] = zmax + cushion

    ffd = np.vstack((bottom_ffd, top_ffd))

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # xs = ffd[:, :, 0].flatten()
    # ys = ffd[:, :, 1].flatten()
    # zs = ffd[:, :, 2].flatten()
    #
    # ax.scatter(xs, ys, zs, c='red')
    #
    # xs = mesh[:, :, 0].flatten()
    # ys = mesh[:, :, 1].flatten()
    # zs = mesh[:, :, 2].flatten()
    #
    # ax.scatter(xs, ys, zs, c='blue')
    #
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-5, 5])
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # plt.show()

    filename = surface['name'] + '_ffd.fmt'

    with open(filename, 'w') as f:
        f.write('1\n')
        f.write('{} {} {}\n'.format(mx, 2, my))
        x = np.array_str(ffd[:, :, 0].flatten(order='F'))[1:-1] + '\n'
        y = np.array_str(ffd[:, :, 1].flatten(order='F'))[1:-1] + '\n'
        z = np.array_str(ffd[:, :, 2].flatten(order='F'))[1:-1] + '\n'

        f.write(x)
        f.write(y)
        f.write(z)

    return filename
