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
        self.inds = coords[:, 0, :]
        self.inds2 = coords[:, 1, :]

        self.add_input('twist', val=0.)
        self.add_input('shape', val=np.zeros((self.mx, self.my)))

        self.add_output('mesh', val=surface['mesh'])

    def compute(self, inputs, outputs):
        surface = self.surface

        dvs = self.DVGeo.getValues()

        nx, ny = surface['mesh'].shape[:2]

        dvs['twist'] = inputs['twist']

        for i, row in enumerate(self.inds):
            for j, ind in enumerate(row):
                ind2 = self.inds2[i, j]
                dvs['shape'][ind] = inputs['shape'][i, j]
                dvs['shape'][ind2] = inputs['shape'][i, j]

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

        for i, ind in enumerate(self.inds.flatten()):
            ind2 = self.inds2.flatten()[i]
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

def write_FFD_file(surface, mx, my):

    mesh = surface['mesh']
    nx, ny = mesh.shape[:2]

    half_ffd = np.zeros((mx, my, 3))

    LE = mesh[0, :, :]
    TE = mesh[-1, :, :]

    half_ffd[0, :, 0] = np.interp(np.linspace(0, 1, my), np.linspace(0, 1, ny), LE[:, 0])
    half_ffd[0, :, 1] = np.interp(np.linspace(0, 1, my), np.linspace(0, 1, ny), LE[:, 1])
    half_ffd[0, :, 2] = np.interp(np.linspace(0, 1, my), np.linspace(0, 1, ny), LE[:, 2])

    half_ffd[-1, :, 0] = np.interp(np.linspace(0, 1, my), np.linspace(0, 1, ny), TE[:, 0])
    half_ffd[-1, :, 1] = np.interp(np.linspace(0, 1, my), np.linspace(0, 1, ny), TE[:, 1])
    half_ffd[-1, :, 2] = np.interp(np.linspace(0, 1, my), np.linspace(0, 1, ny), TE[:, 2])

    for i in range(my):
        half_ffd[:, i, 0] = np.linspace(half_ffd[0, i, 0], half_ffd[-1, i, 0], mx)
        half_ffd[:, i, 1] = np.linspace(half_ffd[0, i, 1], half_ffd[-1, i, 1], mx)
        half_ffd[:, i, 2] = np.linspace(half_ffd[0, i, 2], half_ffd[-1, i, 2], mx)

    cushion = .5

    half_ffd[0, :, 0] -= cushion
    half_ffd[-1, :, 0] += cushion
    half_ffd[:, 0, 1] -= cushion
    half_ffd[:, -1, 1] += cushion

    bottom_ffd = half_ffd.copy()
    bottom_ffd[:, :, 2] -= cushion

    top_ffd = half_ffd.copy()
    top_ffd[:, :, 2] += cushion

    ffd = np.vstack((bottom_ffd, top_ffd))

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # fig = plt.figure()
    # axes = []
    #
    # axes.append(fig.add_subplot(221, projection='3d'))
    # axes.append(fig.add_subplot(222, projection='3d'))
    # axes.append(fig.add_subplot(223, projection='3d'))
    # axes.append(fig.add_subplot(224, projection='3d'))
    #
    # for i, ax in enumerate(axes):
    #     xs = ffd[[2, 5], :, 0].flatten()
    #     ys = ffd[[2, 5], :, 1].flatten()
    #     zs = ffd[[2, 5], :, 2].flatten()
    #
    #     ax.scatter(xs, ys, zs, c='red', alpha=1., clip_on=False)
    #
    #     xs = ffd[[0, 1, 3, 4], :, 0].flatten()
    #     ys = ffd[[0, 1, 3, 4], :, 1].flatten()
    #     zs = ffd[[0, 1, 3, 4], :, 2].flatten()
    #
    #     ax.scatter(xs, ys, zs, c='blue', alpha=1.)
    #
    #     xs = mesh[:, :, 0]
    #     ys = mesh[:, :, 1]
    #     zs = mesh[:, :, 2]
    #
    #     ax.plot_wireframe(xs, ys, zs, color='k')
    #
    #     ax.set_xlim([-5, 5])
    #     ax.set_ylim([-5, 5])
    #     ax.set_zlim([-5, 5])
    #
    #     ax.set_xlim([20, 40])
    #     ax.set_ylim([-25, -5.])
    #     ax.set_zlim([-10, 10])
    #
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #
    #     ax.set_axis_off()
    #
    #     ax.set_axis_off()
    #
    #     if i == 0:
    #         ax.view_init(elev=0, azim=180)
    #     elif i == 1:
    #         ax.view_init(elev=0, azim=90)
    #     elif i == 2:
    #         ax.view_init(elev=100000, azim=0)
    #     else:
    #         ax.view_init(elev=40, azim=-30)
    #
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
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
