from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

np.random.seed(314)

class DisplacementTransfer(ExplicitComponent):
    """
    Perform displacement transfer.

    Apply the computed displacements on the original mesh to obtain
    the deformed mesh.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces.
    disp[ny, 6] : numpy array
        Flattened array containing displacements on the FEM component.
        Contains displacements for all six degrees of freedom, including
        displacements in the x, y, and z directions, and rotations about the
        x, y, and z axes.

    Returns
    -------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']
        self.nx = surface['num_x']

        self.add_input('mesh', val=np.zeros((self.nx, self.ny, 3)), units='m')
        self.add_input('disp', val=np.zeros((self.ny, 6)), units='m')
        self.add_input('transformation_matrix', val=np.zeros((self.ny, 3, 3)))
        self.add_input('ref_curve', val=np.zeros((self.ny, 3)), units='m')

        self.add_output('def_mesh', val=np.random.random_sample((self.nx, self.ny, 3)), units='m')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        mesh = inputs['mesh']
        disp = inputs['disp']

        # Get the location of the spar
        ref_curve = inputs['ref_curve']

        # Compute the distance from each mesh point to the nodal spar points
        Smesh = np.zeros(mesh.shape, dtype=data_type)
        for ind in range(self.nx):
            Smesh[ind, :, :] = mesh[ind, :, :] - ref_curve

        # Set up the mesh displacements array
        mesh_disp = np.zeros(mesh.shape, dtype=data_type)
        cos, sin = np.cos, np.sin

        # Loop through each spanwise FEM element
        for ind in range(self.ny):
            dx, dy, dz, rx, ry, rz = disp[ind, :]

            # 1 eye from the axis rotation matrices
            # -3 eye from subtracting Smesh three times
            T = -2 * np.eye(3, dtype=data_type)
            T[ 1:,  1:] += [[cos(rx), -sin(rx)], [ sin(rx), cos(rx)]]
            T[::2, ::2] += [[cos(ry),  sin(ry)], [-sin(ry), cos(ry)]]
            T[ :2,  :2] += [[cos(rz), -sin(rz)], [ sin(rz), cos(rz)]]

            # Obtain the displacements on the mesh based on the spar response
            mesh_disp[:, ind, :] += np.dot(T, Smesh[:, ind, :].T).T
            mesh_disp[:, ind, 0] += dx
            mesh_disp[:, ind, 1] += dy
            mesh_disp[:, ind, 2] += dz

        # Apply the displacements to the mesh
        def_mesh = mesh + mesh_disp

        outputs['def_mesh'] = def_mesh
