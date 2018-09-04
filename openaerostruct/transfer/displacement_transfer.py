from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import get_array_indices, compute_cross, compute_cross_deriv1, compute_cross_deriv2


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

        self.add_input('mesh', val=np.ones((self.nx, self.ny, 3)), units='m')
        self.add_input('disp', val=np.ones((self.ny, 6)), units='m')
        self.add_input('transformation_matrix', val=np.ones((self.ny, 3, 3)))
        self.add_input('ref_curve', val=np.ones((self.ny, 3)), units='m')

        self.add_output('def_mesh', val=np.random.random_sample((self.nx, self.ny, 3)), units='m')

        disp_indices = get_array_indices(self.ny, 6)
        axis_indices = get_array_indices(self.ny, 3)
        mesh_indices = get_array_indices(self.nx, self.ny, 3)
        transform_indices = get_array_indices(self.ny, 3, 3)
        mesh_disp_indices = get_array_indices(self.nx, self.ny, 3)

        rows = mesh_disp_indices.flatten()
        cols = np.einsum('i,jk->ijk', np.ones(self.nx), disp_indices[:, :3]).flatten()
        self.declare_partials('def_mesh', 'disp', val=1., rows=rows, cols=cols)

        rows = np.einsum('ijk,l->ijkl', mesh_disp_indices, np.ones(3, int)).flatten()
        cols = np.einsum('ik,jl->ijkl', np.ones((self.nx, 3), int), axis_indices).flatten()
        self.declare_partials('def_mesh', 'ref_curve', rows=rows, cols=cols)

        rows = np.einsum('ijk,l->ijkl', mesh_disp_indices, np.ones(3, int)).flatten()
        cols = np.einsum('ijl,k->ijkl', mesh_indices, np.ones(3, int)).flatten()
        self.declare_partials('def_mesh', 'mesh', rows=rows, cols=cols)

        rows = np.einsum('ijl,k->ijkl', mesh_disp_indices, np.ones(3, int)).flatten()
        cols = np.einsum('jlk,i->ijkl', transform_indices, np.ones(self.nx, int)).flatten()
        self.declare_partials('def_mesh', 'transformation_matrix', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        mesh = inputs['mesh']
        disp = inputs['disp']

        # Get the location of the spar
        ref_curve = inputs['ref_curve']

        outputs['def_mesh'] = mesh
        outputs['def_mesh'] += np.einsum('i,jk->ijk',
            np.ones(self.nx), inputs['disp'][:, :3])

        outputs['def_mesh'] += np.einsum('lij,klj->kli',
                                         inputs['transformation_matrix'],
                                         inputs['mesh'] - ref_curve)

    def compute_partials(self, inputs, partials):
        partials['def_mesh', 'ref_curve'] = -np.einsum('i,jlk->ijkl',
            np.ones(self.nx), inputs['transformation_matrix']).flatten()

        partials['def_mesh', 'mesh'] = np.einsum('i,jlk->ijkl',
            np.ones(self.nx), inputs['transformation_matrix']).flatten()
        partials['def_mesh', 'mesh'] += np.tile(np.eye(3), self.nx * self.ny).flatten(order='F')

        partials['def_mesh', 'transformation_matrix'] = np.einsum('ijk,l->ijkl',
            inputs['mesh'] - np.einsum('i,jk->ijk', np.ones(self.nx), inputs['ref_curve']),
            np.ones(3)).flatten()
