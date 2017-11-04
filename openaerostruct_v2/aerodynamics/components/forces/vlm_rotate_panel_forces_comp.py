from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from openaerostruct_v2.utils.misc_utils import tile_sparse_jac


class VLMRotatePanelForcesComp(ExplicitComponent):
    """
    Rotate the computed panel forces.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        velocities_name = '{}_velocities'.format('force_pts')

        self.add_input('alpha_rad', shape=num_nodes)
        self.add_input('panel_forces', shape=(num_nodes, system_size, 3))
        self.add_output('panel_forces_rotated', shape=(num_nodes, system_size, 3))

        rows = np.arange(3 * system_size)
        cols = np.zeros(3 * system_size, int)
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size * 3, 1., num_nodes)
        self.declare_partials('panel_forces_rotated', 'alpha_rad', rows=rows, cols=cols)

        rows = np.einsum('ij,k->ijk',
            np.arange(3 * system_size).reshape((system_size, 3)),
            np.ones(3, int),
        ).flatten()
        cols = np.einsum('ik,j->ijk',
            np.arange(3 * system_size).reshape((system_size, 3)),
            np.ones(3, int),
        ).flatten()
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size * 3, system_size * 3, num_nodes)
        self.declare_partials('panel_forces_rotated', 'panel_forces', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']

        system_size = self.system_size

        alpha_rad = inputs['alpha_rad']
        panel_forces = inputs['panel_forces']

        ones = np.ones(system_size)

        rotation = np.zeros((num_nodes, system_size, 3, 3))
        rotation[:, :, 0, 0] = np.outer( np.cos(alpha_rad), ones)
        rotation[:, :, 0, 1] = np.outer( np.sin(alpha_rad), ones)
        rotation[:, :, 1, 0] = np.outer(-np.sin(alpha_rad), ones)
        rotation[:, :, 1, 1] = np.outer( np.cos(alpha_rad), ones)
        rotation[:, :, 2, 2] = 1.

        outputs['panel_forces_rotated'] = np.einsum('ijkl,ijl->ijk',
            rotation,
            panel_forces,
        )

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']

        system_size = self.system_size

        alpha_rad = inputs['alpha_rad']
        panel_forces = inputs['panel_forces']

        ones = np.ones(system_size)

        rotation = np.zeros((num_nodes, system_size, 3, 3))
        rotation[:, :, 0, 0] = np.outer( np.cos(alpha_rad), ones)
        rotation[:, :, 0, 1] = np.outer( np.sin(alpha_rad), ones)
        rotation[:, :, 1, 0] = np.outer(-np.sin(alpha_rad), ones)
        rotation[:, :, 1, 1] = np.outer( np.cos(alpha_rad), ones)
        rotation[:, :, 2, 2] = 1.

        deriv_rotation = np.zeros((num_nodes, system_size, 3, 3))
        deriv_rotation[:, :, 0, 0] = np.outer(-np.sin(alpha_rad), ones)
        deriv_rotation[:, :, 0, 1] = np.outer( np.cos(alpha_rad), ones)
        deriv_rotation[:, :, 1, 0] = np.outer(-np.cos(alpha_rad), ones)
        deriv_rotation[:, :, 1, 1] = np.outer(-np.sin(alpha_rad), ones)

        deriv_panel_forces = np.einsum('ij,kl->ijkl',
            np.ones((num_nodes, system_size)),
            np.eye(3))

        partials['panel_forces_rotated', 'alpha_rad'] = np.einsum('ijkl,ijl->ijk',
            deriv_rotation,
            panel_forces,
        ).flatten()

        partials['panel_forces_rotated', 'panel_forces'] = np.einsum('ijkl,ijlm->ijkm',
            rotation,
            deriv_panel_forces,
        ).flatten()
