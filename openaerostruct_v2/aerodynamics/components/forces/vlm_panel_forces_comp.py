from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2

from openaerostruct_v2.utils.misc_utils import tile_sparse_jac


class VLMPanelForcesComp(ExplicitComponent):
    """
    Total forces by panel (flattened), aligned with the freestream (the lift and drag axes).
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

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

        self.add_input('rho_kg_m3', shape=num_nodes)
        self.add_input('horseshoe_circulations', shape=(num_nodes, system_size))
        self.add_input(velocities_name, shape=(num_nodes, system_size, 3))
        self.add_input('bound_vecs', shape=(num_nodes, system_size, 3))
        self.add_output('panel_forces', shape=(num_nodes, system_size, 3))

        rows = np.arange(3 * system_size)
        cols = np.zeros(3 * system_size, int)
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size * 3, 1., num_nodes)
        self.declare_partials('panel_forces', 'rho_kg_m3', rows=rows, cols=cols)

        rows = np.arange(3 * system_size)
        cols = np.outer(np.arange(system_size), np.ones(3, int)).flatten()
        _, rows, cols = tile_sparse_jac(1., rows, cols,
            system_size * 3, system_size, num_nodes)
        self.declare_partials('panel_forces', 'horseshoe_circulations', rows=rows, cols=cols)

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
        self.declare_partials('panel_forces', velocities_name, rows=rows, cols=cols)

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
        self.declare_partials('panel_forces', 'bound_vecs', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        velocities_name = '{}_velocities'.format('force_pts')

        system_size = self.system_size

        rho_kg_m3 = np.einsum('i,jk->ijk', inputs['rho_kg_m3'], np.ones((system_size, 3)))
        horseshoe_circulations = np.einsum('ij,k->ijk', inputs['horseshoe_circulations'], np.ones(3))
        velocities = inputs[velocities_name]
        bound_vecs = inputs['bound_vecs']

        outputs['panel_forces'] = \
            rho_kg_m3 * horseshoe_circulations * compute_cross(velocities, bound_vecs)

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']

        system_size = self.system_size

        velocities_name = '{}_velocities'.format('force_pts')

        rho_kg_m3 = np.einsum('i,jk->ijk', inputs['rho_kg_m3'], np.ones((system_size, 3)))
        horseshoe_circulations = np.einsum('ij,k->ijk', inputs['horseshoe_circulations'], np.ones(3))
        velocities = inputs[velocities_name]
        bound_vecs = inputs['bound_vecs']

        horseshoe_circulations_ones = np.einsum('ij,kl->ijkl', inputs['horseshoe_circulations'], np.ones((3, 3)))
        rho_kg_m3_ones = np.einsum('i,jkl->ijkl', inputs['rho_kg_m3'], np.ones((system_size, 3, 3)))

        deriv_array = np.einsum('ij,kl->ijkl',
            np.ones((num_nodes, system_size)),
            np.eye(3))

        partials['panel_forces', 'rho_kg_m3'] = \
            (horseshoe_circulations * compute_cross(velocities, bound_vecs)).flatten()
        partials['panel_forces', 'horseshoe_circulations'] = \
            (rho_kg_m3 * compute_cross(velocities, bound_vecs)).flatten()
        partials['panel_forces', velocities_name] = \
            (rho_kg_m3_ones * horseshoe_circulations_ones * compute_cross_deriv1(deriv_array, bound_vecs)).flatten()
        partials['panel_forces', 'bound_vecs'] = \
            (rho_kg_m3_ones * horseshoe_circulations_ones * compute_cross_deriv2(velocities, deriv_array)).flatten()
