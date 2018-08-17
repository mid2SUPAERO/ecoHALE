from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class PanelForces(ExplicitComponent):
    """
    Total forces by panel (flattened), aligned with the freestream (the lift and drag axes).
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        system_size = 0

        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']

            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        velocities_name = '{}_velocities'.format('force_pts')

        self.add_input('rho', units='kg/m**3')
        self.add_input('horseshoe_circulations', shape=system_size, units='m**2/s')
        self.add_input(velocities_name, shape=(system_size, 3), units='m/s')
        self.add_input('bound_vecs', shape=(system_size, 3), units='m')
        self.add_output('panel_forces', shape=(system_size, 3), units='N')

        self.declare_partials('panel_forces', 'rho',
            rows=np.arange(3 * system_size),
            cols=np.zeros(3 * system_size, int),
        )
        self.declare_partials('panel_forces', 'horseshoe_circulations',
            rows=np.arange(3 * system_size),
            cols=np.outer(np.arange(system_size), np.ones(3, int)).flatten(),
        )
        self.declare_partials('panel_forces', velocities_name,
            rows=np.einsum('ij,k->ijk',
                np.arange(3 * system_size).reshape((system_size, 3)),
                np.ones(3, int),
            ).flatten(),
            cols=np.einsum('ik,j->ijk',
                np.arange(3 * system_size).reshape((system_size, 3)),
                np.ones(3, int),
            ).flatten(),
        )
        self.declare_partials('panel_forces', 'bound_vecs',
            rows=np.einsum('ij,k->ijk',
                np.arange(3 * system_size).reshape((system_size, 3)),
                np.ones(3, int),
            ).flatten(),
            cols=np.einsum('ik,j->ijk',
                np.arange(3 * system_size).reshape((system_size, 3)),
                np.ones(3, int),
            ).flatten(),
        )

    def compute(self, inputs, outputs):
        velocities_name = '{}_velocities'.format('force_pts')

        system_size = self.system_size

        rho = inputs['rho'][0]
        horseshoe_circulations = np.outer(inputs['horseshoe_circulations'], np.ones(3))
        velocities = inputs[velocities_name]
        bound_vecs = inputs['bound_vecs']

        outputs['panel_forces'] = \
            rho * horseshoe_circulations * compute_cross(velocities, bound_vecs)

    def compute_partials(self, inputs, partials):
        velocities_name = '{}_velocities'.format('force_pts')

        system_size = self.system_size

        rho = inputs['rho'][0]
        horseshoe_circulations = np.outer(inputs['horseshoe_circulations'], np.ones(3))
        velocities = inputs[velocities_name]
        bound_vecs = inputs['bound_vecs']

        horseshoe_circulations_ones = np.einsum('i,jk->ijk', inputs['horseshoe_circulations'], np.ones((3, 3)))

        deriv_array = np.einsum('i,jk->ijk',
            np.ones(self.system_size),
            np.eye(3))

        partials['panel_forces', 'rho'] = \
            (horseshoe_circulations * compute_cross(velocities, bound_vecs)).flatten()
        partials['panel_forces', 'horseshoe_circulations'] = \
            (rho * compute_cross(velocities, bound_vecs)).flatten()
        partials['panel_forces', velocities_name] = \
            (rho * horseshoe_circulations_ones * compute_cross_deriv1(deriv_array, bound_vecs)).flatten()
        partials['panel_forces', 'bound_vecs'] = \
            (rho * horseshoe_circulations_ones * compute_cross_deriv2(velocities, deriv_array)).flatten()
