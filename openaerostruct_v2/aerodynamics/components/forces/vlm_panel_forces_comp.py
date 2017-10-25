from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class VLMPanelForcesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']

        system_size = 0

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            system_size += (num_points_x - 1) * (num_points_z - 1)

        self.system_size = system_size

        velocities_name = '{}_velocities'.format('force_pts')

        self.add_input('alpha_rad')
        self.add_input('rho_kg_m3')
        self.add_input('horseshoe_circulations', shape=system_size)
        self.add_input(velocities_name, shape=(system_size, 3))
        self.add_input('bound_vecs', shape=(system_size, 3))
        self.add_output('panel_forces', shape=(system_size, 3))

        self.declare_partials('panel_forces', 'alpha_rad',
            rows=np.arange(3 * system_size),
            cols=np.zeros(3 * system_size, int),
        )
        self.declare_partials('panel_forces', 'rho_kg_m3',
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

        alpha_rad = inputs['alpha_rad'][0]
        rho_kg_m3 = inputs['rho_kg_m3'][0]
        horseshoe_circulations = np.outer(inputs['horseshoe_circulations'], np.ones(3))
        velocities = inputs[velocities_name]
        bound_vecs = inputs['bound_vecs']

        rotation = np.einsum('i,jk->ijk',
            np.ones(system_size),
            np.array([
                [ np.cos(alpha_rad) , np.sin(alpha_rad) , 0. ],
                [-np.sin(alpha_rad) , np.cos(alpha_rad) , 0. ],
                [0., 0., 1.],
            ])
        )

        outputs['panel_forces'] = np.einsum('ijk,ik->ij',
            rotation,
            rho_kg_m3 * horseshoe_circulations * compute_cross(velocities, bound_vecs),
        )

    def compute_partials(self, inputs, partials):
        velocities_name = '{}_velocities'.format('force_pts')

        system_size = self.system_size

        alpha_rad = inputs['alpha_rad'][0]
        rho_kg_m3 = inputs['rho_kg_m3'][0]
        horseshoe_circulations = np.outer(inputs['horseshoe_circulations'], np.ones(3))
        velocities = inputs[velocities_name]
        bound_vecs = inputs['bound_vecs']

        rotation = np.einsum('i,jk->ijk',
            np.ones(system_size),
            np.array([
                [ np.cos(alpha_rad) ,  np.sin(alpha_rad) , 0. ],
                [-np.sin(alpha_rad) ,  np.cos(alpha_rad) , 0. ],
                [0., 0., 1.],
            ])
        )

        deriv_rotation = np.einsum('i,jk->ijk',
            np.ones(system_size),
            np.array([
                [-np.sin(alpha_rad) ,  np.cos(alpha_rad) , 0. ],
                [-np.cos(alpha_rad) , -np.sin(alpha_rad) , 0. ],
                [0., 0., 1.],
            ])
        )

        horseshoe_circulations_ones = np.einsum('i,jk->ijk', inputs['horseshoe_circulations'], np.ones((3, 3)))

        deriv_array = np.einsum('i,jk->ijk',
            np.ones(self.system_size),
            np.eye(3))

        partials['panel_forces', 'alpha_rad'] = np.einsum('ijk,ik->ij',
            deriv_rotation,
            rho_kg_m3 * horseshoe_circulations * compute_cross(velocities, bound_vecs),
        ).flatten()
        partials['panel_forces', 'rho_kg_m3'] = np.einsum('ijk,ik->ij',
            rotation,
            horseshoe_circulations * compute_cross(velocities, bound_vecs),
        ).flatten()
        partials['panel_forces', 'horseshoe_circulations'] = np.einsum('ijk,ik->ij',
            rotation,
            rho_kg_m3 * compute_cross(velocities, bound_vecs),
        ).flatten()
        partials['panel_forces', velocities_name] = np.einsum('ijk,ikl->ijl',
            rotation,
            rho_kg_m3 * horseshoe_circulations_ones * compute_cross_deriv1(deriv_array, bound_vecs),
        ).flatten()
        partials['panel_forces', 'bound_vecs'] = np.einsum('ijk,ikl->ijl',
            rotation,
            rho_kg_m3 * horseshoe_circulations_ones * compute_cross_deriv2(velocities, deriv_array),
        ).flatten()
