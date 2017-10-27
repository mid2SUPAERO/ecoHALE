from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class VLMRotatePanelForcesComp(ExplicitComponent):
    """
    Rotate the computed panel forces.
    """

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
        self.add_input('panel_forces', shape=(system_size, 3))
        self.add_output('panel_forces_rotated', shape=(system_size, 3))

        self.declare_partials('panel_forces_rotated', 'alpha_rad',
            rows=np.arange(3 * system_size),
            cols=np.zeros(3 * system_size, int),
        )

        self.declare_partials('panel_forces_rotated', 'panel_forces',
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
        system_size = self.system_size

        alpha_rad = inputs['alpha_rad'][0]
        panel_forces = inputs['panel_forces']

        rotation = np.einsum('i,jk->ijk',
            np.ones(system_size),
            np.array([
                [ np.cos(alpha_rad) , np.sin(alpha_rad) , 0. ],
                [-np.sin(alpha_rad) , np.cos(alpha_rad) , 0. ],
                [0., 0., 1.],
            ])
        )

        outputs['panel_forces_rotated'] = np.einsum('ijk,ik->ij',
            rotation,
            panel_forces,
        )

    def compute_partials(self, inputs, partials):
        system_size = self.system_size

        alpha_rad = inputs['alpha_rad'][0]
        panel_forces = inputs['panel_forces']

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

        deriv_panel_forces = np.einsum('i,jk->ijk',
            np.ones(system_size),
            np.eye(3))

        partials['panel_forces_rotated', 'alpha_rad'] = np.einsum('ijk,ik->ij',
            deriv_rotation,
            panel_forces,
        ).flatten()

        partials['panel_forces_rotated', 'panel_forces'] = np.einsum('ijk,ikl->ijl',
            rotation,
            deriv_panel_forces,
        ).flatten()
