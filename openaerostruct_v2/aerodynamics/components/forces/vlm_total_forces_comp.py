from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class VLMTotalForcesComp(ExplicitComponent):
    """
    Total lift and drag.
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

        self.add_input('panel_forces', shape=(system_size, 3))
        self.add_output('lift')
        self.add_output('drag')

        self.declare_partials('lift', 'panel_forces', val=1.,
            rows=np.zeros(system_size, int),
            cols=np.arange(3 * system_size).reshape((system_size, 3))[:, 1],
        )
        self.declare_partials('drag', 'panel_forces', val=1.,
            rows=np.zeros(system_size, int),
            cols=np.arange(3 * system_size).reshape((system_size, 3))[:, 0],
        )

    def compute(self, inputs, outputs):
        outputs['lift'] = np.sum(inputs['panel_forces'][:, 1])
        outputs['drag'] = np.sum(inputs['panel_forces'][:, 0])
