from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class PanelForcesSurf(ExplicitComponent):
    """
    Total forces by panel (flattened).
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

        arange = np.arange(3 * system_size)

        self.add_input('panel_forces', shape=(system_size, 3), units='N')

        ind1, ind2 = 0, 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            out_name = '{}_sec_forces'.format(name)

            ind2 += (nx - 1) * (ny - 1) * 3

            self.add_output(out_name, shape=(nx - 1, ny - 1, 3), units='N')

            rows = np.arange((nx - 1) * (ny - 1) * 3)
            cols = arange[ind1:ind2]
            self.declare_partials(out_name, 'panel_forces', val=1., rows=rows, cols=cols)

            ind1 += (nx - 1) * (ny - 1) * 3

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']

        ind1, ind2 = 0, 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            out_name = '{}_sec_forces'.format(name)

            ind2 += (nx - 1) * (ny - 1)
            outputs[out_name] = inputs['panel_forces'][ind1:ind2].reshape(
                (nx - 1, ny - 1, 3))

            ind1 += (nx - 1) * (ny - 1)
