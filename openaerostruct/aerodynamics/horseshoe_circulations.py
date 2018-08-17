from __future__ import print_function
import numpy as np
from scipy.sparse import csc_matrix

from openmdao.api import ExplicitComponent


class HorseshoeCirculations(ExplicitComponent):

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

        self.add_input('circulations', shape=system_size, units='m**2/s')
        self.add_output('horseshoe_circulations', shape=system_size, units='m**2/s')

        data = [np.ones(system_size)]
        rows = [np.arange(system_size)]
        cols = [np.arange(system_size)]

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            arange = np.arange(num).reshape((nx - 1), (ny - 1))

            data_ = -np.ones((nx - 2) * (ny - 1))
            rows_ = ind_1 + arange[1:, :].flatten()
            cols_ = ind_1 + arange[:-1, :].flatten()

            data.append(data_)
            rows.append(rows_)
            cols.append(cols_)

            ind_1 += num

        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)

        self.mtx = csc_matrix((data, (rows, cols)), shape=(system_size, system_size))

        self.declare_partials('horseshoe_circulations', 'circulations', val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['horseshoe_circulations'] = self.mtx.dot(inputs['circulations'])
