""" Manipulate geometry mesh based on high-level design parameters. """

from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent
from openaerostruct.structures.utils import radii

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

class RadiusComp(ExplicitComponent):
    """

    Parameters
    ----------

    Returns
    -------
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.nx, self.ny = surface['num_x'], surface['num_y']
        self.add_input('mesh', val=np.zeros((self.nx, self.ny, 3)), units='m')
        self.add_input('t_over_c', val=np.ones((self.ny-1)))
        self.add_output('radius', val=np.ones((self.ny - 1)), units='m')

        arange  = np.arange(self.ny-1)
        self.declare_partials('radius','t_over_c',rows=arange,cols=arange)
        self.declare_partials('radius','mesh')

    def compute(self, inputs, outputs):
        outputs['radius'] = radii(inputs['mesh'], inputs['t_over_c'])

    def compute_partials(self, inputs, partials):
        """
        Obtain the radii of the FEM element based on local chord.
        """
        mesh = inputs['mesh']
        vectors = mesh[-1, :, :] - mesh[0, :, :]
        chords = np.sqrt(np.sum(vectors**2, axis=1))
        t_c = inputs['t_over_c']
        mean_chords = 0.5 * chords[:-1] + 0.5 * chords[1:]

        dr_dtoc = mean_chords/2
        partials['radius','t_over_c'] = dr_dtoc

        dmean_dchords = np.zeros((self.ny-1,self.ny))
        i,j = np.indices(dmean_dchords.shape)
        dmean_dchords[i==j] = 0.5
        dmean_dchords[i==j-1] = 0.5
        dr_dmean = np.diag(t_c/2)
        dr_dchords = np.squeeze(np.matmul(dr_dmean, dmean_dchords))

        dchords_dmesh = np.zeros((self.ny,self.nx*self.ny*3))

        le_ind = 0
        te_ind = (self.nx - 1) * 3 * self.ny

        dx = mesh[0, :, 0] - mesh[-1, :, 0]
        dy = mesh[0, :, 1] - mesh[-1, :, 1]
        dz = mesh[0, :, 2] - mesh[-1, :, 2]

        l = np.sqrt(dx**2 + dy**2 + dz**2)
        i = np.arange(self.ny)

        dchords_dmesh[i, le_ind + i*3 + 0] += dx[i] / l[i]
        dchords_dmesh[i, te_ind + i*3 + 0] -= dx[i] / l[i]
        dchords_dmesh[i, le_ind + i*3 + 1] += dy[i] / l[i]
        dchords_dmesh[i, te_ind + i*3 + 1] -= dy[i] / l[i]
        dchords_dmesh[i, le_ind + i*3 + 2] += dz[i] / l[i]
        dchords_dmesh[i, te_ind + i*3 + 2] -= dz[i] / l[i]
        
        partials['radius','mesh'] = np.matmul(dr_dchords,dchords_dmesh)