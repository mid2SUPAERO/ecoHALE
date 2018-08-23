from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


class StructuralCG(ExplicitComponent):
    """ Compute center-of-gravity location of the spar elements.

    parameters
    ----------
    A[ny-1] : numpy array
        Areas for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    structural_weight : float
        Weight of the structural spar.
    cg_location[3] : numpy array
        Location of the structural spar's cg.

    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']

        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('structural_weight', val=0., units='N')
        self.add_input('element_weights', val=np.zeros((self.ny-1)), units='N')
        self.add_output('cg_location', val=np.zeros((3)), units='m')

        self.declare_partials('*', '*')
        self.set_check_partial_options('*', method='cs', step=1e-40)

    def compute(self, inputs, outputs):
        nodes = inputs['nodes']
        structural_weight = inputs['structural_weight']
        element_weights = inputs['element_weights']

        # Calculate the center-of-gravity location of the spar elements only
        center_of_elements = ((nodes[1:, :] + nodes[:-1, :]) / 2.).T
        cg_loc = center_of_elements.dot(element_weights) / structural_weight

        # If the tube is symmetric, double the computed weight and set the
        # y-location of the cg to 0, at the symmetry plane
        if self.surface['symmetry']:
            cg_loc[1] = 0.
            cg_loc *= 2.

        outputs['cg_location'] = cg_loc

    def compute_partials(self, inputs, J):
        nodes = inputs['nodes']
        structural_weight = inputs['structural_weight']
        element_weights = ew = inputs['element_weights']

        center_of_elements = ((nodes[1:, :] + nodes[:-1, :]) / 2.).T

        sum_coe_dot_ew = center_of_elements.dot(element_weights)

        is_sym = self.surface['symmetry']

        J['cg_location', 'structural_weight'] = -sum_coe_dot_ew/structural_weight**2
        J['cg_location', 'element_weights'] = center_of_elements/structural_weight

        #derivates with respect to nodes require special ninja math
        n_nodes = self.ny*3
        row = np.zeros(n_nodes)
        for i in range(self.ny-1):
            row[i*3] += ew[i]
            row[i*3 + 3] += ew[i]
        row /= structural_weight*2
        for i in range(3):
            J['cg_location', 'nodes'][i] = np.roll(row,i)

        if is_sym:
            J['cg_location', 'nodes'][1] *= 0
            J['cg_location', 'nodes'] *= 2

            J['cg_location', 'structural_weight'][1] = 0
            J['cg_location', 'structural_weight'] *= 2

            J['cg_location', 'element_weights'][1] = 0
            J['cg_location', 'element_weights'] *= 2
