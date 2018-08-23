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

        # Setup Inputs
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('structural_weight', val=0., units='N')
        self.add_input('element_weights', val=np.zeros((self.ny-1)), units='N')
        
        # Setup Outputs
        self.add_output('cg_location', val=np.zeros((3)), units='m')
        
        # Setup Partials
        self.declare_partials(of='cg_location', wrt='structural_weight')
        self.declare_partials(of='cg_location', wrt='element_weights')
        # Setup Sparce Matrix
        dimensions = 3
        cols_const = np.arange(0,self.ny*dimensions,dimensions)
        rows = np.empty(self.ny*dimensions)
        cols = np.empty(self.ny*dimensions)
        for i in range(dimensions):
            rows[i*self.ny:i*self.ny+self.ny]=i
            cols[i*self.ny:i*self.ny+self.ny]=cols_const+i
            
        self.declare_partials(of='cg_location', wrt='nodes', rows=rows, cols=cols)
        
        # Check partials options
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
        ny = self.ny
        nodes = inputs['nodes']
        structural_weight = inputs['structural_weight']
        element_weights = ew = inputs['element_weights']

        center_of_elements = ((nodes[1:, :] + nodes[:-1, :]) / 2.).T

        sum_coe_dot_ew = center_of_elements.dot(element_weights)

        is_sym = self.surface['symmetry']

        J['cg_location', 'structural_weight'] = -sum_coe_dot_ew/structural_weight**2
        J['cg_location', 'element_weights'] = center_of_elements/structural_weight

        #derivates with respect to nodes require special ninja math
        values = np.zeros(ny)
        values[0:ny-1] = ew
        values[1:ny] += ew
        values /= 2*structural_weight
        dimensions = 3
        
        J['cg_location', 'nodes'] = np.tile(values,dimensions)
        if is_sym:
            J['cg_location', 'nodes'][ny:2*ny] = 0
            J['cg_location', 'nodes'] *= 2

            J['cg_location', 'structural_weight'][1] = 0
            J['cg_location', 'structural_weight'] *= 2

            J['cg_location', 'element_weights'][1] = 0
            J['cg_location', 'element_weights'] *= 2