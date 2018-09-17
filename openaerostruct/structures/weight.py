from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent
from openaerostruct.structures.utils import norm


class Weight(ExplicitComponent):
    """ Compute total weight and center-of-gravity location of the spar elements.

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
    elmenet_weight[ny-1] : float
        weight of each elemnet

    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = ny = surface['mesh'].shape[1]

        self.add_input('A', val=np.ones((self.ny - 1)), units='m**2')
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('load_factor', val=1.)

        self.add_output('structural_weight', val=0., units='N')
        self.add_output('element_weights', val=np.zeros((self.ny-1)), units='N')

        self.declare_partials('structural_weight', ['A','nodes','load_factor'])


        self.declare_partials('element_weights', 'load_factor')
        row_col = np.arange(self.ny-1, dtype=int)
        self.declare_partials('element_weights','A', rows=row_col, cols=row_col)
        ny = self.ny
        dimensions = 3
        rows=np.empty((dimensions*2*(ny-1)))
        cols=np.empty((dimensions*2*(ny-1)))
        for i in range (ny-1):
            rows[i*dimensions*2:(i+1)*dimensions*2] = i
            cols[i*dimensions*2:(i+1)*dimensions*2] = np.linspace(i*dimensions,i*dimensions+(dimensions*2-1),dimensions*2)
        self.declare_partials('element_weights','nodes', rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs', step=1e-40)

    def compute(self, inputs, outputs):
        A = inputs['A']
        nodes = inputs['nodes']
        mrho = self.surface['mrho']
        wwr = self.surface['wing_weight_ratio']
        lf = inputs['load_factor']

        # Calculate the volume and weight of the structure
        element_volumes = norm(nodes[1:, :] - nodes[:-1, :], axis=1) * A

        # nodes[1:, :] - nodes[:-1, :] this is the delta array of the different between the points
        element_weights = element_volumes * mrho * 9.81 * wwr * lf
        weight = np.sum(element_weights)

        # If the tube is symmetric, double the computed weight and set the
        # y-location of the cg to 0, at the symmetry plane
        if self.surface['symmetry']:
            weight *= 2.

        #outputs['structural_weight'] = weight
        outputs['structural_weight'] = weight
        outputs['element_weights'] = element_weights

    def compute_partials(self, inputs, partials):

        A = inputs['A']
        nodes = inputs['nodes']
        mrho = self.surface['mrho']
        wwr = self.surface['wing_weight_ratio']
        ny = self.ny
        lf = inputs['load_factor']

        # Calculate the volume and weight of the structure
        const0 = nodes[1:, :] - nodes[:-1, :]
        const1 = np.linalg.norm(const0, axis=1)
        element_volumes = const1 * A
        volume = np.sum(element_volumes)
        const2 = mrho * 9.81 * wwr * lf
        weight = volume * const2

        # First we will solve for dweight_dA
        # Calculate the volume and weight of the total structure
        norms = const1.reshape(1, -1)

        # Multiply by the material density and force of gravity
        dweight_dA = norms * const2

        # Account for symmetry
        if self.surface['symmetry']:
            dweight_dA *= 2.

        # Save the result to the jacobian dictionary
        partials['structural_weight', 'A'] = dweight_dA

        # Next, we will compute the derivative of weight wrt nodes.
        # Here we're using results from AD to compute the derivative
        # Initialize the reverse seeds.
        nodesb = np.zeros(nodes.shape)
        tempb = (const0) * (A / norms).reshape(-1, 1)
        nodesb[1:, :] += tempb
        nodesb[:-1, :] -= tempb

        # Apply the multipliers for material properties and symmetry
        nodesb *= mrho * 9.81 * wwr * lf

        if self.surface['symmetry']:
            nodesb *= 2.
            partials['structural_weight', 'load_factor'] = weight * 2
        else:
            partials['structural_weight', 'load_factor'] = weight

        # Store the flattened array in the jacobian dictionary
        partials['structural_weight', 'nodes'] = nodesb.reshape(1, -1)

        # Element_weight Partials
        partials['element_weights','A'] = const1 * const2
        partials['element_weights','load_factor'] = const1 * A * mrho * 9.81 * wwr

        precalc = np.sum(np.power(const0,2),axis=1)
        d__dprecalc = 0.5 * precalc**(-.5)

        dimensions = 3
        for i in range(ny-1):
            first_part = const0[i,:] * d__dprecalc[i] * 2 * (-1) * A[i] * const2
            second_part = const0[i,:] * d__dprecalc[i] * 2 * A[i] * const2
            partials['element_weights', 'nodes'][i*dimensions*2:(i+1)*dimensions*2] = np.append(first_part,second_part)

        #dew__dprecalc =



