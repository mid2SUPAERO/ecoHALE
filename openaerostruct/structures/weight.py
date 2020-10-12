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
    structural_mass : float
        Weight of the wing structure.
    elmenet_weight[ny-1] : float
        Weight of each element.

    """
    

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']
        
        self.ny = ny = surface['mesh'].shape[1]

        self.add_input('A', val=np.ones((self.ny - 1)), units='m**2')
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('mrho', val=np.array([1000, 1000]), units='kg/m**3') #ED
        self.add_input('Aspars', val=np.ones((self.ny - 1)), units='m**2')  #VMGM
     
        self.add_output('structural_mass', val=0., units='kg')
        self.add_output('element_mass', val=np.zeros((self.ny-1)), units='kg')
        self.add_output('spars_mass', val=0., units='kg')  #VMGM

        self.declare_partials('structural_mass', ['A','nodes','mrho','Aspars'])
#        self.declare_partials('structural_mass', ['A','nodes'])
#        self.declare_partials('structural_mass', 'mrho', method='fd', step=10, step_calc='abs')
#        self.declare_partials('element_mass', 'mrho', method='fd', step=10, step_calc='abs')
        self.declare_partials('element_mass', 'mrho')

        row_col = np.arange(self.ny-1, dtype=int)
        self.declare_partials('element_mass','A', rows=row_col, cols=row_col)
        self.declare_partials('element_mass', 'Aspars', rows=row_col, cols=row_col)  #VMGM

        dimensions = 3
        rows=np.empty((dimensions*2*(ny-1)))
        cols=np.empty((dimensions*2*(ny-1)))
        for i in range (ny-1):
            rows[i*dimensions*2:(i+1)*dimensions*2] = i
            cols[i*dimensions*2:(i+1)*dimensions*2] = np.linspace(i*dimensions,i*dimensions+(dimensions*2-1),dimensions*2)
        self.declare_partials('element_mass','nodes', rows=rows, cols=cols)
        
        self.declare_partials('spars_mass', ['A','nodes','mrho','Aspars'])  #VMGM

        self.set_check_partial_options('*', method='cs', step=1e-40)

    def compute(self, inputs, outputs):
        A = inputs['A']
        nodes = inputs['nodes']
        mrho = inputs['mrho']
        wwr = self.surface['wing_weight_ratio']
        Aspars = inputs['Aspars']  #VMGM

        # Calculate the volume and weight of the structure
        element_spar_volumes = norm(nodes[1:, :] - nodes[:-1, :], axis=1) * Aspars
        element_skin_volumes = norm(nodes[1:, :] - nodes[:-1, :], axis=1) * (A - Aspars)

        # nodes[1:, :] - nodes[:-1, :] this is the delta array of the difference between the points
        element_mass = element_spar_volumes * mrho[0] * wwr + element_skin_volumes * mrho[1] * wwr
        weight = np.sum(element_mass)
        weight_spars = np.sum(element_spar_volumes * mrho[0] * wwr)  #VMGM

        # If the tube is symmetric, double the computed weight
        if self.surface['symmetry']:
            weight *= 2.
            weight_spars *= 2.  #VMGM

        if weight < 0 :
            print("negative weight")
            
        outputs['structural_mass'] = weight
        outputs['element_mass'] = element_mass
        outputs['spars_mass'] = weight_spars  #VMGM

    def compute_partials(self, inputs, partials):

        A = inputs['A']
        nodes = inputs['nodes']
        mrho = inputs['mrho']
        wwr = self.surface['wing_weight_ratio']
        ny = self.ny
        Aspars = inputs['Aspars']  #VMGM

        # Calculate the volume and weight of the structure
        const0 = nodes[1:, :] - nodes[:-1, :]
        const1 = np.linalg.norm(const0, axis=1) #length of elements
        element_spar_volumes = const1 * Aspars
        element_skin_volumes = const1 * (A - Aspars)
        volume_spar = np.sum(element_spar_volumes) 
        volume_skin = np.sum(element_skin_volumes)
        const2 = mrho[0] * wwr
        const3 = mrho[1] * wwr
        weight = volume_spar * const2 + volume_skin * const3

        # First we will solve for dweight_dA
        # Calculate the volume and weight of the total structure
        norms = const1.reshape(1, -1)  #length of elements

        # Multiply by the material density and force of gravity
        dweight_dA = norms * const3
        dweight_dmrho = np.array([volume_spar * wwr, volume_skin * wwr])
        dweight_dAspars = norms * const2 - norms * const3  #VMGM

        # Account for symmetry
        if self.surface['symmetry']:
            dweight_dA *= 2.
            dweight_dmrho *= 2.
            dweight_dAspars *= 2.  #VMGM

        # Save the result to the jacobian dictionary
        partials['structural_mass', 'A'] = dweight_dA
        partials['structural_mass', 'mrho'] = dweight_dmrho
        partials['structural_mass', 'Aspars'] = dweight_dAspars  #VMGM

        # Next, we will compute the derivative of weight wrt nodes.
        # Here we're using results from AD to compute the derivative
        # Initialize the reverse seeds.
        nodesb = np.zeros(nodes.shape)
        tempb = (const0) * (Aspars * mrho[0] / norms).reshape(-1, 1) + (const0) * ((A - Aspars) * mrho[1] / norms).reshape(-1, 1)
        nodesb[1:, :] += tempb
        nodesb[:-1, :] -= tempb

        # Apply the multipliers for material properties and symmetry
        nodesb *= wwr

        if self.surface['symmetry']:
            nodesb *= 2.

        # Store the flattened array in the jacobian dictionary
        partials['structural_mass', 'nodes'] = nodesb.reshape(1, -1)

        # Element_weight Partials
        partials['element_mass','A'] = const1 * const3
        partials['element_mass', 'mrho'] = np.transpose(np.array([element_spar_volumes * wwr, element_skin_volumes * wwr])) 
        partials['element_mass', 'Aspars'] = const1 * const2 - const1 * const3
        
        precalc = np.sum(np.power(const0,2),axis=1)
        d__dprecalc = 0.5 * precalc**(-.5)

        dimensions = 3
        for i in range(ny-1):
            first_part = const0[i,:] * d__dprecalc[i] * 2 * (-1) * Aspars[i] * const2 + const0[i,:] * d__dprecalc[i] * 2 * (-1) * (A[i] - Aspars[i]) * const3
            second_part = const0[i,:] * d__dprecalc[i] * 2 * Aspars[i] * const2 + const0[i,:] * d__dprecalc[i] * 2 * (A[i] - Aspars[i]) * const3
            partials['element_mass', 'nodes'][i*dimensions*2:(i+1)*dimensions*2] = np.append(first_part,second_part)
        
        dsparsweight_dAspars = norms * const2  #VMGM
        dsparsweight_dmrho = np.array([volume_spar * wwr, 0.])  #VMGM
        
        nodesb2 = np.zeros(nodes.shape)
        tempb2 = (const0) * (Aspars * mrho[0] / norms).reshape(-1, 1)
        nodesb2[1:, :] += tempb2
        nodesb2[:-1, :] -= tempb2

        nodesb2 *= wwr
        
        if self.surface['symmetry']:
            dsparsweight_dAspars *= 2.
            dsparsweight_dmrho *= 2.
            nodesb2 *= 2.
            
        partials['spars_mass','A'] = 0.
        partials['spars_mass','mrho'] = dsparsweight_dmrho
        partials['spars_mass','Aspars'] = dsparsweight_dAspars
        partials['spars_mass','nodes'] = nodesb2.reshape(1, -1)
