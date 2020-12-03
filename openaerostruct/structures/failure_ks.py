from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

##from openaerostruct.HALE.fctMultiMatos import*


class FailureKS(ExplicitComponent):
    """
    Aggregate failure constraints from the structure.

    To simplify the optimization problem, we aggregate the individual
    elemental failure constraints using a Kreisselmeier-Steinhauser (KS)
    function.

    The KS function produces a smoother constraint than using a max() function
    to find the maximum point of failure, which produces a better-posed
    optimization problem.

    The rho inputeter controls how conservatively the KS function aggregates
    the failure constraints. A lower value is more conservative while a greater
    value is more aggressive (closer approximation to the max() function).

    parameters
    ----------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Used to simplify the optimization problem by
        reducing the number of constraints.

    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('rho', types=float, default=100.)

    def setup(self):
        surface = self.options['surface']
        rho = self.options['rho']

        if surface['fem_model_type'] == 'tube':
            num_failure_criteria = 2
        elif surface['fem_model_type'] == 'wingbox':
            num_failure_criteria = 4

        self.ny = surface['mesh'].shape[1]

        self.add_input('vonmises', val=np.zeros((self.ny-1, num_failure_criteria)), units='N/m**2')
        ##self.add_input('mrho', val=1000, units='kg/m**3') #ED
        
        self.add_input('yield', val=np.array([1e8,1e8]), units= 'N/m**2')  #VMGM
        
        self.add_output('failure', val=0.)

#        self.sigma = yieldMM(surface['mrho'],surface['materlist'])  #ED
        self.surface = surface #ED
        self.rho = rho
        """
#        self.declare_partials('*','*')
        self.declare_partials('failure', 'vonmises')
#        self.declare_partials('failure', 'mrho', method='fd', step=0.1, step_calc='abs')
        ##self.declare_partials('failure', 'mrho', method='cs')
        
        self.declare_partials('failure', 'yield')   #VMGM"""
        
        self.declare_partials('*', '*', method='cs')
        
    def compute(self, inputs, outputs):
#        sigma = self.sigma  #ED
        ##mrho = inputs['mrho'] #ED
#        print('failure') #ED
#        print(mrho)  #ED
        ##sigma = yieldMM(mrho,self.surface['materlist'],self.surface['puissanceMM'])  #ED
        
        sigma = inputs['yield'] #VMGM
        
        rho = self.rho
        vonmises = inputs['vonmises']

        vm_sigma = np.zeros(vonmises.shape)  #VMGM
        vm_sigma[0] = vonmises[0] / sigma[1]  #VMGM
        vm_sigma[1] = vonmises[1] / sigma[1]  #VMGM
        vm_sigma[2] = vonmises[2] / sigma[0]  #VMGM
        vm_sigma[3] = vonmises[3] / sigma[0]  #VMGM

        fmax = np.max(vm_sigma - 1)

        nlog, nsum, nexp = np.log, np.sum, np.exp
        ks = 1 / rho * nlog(nsum(nexp(rho * (vm_sigma - 1 - fmax))))
        outputs['failure'] = fmax + ks

    """
    def compute_partials(self, inputs, partials):
        vonmises = inputs['vonmises']
        ##mrho = inputs['mrho']  #ED
#        sigma = self.sigma
        ##sigma = yieldMM(mrho,self.surface['materlist'],self.surface['puissanceMM'])  #ED
        rho = self.rho
        
        sigma = np.array([inputs['yield'][1],inputs['yield'][1],inputs['yield'][0],inputs['yield'][0]]) #VMGM
        
        vm_sigma = vonmises / sigma  #VMGM

        # Find the location of the max stress constraint
        fmax = np.max(vm_sigma - 1)
        i, j = np.where((vm_sigma - 1)==fmax)
        i, j = i[0], j[0]

        # Set incoming seed as 1 so we simply get the jacobian entries
        ksb = 1.

        # Use results from the AD code to compute the jacobian entries
        tempb0 = ksb / (rho * np.sum(np.exp(rho * (vm_sigma - fmax - 1))))
        tempb = np.exp(rho*(vm_sigma-fmax-1))*rho*tempb0
        fmaxb = ksb - np.sum(tempb)
        
        # Populate the entries
        derivs = tempb / sigma
        derivs[i, j] += fmaxb / sigma[j]

        # Reshape and save them to the jac dict
        partials['failure', 'vonmises'] = derivs.reshape(1, -1)
        derivYield = -partials['failure', 'vonmises'] * vm_sigma.reshape(1, -1) #VMGM
        partials['failure', 'yield'] = np.sum(derivYield[0]) #VMGM """