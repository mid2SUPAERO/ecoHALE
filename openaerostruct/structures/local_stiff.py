from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

##from openaerostruct.HALE.fctMultiMatos import*


coeffs_2 = np.array([
    [ 1., -1.],
    [-1.,  1.],
])

coeffs_y = np.array([
    [ 12.,  -6., -12.,  -6.],
    [ -6.,   4.,   6.,   2.],
    [-12.,   6.,  12.,   6.],
    [ -6.,   2.,   6.,   4.],
])

coeffs_z = np.array([
    [ 12.,   6., -12.,   6.],
    [  6.,   4.,  -6.,   2.],
    [-12.,  -6.,  12.,  -6.],
    [  6.,   2.,  -6.,   4.],
])


class LocalStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.ny = ny = surface['mesh'].shape[1]

        self.add_input('A', shape=ny - 1, units='m**2')
        self.add_input('J', shape=ny - 1, units='m**4')
        self.add_input('Iy', shape=ny - 1, units='m**4')
        self.add_input('Iz', shape=ny - 1, units='m**4')
        self.add_input('element_lengths', shape=ny - 1, units='m')
        ##self.add_input('mrho', val=1000, units='kg/m**3') #ED
        
        self.add_input('young', val=np.array([1e10,1e10]), units= 'N/m**2')  #VMGM
        self.add_input('shear', val=np.array([1e10,1e10]), units= 'N/m**2')  #VMGM
        self.add_input('Aspars', shape=ny - 1, units='m**2')  #VMGM

        self.add_output('local_stiff', shape=(ny - 1, 12, 12))

        rows = np.arange(144 * (ny - 1))
        cols = np.outer(np.arange(ny - 1), np.ones(144, int)).flatten()

        self.declare_partials('local_stiff', 'A', rows=rows, cols=cols)
        self.declare_partials('local_stiff', 'J', rows=rows, cols=cols)
        self.declare_partials('local_stiff', 'Iy', rows=rows, cols=cols)
        self.declare_partials('local_stiff', 'Iz', rows=rows, cols=cols)
        self.declare_partials('local_stiff', 'element_lengths', rows=rows, cols=cols)
#        self.declare_partials('local_stiff', 'mrho', method='fd', step=0.1, step_calc='abs')
        ##self.declare_partials('local_stiff', 'mrho', method='cs')
        
        self.declare_partials('local_stiff','young', method='cs')   #VMGM
        self.declare_partials('local_stiff','shear', method='cs')   #VMGM
        self.declare_partials('local_stiff','Aspars', rows=rows, cols=cols)   #VMGM

    def compute(self, inputs, outputs):
        ##surface = self.options['surface']
        ##puissanceMM = surface['puissanceMM']

        ##ny = self.ny
        ##E = youngMM(inputs['mrho'],surface['materlist'],puissanceMM)  #ED
        ##G = shearMM(inputs['mrho'],surface['materlist'],puissanceMM)  #ED
        
        Espar = inputs['young'][0]  #VMGM
        Gspar = inputs['shear'][0]  #VMGM 
        Eskin = inputs['young'][1]  #VMGM
        Gskin = inputs['shear'][1]  #VMGM 
        Aspars = inputs['Aspars']  #VMGM

        A  = inputs['A']
        Iy = inputs['Iy']
        Iz = inputs['Iz']
        J  = inputs['J']
        L  = inputs['element_lengths']

        outputs['local_stiff'] = 0.

        for i in range(2):
            for j in range(2):
                outputs['local_stiff'][:, 0 + i, 0 + j] = (Espar * Aspars + Eskin * (A - Aspars)) / L * coeffs_2[i, j]
                outputs['local_stiff'][:, 2 + i, 2 + j] = Gspar * J / L * coeffs_2[i, j]

        for i in range(4):
            for j in range(4):
                outputs['local_stiff'][:, 4 + i, 4 + j] = Espar * Iy / L ** 3 * coeffs_y[i, j]
                outputs['local_stiff'][:, 8 + i, 8 + j] = Eskin * Iz / L ** 3 * coeffs_z[i, j]

        for i in [1, 3]:
            for j in range(4):
                outputs['local_stiff'][:, 4 + i, 4 + j] *= L
                outputs['local_stiff'][:, 8 + i, 8 + j] *= L
        for i in range(4):
            for j in [1, 3]:
                outputs['local_stiff'][:, 4 + i, 4 + j] *= L
                outputs['local_stiff'][:, 8 + i, 8 + j] *= L

    def compute_partials(self, inputs, partials):
        surface = self.options['surface']
        ##puissanceMM = surface['puissanceMM']
        ny = surface['mesh'].shape[1]
        ##E = youngMM(inputs['mrho'],surface['materlist'],puissanceMM)  #ED
        ##G = shearMM(inputs['mrho'],surface['materlist'],puissanceMM)  #ED
        
        Espar = inputs['young'][0]  #VMGM
        Gspar = inputs['shear'][0]  #VMGM 
        Eskin = inputs['young'][1]  #VMGM
        Gskin = inputs['shear'][1]  #VMGM 
        Aspars = inputs['Aspars']  #VMGM
        Aspars = inputs['Aspars']  #VMGM

        A  = inputs['A']
        Iy = inputs['Iy']
        Iz = inputs['Iz']
        J  = inputs['J']
        L  = inputs['element_lengths']

        derivs_A = partials['local_stiff', 'A'].reshape((ny - 1, 12, 12))
        derivs_Iy = partials['local_stiff', 'Iy'].reshape((ny - 1, 12, 12))
        derivs_Iz = partials['local_stiff', 'Iz'].reshape((ny - 1, 12, 12))
        derivs_J = partials['local_stiff', 'J'].reshape((ny - 1, 12, 12))
        derivs_L = partials['local_stiff', 'element_lengths'].reshape((ny - 1, 12, 12))
#        derivs_mrho = partials['local_stiff', 'mrho'].reshape((ny - 1, 12, 12))
        
        ##derivs_Espar = partials['local_stiff', 'young'][0].reshape((ny - 1, 12, 12))   #VMGM
        ##derivs_Eskin = partials['local_stiff', 'young'][1].reshape((ny - 1, 12, 12))   #VMGM
        ##derivs_Gspar = partials['local_stiff', 'shear'][0].reshape((ny - 1, 12, 12))   #VMGM
        ##derivs_Gskin = partials['local_stiff', 'shear'][1].reshape((ny - 1, 12, 12))   #VMGM
        derivs_Aspars = partials['local_stiff', 'Aspars'].reshape((ny - 1, 12, 12))  #VMGM
        
        derivs_A[:] = 0.
        derivs_Iy[:] = 0.
        derivs_Iz[:] = 0.
        derivs_J[:] = 0.
        derivs_L[:] = 0.
#        derivs_mrho[:] = 0.
        
        ##derivs_Espar[:] = 0.  #VMGM
        ##derivs_Eskin[:] = 0.  #VMGM
        ##derivs_Gspar[:] = 0.  #VMGM
        ##derivs_Gskin[:] = 0.  #VMGM
        derivs_Aspars[:] = 0.  #VMGM

        for i in range(2):
            for j in range(2):
                derivs_A[:, 0 + i, 0 + j] = Eskin / L * coeffs_2[i, j]
                derivs_L[:, 0 + i, 0 + j] = -(Espar * Aspars + Eskin * (A - Aspars)) / L ** 2 * coeffs_2[i, j]

                derivs_J[:, 2 + i, 2 + j] = Gspar / L * coeffs_2[i, j]
                derivs_L[:, 2 + i, 2 + j] = -Gspar * J / L ** 2 * coeffs_2[i, j]
                
                ##derivs_Espar[:, 0 + i, 0 + j] = Aspars / L * coeffs_2[i, j]  #VMGM
                ##derivs_Eskin[:, 0 + i, 0 + j] = (A - Aspars) / L * coeffs_2[i, j]  #VMGM
                ##derivs_Gspar[:, 2 + i, 2 + j] = J / L * coeffs_2[i, j]  #VMGM
                derivs_Aspars[:, 0 + i, 0 + j] = (Espar - Eskin) / L * coeffs_2[i, j]  #VMGM

        for i in range(4):
            for j in range(4):
                derivs_Iy[:, 4 + i, 4 + j] = Espar / L ** 3 * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -3 * Espar * Iy / L ** 4 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = Eskin / L ** 3 * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -3 * Eskin * Iz / L ** 4 * coeffs_z[i, j]

                ##derivs_Espar[:, 4 + i, 4 + j] = Iy / L ** 3 * coeffs_y[i, j]  #VMGM
                ##derivs_Eskin[:, 8 + i, 8 + j] = Iz / L ** 3 * coeffs_z[i, j]  #VMGM
                
        for i in [1, 3]:
            for j in range(4):
                derivs_Iy[:, 4 + i, 4 + j] = Espar / L ** 2 * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -2 * Espar * Iy / L ** 3 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = Eskin / L ** 2 * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -2 * Eskin * Iz / L ** 3 * coeffs_z[i, j]
                
                ##derivs_Espar[:, 4 + i, 4 + j] = Iy / L ** 2 * coeffs_y[i, j]  #VMGM
                ##derivs_Eskin[:, 8 + i, 8 + j] = Iz / L ** 2 * coeffs_z[i, j]  #VMGM
                
        for i in range(4):
            for j in [1, 3]:
                derivs_Iy[:, 4 + i, 4 + j] = Espar / L ** 2 * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -2 * Espar * Iy / L ** 3 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = Eskin / L ** 2 * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -2 * Eskin * Iz / L ** 3 * coeffs_z[i, j]
                
                ##derivs_Espar[:, 4 + i, 4 + j] = Iy / L ** 2 * coeffs_y[i, j]  #VMGM
                ##derivs_Eskin[:, 8 + i, 8 + j] = Iz / L ** 2 * coeffs_z[i, j]  #VMGM

        for i in [1, 3]:
            for j in [1, 3]:
                derivs_Iy[:, 4 + i, 4 + j] = Espar / L * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -Espar * Iy / L ** 2 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = Eskin / L * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -Eskin * Iz / L ** 2 * coeffs_z[i, j]
                
                ##derivs_Espar[:, 4 + i, 4 + j] = Iy / L * coeffs_y[i, j]  #VMGM
                ##derivs_Eskin[:, 8 + i, 8 + j] = Iz / L * coeffs_z[i, j]  #VMGM