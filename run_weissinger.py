from __future__ import division
import numpy

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer
from geometry import GeometryMesh
from weissinger import WeissingerPreproc, WeissingerCirculations, WeissingerForces, WeissingerLift, WeissingerLiftCoeff, WeissingerDragCoeff
from transfer import TransferDisplacements

import sys


class WeissingerGroup(Group):

    def __init__(self, mesh_params, aero_params):
        super(WeissingerGroup, self).__init__()

        num_y = mesh_params['num_y']
        span = mesh_params['span']
        chord = mesh_params['chord']
    
        v = aero_params['v']
        alpha = aero_params['alpha']
        rho = aero_params['rho']
        disp = numpy.zeros((num_y, 6))

        self.add('twist',
                 IndepVarComp('twist', numpy.zeros((num_y))),
                 promotes=['*'])
        self.add('v',
                 IndepVarComp('v', v),
                 promotes=['*'])
        self.add('alpha',
                 IndepVarComp('alpha', alpha),
                 promotes=['*'])
        self.add('rho',
                 IndepVarComp('rho', rho),
                 promotes=['*'])
        self.add('disp',
                 IndepVarComp('disp', disp),
                 promotes=['*'])
        
        self.add('mesh',
                 GeometryMesh(num_y, span, chord),
                 promotes=['*'])
        self.add('def_mesh',
                 TransferDisplacements(num_y),
                 promotes=['*'])
        self.add('preproc',
                 WeissingerPreproc(num_y),
                 promotes=['*'])
        self.add('circ',
                 WeissingerCirculations(num_y),
                 promotes=['*'])
        self.add('forces',
                 WeissingerForces(num_y),
                 promotes=['*'])
        self.add('lift',
                 WeissingerLift(num_y),
                 promotes=['*'])
        self.add('CL',
                 WeissingerLiftCoeff(num_y),
                 promotes=['*'])
        self.add('CD',
                 WeissingerDragCoeff(num_y),
                 promotes=['*'])


if __name__ == '__main__':

    mesh_params = {
        'num_y': 3,
        'span': 232.02,
        'chord': 39.37,
    }

    aero_params = {
        'v': 200.,
        'alpha': 3.,
        'rho': 1.225,
    }

    top = Problem()
    top.root = WeissingerGroup(mesh_params, aero_params)

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'
    top.driver.options['disp'] = True
    # top.driver.options['tol'] = 1.0e-12

    num_y = mesh_params['num_y']

    top.driver.add_desvar('twist',lower=numpy.ones((num_y)) * -10.,
                          upper=numpy.ones((num_y)) * 10.)
    top.driver.add_desvar('alpha', lower=-10., upper=10., scaler=100)

    top.driver.add_objective('CD')

    top.driver.add_constraint('CL', equals=0.5)

    top.setup()

    if sys.argv[1] == '0':
        top.run_once()

        #data = top.check_total_derivatives()
        data = top.check_partial_derivatives(comps=['circ', 'forces'])

        # top.run()

    elif sys.argv[1] == '1':
       
        top.run()

    print top['CL'], top['CD']

