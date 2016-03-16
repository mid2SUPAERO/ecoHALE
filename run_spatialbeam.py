from __future__ import division
import numpy

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer
from geometry import GeometryMesh
from spatialbeam import SpatialBeamTube, SpatialBeamFEM, SpatialBeamDisp, SpatialBeamEnergy, SpatialBeamWeight

import sys


class SpatialBeamGroup(Group):

    def __init__(self, mesh_params, mat_params, elem_params):
        super(SpatialBeamGroup, self).__init__()

        num_y = mesh_params['num_y']
        span = mesh_params['span']
        chord = mesh_params['chord']

        E = mat_params['E']
        G = mat_params['G']

        r = elem_params['r']
        t = elem_params['t']
        loads = numpy.zeros((num_y, 6))
        loads[0, 2] = loads[-1, 2] = 1e3
        loads[:, 2] = 1e3

        cons = numpy.array([int((num_y-1)/2)])

        self.add('twist',
                 IndepVarComp('twist', numpy.zeros((num_y))),
                 promotes=['*'])
        self.add('r', IndepVarComp('r', r), promotes=['*'])
        self.add('t', IndepVarComp('t', t), promotes=['*'])
        self.add('loads', IndepVarComp('loads', loads), promotes=['*'])

        self.add('mesh',
                 GeometryMesh(num_y, span, chord),
                 promotes=['*'])
        self.add('tube',
                 SpatialBeamTube(num_y),
                 promotes=['*'])
        self.add('fem',
                 SpatialBeamFEM(num_y, cons, E, G),
                 promotes=['*'])
        self.add('disp',
                 SpatialBeamDisp(num_y, cons),
                 promotes=['*'])
        self.add('energy',
                 SpatialBeamEnergy(num_y),
                 promotes=['*'])
        self.add('weight',
                 SpatialBeamWeight(num_y),
                 promotes=['*'])


if __name__ == '__main__':

    num_y = 3

    mesh_params = {
        'num_y': num_y,
        'span': 232.02,
        'chord': 39.37,
    }

    mat_params = {
        'E': 200.e9,
        'G': 30.e9,
    }

    elem_params = {
        'r': 0.3 * numpy.ones(num_y-1),
        't': 0.02 * numpy.ones(num_y-1)
    }
    
    if sys.argv[1] == '0':
        top = Problem()
        top.root = SpatialBeamGroup(mesh_params, mat_params, elem_params)

        top.setup()
        top.run()

        #data = top.check_total_derivatives()
        data = top.check_partial_derivatives()

        top.run()
        print
        print top['A']
        print top['Iy']
        print top['Iz']
        print top['J']
        print
        print top['disp']

    elif sys.argv[1] == '1':
        top = Problem()
        top.root = SpatialBeamGroup(mesh_params, mat_params, elem_params)

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['disp'] = True
        # top.driver.options['tol'] = 1.0e-12

        num_y = mesh_params['num_y']

        top.driver.add_desvar('t',
                              lower=numpy.ones((num_y)) * 0.001,
                              upper=numpy.ones((num_y)) * 0.25)

        top.driver.add_objective('energy')
        top.driver.add_constraint('weight', upper = 0.5)
#        top.driver.add_constraint('t', lower=numpy.ones((num_y)) * 0.005)

        top.setup()
        top.run()
