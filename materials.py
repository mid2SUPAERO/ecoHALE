from __future__ import division
import numpy

from openmdao.api import Component



class MaterialsTube(Component):
    """ Computes geometric properties for a tube element """

    def __init__(self, aero_ind):
        super(MaterialsTube, self).__init__()

        n = aero_ind[0, 1]

        self.add_param('r', val=numpy.zeros((n - 1)))
        self.add_param('thickness', val=numpy.zeros((n - 1)))
        self.add_output('A', val=numpy.zeros((n - 1)))
        self.add_output('Iy', val=numpy.zeros((n - 1)))
        self.add_output('Iz', val=numpy.zeros((n - 1)))
        self.add_output('J', val=numpy.zeros((n - 1)))

#        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

        self.arange = numpy.arange(n-1)

    def solve_nonlinear(self, params, unknowns, resids):
        pi = numpy.pi
        r1 = params['r'] - 0.5 * params['thickness']
        r2 = params['r'] + 0.5 * params['thickness']

        unknowns['A'] = pi * (r2**2 - r1**2)
        unknowns['Iy'] = pi * (r2**4 - r1**4) / 4.
        unknowns['Iz'] = pi * (r2**4 - r1**4) / 4.
        unknowns['J'] = pi * (r2**4 - r1**4) / 2.

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()

        pi = numpy.pi
        r = params['r'].real
        t = params['thickness'].real
        r1 = r - 0.5 * t
        r2 = r + 0.5 * t

        dr1_dr = 1.
        dr2_dr = 1.
        dr1_dt = -0.5
        dr2_dt =  0.5

        r1_3 = r1**3
        r2_3 = r2**3

        a = self.arange
        jac['A', 'r'][a, a] = 2 * pi * (r2 * dr2_dr - r1 * dr1_dr)
        jac['A', 'thickness'][a, a] = 2 * pi * (r2 * dr2_dt - r1 * dr1_dt)
        jac['Iy', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        jac['Iy', 'thickness'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        jac['Iz', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        jac['Iz', 'thickness'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        jac['J', 'r'][a, a] = 2 * pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        jac['J', 'thickness'][a, a] = 2 * pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)

        return jac
