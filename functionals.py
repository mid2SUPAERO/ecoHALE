from __future__ import division
import numpy

from openmdao.api import Component



class FunctionalBreguetRange(Component):
    """ Computes the fuel burn using the Breguet range equation """

    def __init__(self, W0, CT, a, R, M):
        super(FunctionalBreguetRange, self).__init__()

        self.add_param('CL', val=1.)
        self.add_param('CD', val=1.)
        self.add_param('weight', val=0.)

        self.add_output('fuelburn', val=0.)

        self.W0 = W0
        self.CT = CT
        self.a = a
        self.R = R
        self.M = M
        
        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        W0 = self.W0
        CT = self.CT
        a = self.a
        R = self.R
        M = self.M

        CL = params['CL']
        CD = params['CD']
        Ws = params['weight']

        unknowns['fuelburn'] = (W0 + Ws) * (numpy.exp(R * CT / a / M * CD / CL) - 1)



class FunctionalEquilibrium(Component):
    """ L = W constraint """

    def __init__(self, W0):
        super(FunctionalEquilibrium, self).__init__()

        self.add_param('L', val=1.)
        self.add_param('weight', val=1.)

        self.add_output('eq_con', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self.W0 = W0

    def solve_nonlinear(self, params, unknowns, resids):
        W0 = self.W0

        unknowns['eq_con'] = params['weight'] + W0 - params['L']
