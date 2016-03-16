from __future__ import division
import numpy

from openmdao.api import Component



class Mesh(Component):
    """ Creates an aero mesh based on high-level design parameters. """

    def __init__(self, n, span, chord):
        super(Mesh, self).__init__()

        self.n = n
        self.span = span
        self.chord = chord

        self.add_param('twist', val=numpy.zeros((n)))
        self.add_output('mesh', val=numpy.zeros((2, n, 3)))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        n = self.n
        span = self.span
        chord = self.chord
        twist = params['twist']
        mesh = numpy.zeros((2, n, 3), dtype='complex')
        for ind_x in xrange(2):
            for ind_y in xrange(n):
                mesh[ind_x, ind_y, :] = [ind_x * chord, ind_y / (n-1) * span, 0]

        mesh[1, :, 2] = -twist
        unknowns['mesh'] = mesh
