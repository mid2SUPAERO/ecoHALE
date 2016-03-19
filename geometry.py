from __future__ import division
import numpy

from openmdao.api import Component


class GeometryMesh(Component):
    """ Creates an aero mesh based on high-level design parameters. """

    def __init__(self, n, chord):
        super(GeometryMesh, self).__init__()

        self.n = n
        self.chord = chord

        self.add_param('span', val=0.)
        self.add_param('twist', val=numpy.zeros(n))
        self.add_output('mesh', val=numpy.zeros((2, n, 3)))

        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        ones = numpy.ones(n, dtype="int")
        zeros = numpy.zeros(self.n, dtype=int)
        idx = [ones,range(n),2*ones]
        self.twist_idx_flat = (numpy.ravel_multi_index(idx, (2,n,3)), range(n))

        idx = [zeros, range(n), ones]
        self.span_idx_flat_le = (numpy.ravel_multi_index(idx, (2,n,3)), zeros)

        idx = [ones, range(n), ones]
        self.span_idx_flat_te = (numpy.ravel_multi_index(idx, (2,n,3)), zeros)

        idx = [zeros, range(n), zeros]
        self.chord_idx_flat_le = (numpy.ravel_multi_index(idx, (2,n,3)), range(n))

        idx = [ones, range(n), zeros]
        self.chord_idx_flat_te = (numpy.ravel_multi_index(idx, (2,n,3)), range(n))

        J = {}
        J['mesh','twist'] = numpy.zeros((2*n*3, n))
        J['mesh','twist'][self.twist_idx_flat] = -1.
        J['mesh', 'span'] = numpy.zeros((6*n,1))
        J['mesh', 'chord'] = numpy.zeros((6*n,n))
        self.J = J

    def solve_nonlinear(self, params, unknowns, resids):
        n = self.n
        span = params['span']
        chord = self.chord
        twist = params['twist']
        mesh = numpy.zeros((2, n, 3), dtype='complex')
        for ind_x in xrange(2): # 0 for LE, 1 for TE
            if ind_x == 0: 
                chord_sign = 1
            else: 
                chord_sign = -1 
            for ind_y in xrange(n): # span-wise direction
                mesh[ind_x, ind_y, :] = [ind_x*chord, ind_y / (n-1) * span, 0]

        mesh[1, :, 2] = -twist
        unknowns['mesh'] = mesh

    def linearize(self, params, unknowns, resids): 

        dmesh_dspan = unknowns['mesh'][0,:,1].flatten() * 1/params['span']
        self.J['mesh', 'span'][self.span_idx_flat_le] = dmesh_dspan
        dmesh_dspan = unknowns['mesh'][1,:,1].flatten() * 1/params['span']
        self.J['mesh', 'span'][self.span_idx_flat_te] = dmesh_dspan

       

        return self.J
