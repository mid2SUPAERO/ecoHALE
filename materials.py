from __future__ import division
import numpy

from openmdao.api import Component



class MaterialsTube(Component):
    """ Computes geometric properties for a tube element """

    def __init__(self, aero_ind, fem_ind, fem_origin=0.35):
        super(MaterialsTube, self).__init__()

        n_fem, i_fem = fem_ind[0, :]
        tot_n = numpy.sum(aero_ind[:, 2])
        num_surf = fem_ind.shape[0]
        self.fem_ind = fem_ind
        self.aero_ind = aero_ind

        num_surf = fem_ind.shape[0]
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        size = 6 * tot_n_fem + 6 * num_surf
        self.tot_n_fem = tot_n_fem

        self.fem_origin = fem_origin

        self.add_param('r', val=numpy.zeros((tot_n_fem - num_surf)))
        self.add_param('thickness', val=numpy.zeros((tot_n_fem - num_surf)))
        self.add_param('mesh', val=numpy.zeros((tot_n, 3)))
        self.add_output('A', val=numpy.zeros((tot_n_fem - num_surf)))
        self.add_output('Iy', val=numpy.zeros((tot_n_fem - num_surf)))
        self.add_output('Iz', val=numpy.zeros((tot_n_fem - num_surf)))
        self.add_output('J', val=numpy.zeros((tot_n_fem - num_surf)))
        self.add_output('nodes', val=numpy.zeros((tot_n_fem, 3)))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

        self.arange = numpy.arange(tot_n_fem - num_surf)

    def solve_nonlinear(self, params, unknowns, resids):
        pi = numpy.pi
        r1 = params['r'] - 0.5 * params['thickness']
        r2 = params['r'] + 0.5 * params['thickness']

        unknowns['A'] = pi * (r2**2 - r1**2)
        unknowns['Iy'] = pi * (r2**4 - r1**4) / 4.
        unknowns['Iz'] = pi * (r2**4 - r1**4) / 4.
        unknowns['J'] = pi * (r2**4 - r1**4) / 2.

        w = self.fem_origin
        for i_surf, row in enumerate(self.fem_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = self.aero_ind[i_surf, :]
            n_fem, i_fem = row
            mesh = params['mesh'][i:i+n, :].reshape(nx, ny, 3)
            unknowns['nodes'][i_fem:i_fem+n_fem] = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

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
