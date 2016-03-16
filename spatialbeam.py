from __future__ import division
import numpy

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve



def norm(vec): 
    return numpy.sqrt(numpy.sum(vec**2))


class SpatialBeamTube(Component):
    """ Computes geometric properties for a tube element """

    def __init__(self, n):
        super(SpatialBeamTube, self).__init__()

        self.add_param('r', val=numpy.zeros((n - 1)))
        self.add_param('t', val=numpy.zeros((n - 1)))
        self.add_output('A', val=numpy.zeros((n - 1)))
        self.add_output('Iy', val=numpy.zeros((n - 1)))
        self.add_output('Iz', val=numpy.zeros((n - 1)))
        self.add_output('J', val=numpy.zeros((n - 1)))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        pi = numpy.pi
        r1 = params['r'] - 0.5 * params['t']
        r2 = params['r'] + 0.5 * params['t']

        unknowns['A'] = pi * (r2**2 - r1**2)
        unknowns['Iy'] = pi * (r2**4 - r1**4) / 4.
        unknowns['Iz'] = pi * (r2**4 - r1**4) / 4.
        unknowns['J'] = pi * (r2**4 - r1**4) / 2.

#        print params['t']



class SpatialBeamFEM(Component):
    """ Computes the displacements and rotations """

    def __init__(self, n, cons, E, G, fem_origin=0.35):
        super(SpatialBeamFEM, self).__init__()

        size = 6 * n + 6 * cons.shape[0]

        self.add_param('A', val=numpy.zeros((n - 1)))
        self.add_param('Iy', val=numpy.zeros((n - 1)))
        self.add_param('Iz', val=numpy.zeros((n - 1)))
        self.add_param('J', val=numpy.zeros((n - 1)))
        self.add_param('mesh', val=numpy.zeros((2, n, 3)))

        self.add_param('loads', val=numpy.zeros((n, 6)))

        self.add_state('disp_aug', val=numpy.zeros((size)), dtype="complex")

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"
        self.fd_options['linearize'] = True # only for circulations

        self.E = E
        self.G = G
        self.cons = cons
        self.fem_origin = fem_origin

        elem_IDs = numpy.zeros((n-1, 2), int)
        elem_IDs[:, 0] = numpy.arange(n-1)
        elem_IDs[:, 1] = numpy.arange(n-1) + 1
        self.elem_IDs = elem_IDs

        self.const2 = numpy.array([
            [1, -1],
            [-1, 1],
        ], dtype='complex')
        self.const_y = numpy.array([
            [12, -6, -12, -6],
            [-6, 4, 6, 2],
            [-12, 6, 12, 6],
            [-6, 2, 6, 4],
        ], dtype='complex')
        self.const_z = numpy.array([
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4],
        ], dtype='complex')
        self.x_gl = numpy.array([1, 0, 0], dtype='complex')

        self.K_elem = numpy.zeros((12, 12), dtype='complex')
        self.T_elem = numpy.zeros((12, 12), dtype='complex')
        self.T = numpy.zeros((3, 3), dtype='complex')

        num_nodes = n
        num_cons = self.cons.shape[0]
        size = 6*num_nodes + 6*num_cons
        self.mtx = numpy.zeros((size, size), dtype='complex')
        self.rhs = numpy.zeros(size, dtype='complex')

        self.M_a = numpy.zeros((2, 2), dtype='complex')
        self.M_t = numpy.zeros((2, 2), dtype='complex')
        self.M_y = numpy.zeros((4, 4), dtype='complex')
        self.M_z = numpy.zeros((4, 4), dtype='complex')

        self.S_a = numpy.zeros((2, 12), dtype='complex')
        self.S_a[(0, 1), (0, 6)] = 1.

        self.S_t = numpy.zeros((2, 12), dtype='complex')
        self.S_t[(0, 1), (3, 9)] = 1.

        self.S_y = numpy.zeros((4, 12), dtype='complex')
        self.S_y[(0, 1, 2, 3), (2, 4, 8, 10)] = 1.

        self.S_z = numpy.zeros((4, 12), dtype='complex')
        self.S_z[(0, 1, 2, 3), (1, 5, 7, 11)] = 1.

    def _assemble_system(self, params):
        def unit(vec):
            return vec / norm(vec)

        elem_IDs = self.elem_IDs
        cons = self.cons

        mesh = params['mesh']
        w = self.fem_origin
        nodes = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

        num_elems = elem_IDs.shape[0]
        num_nodes = nodes.shape[0]
        num_cons = cons.shape[0]

        elem_nodes = numpy.zeros((num_elems, 2, 3), dtype='complex')

        for ielem in xrange(num_elems):
            in0, in1 = elem_IDs[ielem, :]
            elem_nodes[ielem, 0, :] = nodes[in0, :]
            elem_nodes[ielem, 1, :] = nodes[in1, :]

        E, G = self.E * numpy.ones(num_nodes - 1), self.G * numpy.ones(num_nodes - 1)
        A, J = params['A'], params['J']
        Iy, Iz = params['Iy'], params['Iz']

        self.mtx[:] = 0.
        for ielem in xrange(num_elems):
            P0 = elem_nodes[ielem, 0, :]
            P1 = elem_nodes[ielem, 1, :]

            x_loc = unit(P1 - P0)
            y_loc = unit(numpy.cross(x_loc, self.x_gl))
            z_loc = unit(numpy.cross(x_loc, y_loc))

            self.T[0, :] = x_loc
            self.T[1, :] = y_loc
            self.T[2, :] = z_loc

            for ind in xrange(4):
                self.T_elem[3*ind:3*ind+3, 3*ind:3*ind+3] = self.T

            L = norm(P1 - P0)
            EA_L = E[ielem] * A[ielem] / L
            GJ_L = G[ielem] * J[ielem] / L
            EIy_L3 = E[ielem] * Iy[ielem] / L**3
            EIz_L3 = E[ielem] * Iz[ielem] / L**3

            self.M_a[:, :] = EA_L * self.const2
            self.M_t[:, :] = GJ_L * self.const2

            self.M_y[:, :] = EIy_L3 * self.const_y
            self.M_y[1, :] *= L
            self.M_y[3, :] *= L
            self.M_y[:, 1] *= L
            self.M_y[:, 3] *= L

            self.M_z[:, :] = EIz_L3 * self.const_z
            self.M_z[1, :] *= L
            self.M_z[3, :] *= L
            self.M_z[:, 1] *= L
            self.M_z[:, 3] *= L

            self.K_elem[:] = 0
            self.K_elem += self.S_a.T.dot(self.M_a).dot(self.S_a)
            self.K_elem += self.S_t.T.dot(self.M_t).dot(self.S_t)
            self.K_elem += self.S_y.T.dot(self.M_y).dot(self.S_y)
            self.K_elem += self.S_z.T.dot(self.M_z).dot(self.S_z)

            res = self.T_elem.T.dot(self.K_elem).dot(self.T_elem)

            in0, in1 = elem_IDs[ielem, :]

            self.mtx[6*in0:6*in0+6, 6*in0:6*in0+6] += res[:6, :6]
            self.mtx[6*in1:6*in1+6, 6*in0:6*in0+6] += res[6:, :6]
            self.mtx[6*in0:6*in0+6, 6*in1:6*in1+6] += res[:6, 6:]
            self.mtx[6*in1:6*in1+6, 6*in1:6*in1+6] += res[6:, 6:]

        for ind in xrange(num_cons):
            for k in xrange(6):
                self.mtx[6*num_nodes + 6*ind + k, 6*cons[ind]+k] = 1.
                self.mtx[6*cons[ind]+k, 6*num_nodes + 6*ind + k] = 1.

        self.rhs[:] = 0.0
        self.rhs[:6*num_nodes] = params['loads'].reshape((6*num_nodes))
        
    def solve_nonlinear(self, params, unknowns, resids):
        self._assemble_system(params)

        unknowns['disp_aug'] = numpy.linalg.solve(self.mtx, self.rhs)
        
    def apply_nonlinear(self, params, unknowns, resids):
        self._assemble_system(params)

        disp_aug = unknowns['disp_aug']
        resids['disp_aug'] = self.mtx.dot(disp_aug) - self.rhs

    def linearize(self, params, unknowns, resids):
        """ Jacobian for disp."""
        self.lup = lu_factor(self.mtx.real)

    def solve_linear(self, dumat, drmat, vois, mode=None):

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat

        for voi in vois:
            if mode == "fwd":
                sol_vec[voi].vec[:] = lu_solve(self.lup, -rhs_vec[voi].vec)
            else:
                sol_vec[voi].vec[:] = lu_solve(self.lup, -rhs_vec[voi].vec, trans=1)



class SpatialBeamDisp(Component):
    """ Selects displacements from augmented vector """

    def __init__(self, n, cons):
        super(SpatialBeamDisp, self).__init__()

        size = 6 * n + 6 * cons.shape[0]
        self.n = n
        
        self.add_param('disp_aug', val=numpy.zeros((size)))
        self.add_output('disp', val=numpy.zeros((n, 6)))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        n = self.n
        unknowns['disp'] = numpy.array(params['disp_aug'][:6*n].reshape((n, 6)))



class SpatialBeamEnergy(Component):
    """ Computes strain energy """

    def __init__(self, n):
        super(SpatialBeamEnergy, self).__init__()

        self.add_param('disp', val=numpy.zeros((n, 6)))
        self.add_param('loads', val=numpy.zeros((n, 6)))
        self.add_output('energy', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['energy'] = 1e-5*numpy.sum(params['disp'] * params['loads'])



class SpatialBeamWeight(Component):
    """ Computes total weight """

    def __init__(self, n):
        super(SpatialBeamWeight, self).__init__()

        self.add_param('t', val=numpy.zeros((n-1)))
        self.add_output('weight', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['weight'] = numpy.sum(params['t'])



class SpatialBeamGroup(Group):

    def __init__(self, num_y, cons, E, G):
        super(SpatialBeamGroup, self).__init__()

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
