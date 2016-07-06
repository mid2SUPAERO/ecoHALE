""" Defines the structural analysis component using spatial beam theory """

from __future__ import division
import numpy

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve
try:
    import lib
    fortran_flag = True
except:
    fortran_flag = False
fortran_flag = False

def norm(vec):
    return numpy.sqrt(numpy.sum(vec**2))

def unit(vec):
    return vec / norm(vec)

def radii(mesh, t_c=0.15):
    vectors = mesh[-1, :, :] - mesh[0, :, :]
    chords = numpy.sqrt(numpy.sum(vectors**2, axis=1))
    chords = 0.5 * chords[:-1] + 0.5 * chords[1:]
    return t_c * chords

def _assemble_system(aero_ind, fem_ind, flat_mesh, A, J, Iy, Iz, loads,
                     M_a, M_t, M_y, M_z,
                     elem_IDs, cons, fem_origin,
                     E, G, x_gl, T,
                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                     const2, const_y, const_z, n, size, mtx, rhs):

    for i_surf, row in enumerate(fem_ind):
        nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = aero_ind[i_surf, :]
        n_fem, i_fem = row

        num_cons = cons.shape[0]

        small_mat = numpy.zeros((size, size), dtype="complex")
        full_mesh = flat_mesh[i:i+n, :].reshape(nx, ny, 3)
        mesh = numpy.zeros((2, ny, 3), dtype="complex")
        mesh[0] = full_mesh[0]
        mesh[1] = full_mesh[-1]


        if 1:
            small_mat, rhs = lib.assemblestructmtx(mesh, A, J, Iy, Iz, loads,
                                M_a, M_t, M_y, M_z,
                                elem_IDs, cons, fem_origin,
                                E, G, x_gl, T,
                                K_elem, S_a, S_t, S_y, S_z, T_elem,
                                const2, const_y, const_z, n_fem, size)
            fort_mat = small_mat.copy()
            small_mat[:] = 0.

        if 1:
            w = fem_origin
            nodes = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

            num_elems = elem_IDs.shape[0]
            num_nodes = nodes.shape[0]
            num_cons = cons.shape[0]

            elem_nodes = numpy.zeros((num_elems, 2, 3), dtype='complex')

            for ielem in xrange(num_elems):
                in0, in1 = elem_IDs[ielem, :]
                elem_nodes[ielem, 0, :] = nodes[in0, :]
                elem_nodes[ielem, 1, :] = nodes[in1, :]

            E, G = E * numpy.ones(num_nodes - 1), G * numpy.ones(num_nodes - 1)

            mtx[:] = 0.
            for ielem in xrange(num_elems):
                P0 = elem_nodes[ielem, 0, :]
                P1 = elem_nodes[ielem, 1, :]

                x_loc = unit(P1 - P0)
                y_loc = unit(numpy.cross(x_loc, x_gl))
                z_loc = unit(numpy.cross(x_loc, y_loc))

                T[0, :] = x_loc
                T[1, :] = y_loc
                T[2, :] = z_loc

                for ind in xrange(4):
                    T_elem[3*ind:3*ind+3, 3*ind:3*ind+3] = T

                L = norm(P1 - P0)
                EA_L = E[ielem] * A[ielem] / L
                GJ_L = G[ielem] * J[ielem] / L
                EIy_L3 = E[ielem] * Iy[ielem] / L**3
                EIz_L3 = E[ielem] * Iz[ielem] / L**3

                M_a[:, :] = EA_L * const2
                M_t[:, :] = GJ_L * const2

                M_y[:, :] = EIy_L3 * const_y
                M_y[1, :] *= L
                M_y[3, :] *= L
                M_y[:, 1] *= L
                M_y[:, 3] *= L

                M_z[:, :] = EIz_L3 * const_z
                M_z[1, :] *= L
                M_z[3, :] *= L
                M_z[:, 1] *= L
                M_z[:, 3] *= L

                K_elem[:] = 0
                K_elem += S_a.T.dot(M_a).dot(S_a)
                K_elem += S_t.T.dot(M_t).dot(S_t)
                K_elem += S_y.T.dot(M_y).dot(S_y)
                K_elem += S_z.T.dot(M_z).dot(S_z)

                res = T_elem.T.dot(K_elem).dot(T_elem)

                in0, in1 = elem_IDs[ielem, :]

                small_mat[6*in0:6*in0+6, 6*in0:6*in0+6] += res[:6, :6]
                small_mat[6*in1:6*in1+6, 6*in0:6*in0+6] += res[6:, :6]
                small_mat[6*in0:6*in0+6, 6*in1:6*in1+6] += res[:6, 6:]
                small_mat[6*in1:6*in1+6, 6*in1:6*in1+6] += res[6:, 6:]

            for ind in xrange(num_cons):
                for k in xrange(6):
                    small_mat[6*num_nodes + 6*ind + k, 6*cons[ind]+k] = 1.
                    small_mat[6*cons[ind]+k, 6*num_nodes + 6*ind + k] = 1.

            rhs[:] = 0.0
            rhs[:6*num_nodes] = loads.reshape((6*num_nodes))

        from weissinger import view_mat
        py_mat = small_mat.copy()
        diff = fort_mat - py_mat

        if not numpy.iscomplexobj(small_mat):
            '====== not complex ======='

        mtx[i_fem*6:(i_fem+n_fem+num_cons)*6, i_fem*6:(i_fem+n_fem+num_cons)*6] = small_mat


class SpatialBeamFEM(Component):
    """ Computes the displacements and rotations """

    def __init__(self, aero_ind, fem_ind, cons, E, G, fem_origin=0.35):
        super(SpatialBeamFEM, self).__init__()

        n_fem, i_fem = fem_ind[0, :]
        tot_n = numpy.sum(fem_ind[:, 0])
        self.fem_ind = fem_ind
        self.aero_ind = aero_ind

        self.size = size = 6 * n_fem + 6 * cons.shape[0]
        self.n = n_fem

        self.add_param('A', val=numpy.zeros((n_fem - 1)))
        self.add_param('Iy', val=numpy.zeros((n_fem - 1)))
        self.add_param('Iz', val=numpy.zeros((n_fem - 1)))
        self.add_param('J', val=numpy.zeros((n_fem - 1)))
        self.add_param('mesh', val=numpy.zeros((2*tot_n, 3), dtype="complex"))

        self.add_param('loads', val=numpy.zeros((n_fem, 6)))

        self.add_state('disp_aug', val=numpy.zeros((size), dtype="complex"))

        # self.deriv_options['type'] = 'cs'
        # self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"
        self.deriv_options['linearize'] = True # only for circulations

        self.arange = numpy.arange(6*n_fem)

        self.E = E
        self.G = G
        self.cons = cons
        self.fem_origin = fem_origin

        elem_IDs = numpy.zeros((n_fem-1, 2), int)
        elem_IDs[:, 0] = numpy.arange(n_fem-1)
        elem_IDs[:, 1] = numpy.arange(n_fem-1) + 1
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

        num_nodes = n_fem
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


    def solve_nonlinear(self, params, unknowns, resids):
        self.mesh = params['mesh']

        _assemble_system(self.aero_ind, self.fem_ind, self.mesh, params['A'], params['J'],
                            params['Iy'], params['Iz'], params['loads'],
                            self.M_a, self.M_t, self.M_y, self.M_z,
                            self.elem_IDs, self.cons, self.fem_origin,
                            self.E, self.G, self.x_gl, self.T,
                            self.K_elem, self.S_a, self.S_t, self.S_y, self.S_z, self.T_elem,
                            self.const2, self.const_y, self.const_z, self.n, self.size, self.mtx, self.rhs)

        unknowns['disp_aug'] = numpy.linalg.solve(self.mtx, self.rhs)

    def apply_nonlinear(self, params, unknowns, resids):
        _assemble_system(self.aero_ind, self.fem_ind, self.mesh, params['A'], params['J'],
                            params['Iy'], params['Iz'], params['loads'],
                            self.M_a, self.M_t, self.M_y, self.M_z,
                            self.elem_IDs, self.cons, self.fem_origin,
                            self.E, self.G, self.x_gl, self.T,
                            self.K_elem, self.S_a, self.S_t, self.S_y, self.S_z, self.T_elem,
                            self.const2, self.const_y, self.const_z, self.n, self.size, self.mtx, self.rhs)

        disp_aug = unknowns['disp_aug']
        resids['disp_aug'] = self.mtx.dot(disp_aug) - self.rhs

    def linearize(self, params, unknowns, resids):
        """ Jacobian for disp."""

        jac = self.alloc_jacobian()
        fd_jac = self.complex_step_jacobian(params, unknowns, resids, \
                                            fd_params=['A','Iy','Iz','J','mesh', 'loads'], \
                                            fd_states=[])
        jac.update(fd_jac)
        jac['disp_aug', 'disp_aug'] = self.mtx.real

        self.lup = lu_factor(self.mtx.real)

        return jac

    def solve_linear(self, dumat, drmat, vois, mode=None):

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t = 0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t = 1

        for voi in vois:
            sol_vec[voi].vec[:] = lu_solve(self.lup, rhs_vec[voi].vec, trans=t)



class SpatialBeamDisp(Component):
    """ Selects displacements from augmented vector """

    def __init__(self, aero_ind, cons):
        super(SpatialBeamDisp, self).__init__()

        n = aero_ind[0, 1]

        size = 6 * n + 6 * cons.shape[0]
        self.n = n

        self.add_param('disp_aug', val=numpy.zeros((size)))
        self.add_output('disp', val=numpy.zeros((n, 6)))
        self.arange = numpy.arange(6*n)

    def solve_nonlinear(self, params, unknowns, resids):
        n = self.n
        unknowns['disp'] = numpy.array(params['disp_aug'][:6*n].reshape((n, 6)))

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        arange = self.arange
        jac['disp', 'disp_aug'][arange, arange] = 1.
        return jac



class SpatialBeamEnergy(Component):
    """ Computes strain energy """

    def __init__(self, aero_ind):
        super(SpatialBeamEnergy, self).__init__()

        n = aero_ind[0, 1]

        self.add_param('disp', val=numpy.zeros((n, 6)))
        self.add_param('loads', val=numpy.zeros((n, 6)))
        self.add_output('energy', val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['energy'] = numpy.sum(params['disp'] * params['loads'])

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['energy', 'disp'][0, :] = params['loads'].real.flatten()
        jac['energy', 'loads'][0, :] = params['disp'].real.flatten()
        return jac



class SpatialBeamWeight(Component):
    """ Computes total weight """

    def __init__(self, aero_ind, mrho, fem_origin=0.35):
        super(SpatialBeamWeight, self).__init__()

        ny = aero_ind[0, 1]
        tot_n = numpy.sum(aero_ind[:, 2])
        self.aero_ind = aero_ind

        self.add_param('A', val=numpy.zeros((ny-1)))
        self.add_param('mesh', val=numpy.zeros((tot_n, 3)))
        self.add_output('weight', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

        elem_IDs = numpy.zeros((ny-1, 2), int)
        elem_IDs[:, 0] = numpy.arange(ny-1)
        elem_IDs[:, 1] = numpy.arange(ny-1) + 1
        self.elem_IDs = elem_IDs

        self.mrho = mrho
        self.fem_origin = fem_origin

    def solve_nonlinear(self, params, unknowns, resids):
        nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = self.aero_ind[0, :]
        mesh = params['mesh'][i:i+n, :].reshape(nx, ny, 3)
        A = params['A']
        num_elems = self.elem_IDs.shape[0]

        w = self.fem_origin
        nodes = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

        volume = 0.
        for ielem in xrange(num_elems):
            in0, in1 = self.elem_IDs[ielem, :]
            P0 = nodes[in0, :]
            P1 = nodes[in1, :]
            L = norm(P1 - P0)
            volume += L * A[ielem]

        unknowns['weight'] = volume  * self.mrho * 9.81

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['weight', 't'][0, :] = 1.0
        return jac



class SpatialBeamVonMisesTube(Component):
    """ Computes the max Von Mises stress in each element """

    def __init__(self, aero_ind, E, G, fem_origin=0.35):
        super(SpatialBeamVonMisesTube, self).__init__()

        ny = aero_ind[0, 1]
        tot_n = numpy.sum(aero_ind[:, 2])
        self.aero_ind = aero_ind

        self.add_param('mesh', val=numpy.zeros((tot_n, 3), dtype="complex"))
        self.add_param('r', val=numpy.zeros((ny-1), dtype="complex"))
        self.add_param('disp', val=numpy.zeros((ny, 6), dtype="complex"))

        self.add_output('vonmises', val=numpy.zeros((ny-1, 2), dtype="complex"))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

        elem_IDs = numpy.zeros((ny-1, 2), int)
        elem_IDs[:, 0] = numpy.arange(ny-1)
        elem_IDs[:, 1] = numpy.arange(ny-1) + 1
        self.elem_IDs = elem_IDs

        self.T_elem = numpy.zeros((12, 12), dtype='complex')
        self.T = numpy.zeros((3, 3), dtype='complex')
        self.x_gl = numpy.array([1, 0, 0], dtype='complex')

        self.E = E
        self.G = G
        self.fem_origin = fem_origin

    def solve_nonlinear(self, params, unknowns, resids):
        elem_IDs = self.elem_IDs

        r = params['r']
        nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = self.aero_ind[0, :]
        mesh = params['mesh'][i:i+n, :].reshape(nx, ny, 3)
        disp = params['disp']
        vonmises = unknowns['vonmises']

        w = self.fem_origin
        nodes = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

        num_elems = elem_IDs.shape[0]
        for ielem in xrange(num_elems):
            in0, in1 = elem_IDs[ielem, :]

            P0 = nodes[in0, :]
            P1 = nodes[in1, :]
            L = norm(P1 - P0)

            d1 = disp[in0, :3]

            x_loc = unit(P1 - P0)
            y_loc = unit(numpy.cross(x_loc, self.x_gl))
            z_loc = unit(numpy.cross(x_loc, y_loc))

            self.T[0, :] = x_loc
            self.T[1, :] = y_loc
            self.T[2, :] = z_loc

            u0x, u0y, u0z = self.T.dot(disp[in0, :3])
            r0x, r0y, r0z = self.T.dot(disp[in0, 3:])
            u1x, u1y, u1z = self.T.dot(disp[in1, :3])
            r1x, r1y, r1z = self.T.dot(disp[in1, 3:])

            tmp = numpy.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
            sxx0 = self.E * (u1x - u0x) / L \
                  + self.E * r[ielem] / L * tmp
            sxx1 = self.E * (u0x - u1x) / L \
                  + self.E * r[ielem] / L * tmp
            sxt = self.G * r[ielem] * (r1x - r0x) / L

            vonmises[ielem, 0] = numpy.sqrt(sxx0**2 + sxt**2)
            vonmises[ielem, 1] = numpy.sqrt(sxx1**2 + sxt**2)


class SpatialBeamFailureKS(Component):
    """ Aggregates failure constraints from the structure """

    def __init__(self, aero_ind, sigma, rho=10):
        super(SpatialBeamFailureKS, self).__init__()

        ny = aero_ind[0, 1]

        self.add_param('vonmises', val=numpy.zeros((ny-1, 2)))

        self.add_output('failure', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

        self.sigma = sigma
        self.rho = rho

    def solve_nonlinear(self, params, unknowns, resids):
        sigma = self.sigma
        rho = self.rho
        vonmises = params['vonmises']

        fmax = numpy.max(vonmises/sigma - 1)

        nlog, nsum, nexp = numpy.log, numpy.sum, numpy.exp
        unknowns['failure'] = fmax + 1 / rho * \
                              nlog(nsum(nexp(rho * (vonmises/sigma - 1 - fmax))))



class SpatialBeamStates(Group):

    def __init__(self, aero_ind, fem_ind, E, G):
        super(SpatialBeamStates, self).__init__()

        n = aero_ind[0, 1]

        cons = numpy.array([int((n-1)/2)])

        self.add('fem',
                 SpatialBeamFEM(aero_ind, fem_ind, cons, E, G),
                 promotes=['*'])
        self.add('disp',
                 SpatialBeamDisp(aero_ind, cons),
                 promotes=['*'])



class SpatialBeamFunctionals(Group):

    def __init__(self, aero_ind, E, G, stress, mrho):
        super(SpatialBeamFunctionals, self).__init__()

        self.add('energy',
                 SpatialBeamEnergy(aero_ind),
                 promotes=['*'])
        self.add('weight',
                 SpatialBeamWeight(aero_ind, mrho),
                 promotes=['*'])
        self.add('vonmises',
                 SpatialBeamVonMisesTube(aero_ind, E, G),
                 promotes=['*'])
        self.add('failure',
                 SpatialBeamFailureKS(aero_ind, stress),
                 promotes=['*'])
