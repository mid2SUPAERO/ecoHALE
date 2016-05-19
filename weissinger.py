from __future__ import division
import numpy
from time import time

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve
import lib


def norm(vec):
    return numpy.sqrt(numpy.sum(vec**2))


def _biot_savart(A, B, P, inf=False, rev=False, eps=1e-5):
    """
    Apply Biot-Savart's law to compute v
    induced at a control point by a vortex line
    - A[3], B[3] : coordinates associated with the vortex line
    - inf : the vortex line is semi-infinite, originating at A
    - rev : signifies the following about the direction of the vortex line:
       If False, points from A to B
       If True,  points from B to A
    - eps : parameter used to avoid singularities when points are on a vortex line
    """

    rPA = norm(A - P)
    rPB = norm(B - P)
    rAB = norm(B - A)
    rH = norm(P - A - numpy.dot((B - A), (P - A)) / \
              numpy.dot((B - A), (B - A)) * (B - A)) + eps
    cosA = numpy.dot((P - A), (B - A)) / (rPA * rAB)
    cosB = numpy.dot((P - B), (A - B)) / (rPB * rAB)
    C = numpy.cross(B - P, A - P)
    C /= norm(C)

    if inf:
        v = -C / rH * (cosA + 1) / (4 * numpy.pi)
    else:
        v = -C / rH * (cosA + cosB) / (4 * numpy.pi)

    if rev:
        v = -v

    return v


def _assemble_AIC_mtx(mtx, mesh, points, b_pts, alpha):
    """
    Compute the aerodynamic influence coefficient matrix
    either for the circulation linear system or Trefftz-plane drag computation
    - mtx[num_y-1, num_y-1, 3] : derivative of v*n w.r.t. circulation
    - mesh[2, num_y, 3] : contains LE and TE coordinates at each section
    - points[num_y-1, 3] : control points
    - b_pts[num_y, 3] : bound vortex coordinates
    """
    
    num_y = mesh.shape[1]

    if 1:
        mtx[:, :, :] = lib.assembleaeromtx(num_y, alpha, mesh, points, b_pts)
        #mtx[:, :, :] = lib.assembleaeromtx(num_y, alpha.real, mesh.real, points.real, b_pts.real)
    else:
        mtx[:, :, :] = 0.0
        alpha = alpha * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        # Loop through control points
        for ind_i in xrange(num_y - 1):
            P = points[ind_i]

            # Loop through elements
            for ind_j in xrange(num_y - 1):
                A = b_pts[ind_j + 0, :]
                B = b_pts[ind_j + 1, :]
                D = mesh[-1, ind_j + 0, :]
                E = mesh[-1, ind_j + 1, :]
                F = D + numpy.array([cosa, 0, sina])
                G = E + numpy.array([cosa, 0, sina])

                mtx[ind_i, ind_j, :] += _biot_savart(A, B, P, inf=False, rev=False)
                mtx[ind_i, ind_j, :] += _biot_savart(B, E, P, inf=False, rev=False)
                mtx[ind_i, ind_j, :] += _biot_savart(A, D, P, inf=False, rev=True)
                mtx[ind_i, ind_j, :] += _biot_savart(E, G, P, inf=True,  rev=False)
                mtx[ind_i, ind_j, :] += _biot_savart(D, F, P, inf=True,  rev=True)


class WeissingerGeometry(Component):
    """ Computes various geometric properties for Weissinger analysis """

    def __init__(self, n):
        super(WeissingerGeometry, self).__init__()

        self.add_param('def_mesh', val=numpy.zeros((2, n, 3)))
        self.add_output('b_pts', val=numpy.zeros((n, 3)))
        self.add_output('c_pts', val=numpy.zeros((n-1, 3)))
        self.add_output('widths', val=numpy.zeros((n-1)))
        self.add_output('cos_dih', val=numpy.zeros((n-1)))
        self.add_output('normals', val=numpy.zeros((n-1, 3)))
        self.add_output('S_ref', val=0.)
        self.num_y = n

        self.fd_options['force_fd'] = True   # Not worth doing manual partials
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def _get_lengths(self, A, B, axis):
        return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['def_mesh']
        unknowns['b_pts'] = mesh[0, :, :] * .75 + mesh[1, :, :] * .25
        unknowns['c_pts'] = 0.5 * 0.25 * mesh[0, :-1, :] + \
                            0.5 * 0.75 * mesh[1, :-1, :] + \
                            0.5 * 0.25 * mesh[0,  1:, :] + \
                            0.5 * 0.75 * mesh[1,  1:, :]

        b_pts = unknowns['b_pts']
        unknowns['widths'] = self._get_lengths(b_pts[1:, :], b_pts[:-1, :], 1)
        unknowns['cos_dih'] = (b_pts[1:, 1] - b_pts[:-1, 1]) / unknowns['widths']

        normals = numpy.cross(
            mesh[:-1,  1:, :] - mesh[ 1:, :-1, :],
            mesh[:-1, :-1, :] - mesh[ 1:,  1:, :],
            axis=2)

        norms = numpy.sqrt(numpy.sum(normals**2, axis=2))
        for ind in xrange(3):
            normals[:, :, ind] /= norms

        unknowns['normals'] = normals

        unknowns['S_ref'] = 0.5 * numpy.sum(norms)


class WeissingerCirculations(Component):
    """ Defines circulations """

    def __init__(self, n):
        super(WeissingerCirculations, self).__init__()
        self.add_param('v', val=10.)
        self.add_param('alpha', val=3.)
        self.add_param('def_mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('b_pts', val=numpy.zeros((n, 3)))
        self.add_param('c_pts', val=numpy.zeros((n-1, 3)))
        self.add_state('circulations', val=numpy.zeros((n-1)))

#        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "forward"
        self.fd_options['linearize'] = True # only for circulations

        size = n - 1
        self.num_y = n
        self.mtx = numpy.zeros((size, size), dtype="complex")
        self.mtx2 = numpy.zeros((size, size, 3), dtype="complex")
        self.rhs = numpy.zeros((size), dtype="complex")

    def _assemble_system(self, params):
        _assemble_AIC_mtx(self.mtx2, params['def_mesh'],
                          params['c_pts'], params['b_pts'], params['alpha'])
            
        self.mtx[:, :] = 0.
        for ind in xrange(3):
            self.mtx[:, :] += (self.mtx2[:, :, ind].T * params['normals'][:, ind]).T

        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'] * numpy.array([cosa, 0., sina], dtype="complex")
        self.rhs[:] = -params['normals'].dot(v_inf)

    def solve_nonlinear(self, params, unknowns, resids):
        self._assemble_system(params)
        unknowns['circulations'] = numpy.linalg.solve(self.mtx, self.rhs)

    def apply_nonlinear(self, params, unknowns, resids):
        self._assemble_system(params)

        circ = unknowns['circulations']
        resids['circulations'] = self.mtx.dot(circ) - self.rhs

    def linearize(self, params, unknowns, resids):
        """ Jacobian for circulations."""

        self.lup = lu_factor(self.mtx.real)
        jac = self.alloc_jacobian()

        n = self.num_y

        boop = time()

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['normals', 'def_mesh',
                                                    'b_pts', 'c_pts'],
                                         fd_states=[])
        jac.update(fd_jac)

        print '@@@@ cost: {} secs'.format(numpy.round(time()-boop, 5))

        jac['circulations', 'circulations'] = self.mtx.real

        normals = params['normals'].real
        alpha = params['alpha'].real * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'].real * numpy.array([cosa, 0., sina], dtype="complex")

        jac['circulations', 'v'][:, 0] = -self.rhs.real / params['v'].real

        dv_da = params['v'].real * numpy.array([-sina, 0., cosa]) * numpy.pi / 180.
        jac['circulations', 'alpha'][:, 0] = normals.dot(dv_da)

        # print jac['circulations', 'normals']
        # print normals

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


class WeissingerForces(Component):
    """ Defines aerodynamic forces acting on each section """

    def __init__(self, n):
        super(WeissingerForces, self).__init__()
        self.add_param('def_mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('b_pts', val=numpy.zeros((n, 3)))
        self.add_param('c_pts', val=numpy.zeros((n-1, 3)))
        self.add_param('circulations', val=numpy.zeros((n-1)))
        self.add_param('alpha', val=3.)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_output('sec_forces', val=numpy.zeros((n-1, 3)))

        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "forward"

        size = n - 1
        self.num_y = n
        self.mtx = numpy.zeros((size, size, 3), dtype="complex")
        self.v = numpy.zeros((size, 3), dtype="complex")

    def solve_nonlinear(self, params, unknowns, resids):
        circ = params['circulations']

        _assemble_AIC_mtx(self.mtx, params['def_mesh'],
                          params['c_pts'], params['b_pts'], params['alpha'])

        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        for ind in xrange(3):
            self.v[:, ind] = self.mtx[:, :, ind].dot(params['circulations'])
        self.v[:, 0] += cosa * params['v']
        self.v[:, 2] += sina * params['v']

        bound = params['b_pts'][1:, :] - params['b_pts'][:-1, :]

        cross = numpy.cross(self.v, bound)
        for ind in xrange(3):
            unknowns['sec_forces'][:, ind] = params['rho'] * circ * cross[:, ind]

    def linearize(self, params, unknowns, resids):
        """ Jacobian for forces."""

        s = time()

        jac = self.alloc_jacobian()

        n = self.num_y

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['def_mesh', 'b_pts',
                                                    'c_pts', 'alpha'],
                                         fd_states=[])
        jac.update(fd_jac)

        arange = numpy.arange(n-1)
        circ = params['circulations'].real
        rho = params['rho'].real
        v = params['v'].real
        widths = params['widths'].real
        sec_forces = unknowns['sec_forces'].real

        jac['sec_forces', 'v'] = sec_forces.flatten() / v
        jac['sec_forces', 'rho'] = sec_forces.flatten() / rho

        forces_circ = jac['sec_forces', 'circulations']
        for ind in xrange(3):
             forces_circ[ind+3*arange, arange] = sec_forces[:, ind] / circ

        print '### Time: {} secs'.format(numpy.round(time()-s, 6))

        # print jac['sec_forces', 'circulations']
        # exit()

        return jac



class WeissingerForces2(Component):
    """ Computes section forces """

    def __init__(self, n):
        super(WeissingerForces, self).__init__()

        self.add_param('circulations', val=numpy.zeros((n-1)))
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_param('widths', val=numpy.zeros((n-1)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_output('sec_forces', val=numpy.zeros((n-1, 3)))

        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        # pre-allocate memory is a little faster
        n_segs = n-1

        self.arange = numpy.arange(n-1)

    def solve_nonlinear(self, params, unknowns, resids):
        circ = params['circulations']
        rho = params['rho']
        v = params['v']
        widths = params['widths']
        normals = params['normals']

        sec_forces = numpy.array(normals, dtype="complex")
        for ind in xrange(3):
            sec_forces[:, ind] *= rho * v * circ * widths
        unknowns['sec_forces'] = sec_forces

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift."""

        jac = self.alloc_jacobian()

        arange = self.arange
        circ = params['circulations'].real
        rho = params['rho'].real
        v = params['v'].real
        widths = params['widths'].real
        sec_forces = unknowns['sec_forces'].real

        jac['sec_forces', 'v'] = sec_forces.flatten() / v
        jac['sec_forces', 'rho'] = sec_forces.flatten() / rho

        forces_circ = jac['sec_forces', 'circulations']
        for ind in xrange(3):
            forces_circ[ind+3*arange, arange] = sec_forces[:, ind] / circ

        forces_widths = jac['sec_forces', 'widths']
        for ind in xrange(3):
            forces_widths[ind+3*arange, arange] = sec_forces[:, ind] / widths

        forces_normals = jac['sec_forces', 'normals']
        for ind in xrange(3):
            forces_normals[ind+3*arange, ind+3*arange] = rho * v * circ * widths

        return jac


class WeissingerLiftDrag(Component):
    """ Calculates total lift in force units """

    def __init__(self, n):
        super(WeissingerLiftDrag, self).__init__()

        self.add_param('sec_forces', val=numpy.zeros((n-1, 3)))
        self.add_param('alpha', val=3.)
        self.add_output('L', val=0.)
        self.add_output('D', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self.num_y = n

    def solve_nonlinear(self, params, unknowns, resids):
        alpha = params['alpha'] * numpy.pi / 180.
        forces = params['sec_forces']

        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        unknowns['L'] = numpy.sum(-forces[:, 0] * sina + forces[:, 2] * cosa)
        unknowns['D'] = numpy.sum( forces[:, 0] * cosa + forces[:, 2] * sina)

class WeissingerCoeffs(Component):
    """ Computes lift coefficient """

    def __init__(self, n):
        super(WeissingerCoeffs, self).__init__()

        self.add_param('S_ref', val=0.)
        self.add_param('L', val=0.)
        self.add_param('D', val=0.)
        self.add_param('v', val=0.)
        self.add_param('rho', val=0.)
        self.add_output('CL1', val=0.)
        self.add_output('CDi', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        D = params['D']
        unknowns['CL1'] = L / (0.5*rho*v**2*S_ref)
        unknowns['CDi'] = D / (0.5*rho*v**2*S_ref)


class WeissingerLift(Component):
    """ Calculates total lift in force units """

    def __init__(self, n):
        super(WeissingerLift, self).__init__()

        self.add_param('cos_dih', val=numpy.zeros((n-1)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('sec_forces', val=numpy.zeros((n-1, 3)))
        self.add_output('L', val=0.)

        #self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self.num_y = n

    def solve_nonlinear(self, params, unknowns, resids):
        cos_dih = params['cos_dih']
        normals = params['normals']
        sec_forces = params['sec_forces']

        L = sec_forces[:, 2] / normals[:, 2]
        unknowns['L'] = numpy.sum(L.T * cos_dih)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift."""

        jac = self.alloc_jacobian()

        cos_dih = params['cos_dih'].real
        normals = params['normals'].real
        sec_forces = params['sec_forces'].real

        n = self.num_y
        arange = numpy.arange(n-1)

        lift_cos = jac['L', 'cos_dih']
        lift_cos[0, :] = sec_forces[:, 2] / normals[:, 2]

        lift_normals = jac['L', 'normals']
        lift_normals[0, 3*arange+2] = -sec_forces[:, 2] / normals[:, 2]**2 * cos_dih

        lift_forces = jac['L', 'sec_forces']
        lift_forces[0, 3*arange+2] = cos_dih / normals[:, 2]

        return jac


class WeissingerLiftCoeff(Component):
    """ Computes lift coefficient """

    def __init__(self, n):
        super(WeissingerLiftCoeff, self).__init__()

        self.add_param('S_ref', val=0.)
        self.add_param('L', val=0.)
        self.add_param('v', val=0.)
        self.add_param('rho', val=0.)
        self.add_output('CL1', val=0.)

        #self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        unknowns['CL1'] = L / (0.5*rho*v**2*S_ref)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift."""

        jac = self.alloc_jacobian()

        S_ref = params['S_ref'].real
        rho = params['rho'].real
        v = params['v'].real
        L = params['L'].real

        jac['CL1', 'S_ref'] = -L / (0.5*rho*v**2*S_ref**2)
        jac['CL1', 'L'] = 1.0 / (0.5*rho*v**2*S_ref)
        jac['CL1', 'v'] = -2 * L / (0.5*rho*v**3*S_ref)
        jac['CL1', 'rho'] = -L / (0.5*rho**2*v**2*S_ref)

        return jac


class WeissingerDragCoeff(Component):
    """ Calculates induced drag coefficient """

    def __init__(self, n):
        super(WeissingerDragCoeff, self).__init__()
        self.add_param('v', val=0.)
        self.add_param('circulations', val=numpy.zeros((n-1)))
        self.add_param('alpha', val=3.)
        self.add_param('def_mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('b_pts', val=numpy.zeros((n, 3)))
        self.add_param('widths', val=numpy.zeros((n-1)))
        self.add_param('S_ref', val=0.)
        self.add_output('CDi', val=0., desc="induced drag coefficient")

        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self._trefftz_dist = 10000.
        self.mtx = numpy.zeros((n - 1, n - 1), dtype="complex")
        self.intersections = numpy.zeros((n - 1, 2, 3), dtype="complex")
        self.new_normals = numpy.zeros((n - 1, 3), dtype="complex")

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['def_mesh']
        b_pts = params['b_pts']
        num_y = mesh.shape[1]

        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        trefftz_normal = numpy.array([cosa, 0., sina], dtype="complex")
        a = [self._trefftz_dist, 0, 0]

        for ind in xrange(num_y - 1):
            A = b_pts[ind + 0, :]
            B = b_pts[ind + 1, :]
            D = mesh[1, ind + 0, :]
            E = mesh[1, ind + 1, :]

            t = -numpy.dot(trefftz_normal, (A - a)) / numpy.dot(trefftz_normal, D - A)
            self.intersections[ind, 0, :] = A + (D - A) * t

            t = -numpy.dot(trefftz_normal, (B - a)) / numpy.dot(trefftz_normal, E - B)
            self.intersections[ind, 1, :] = B + (E - B) * t

        trefftz_points = (self.intersections[:, 1, :] + self.intersections[:, 0, :]) / 2.

        '''
        # Equation of trefftz plane is P*N = T*N
        # (P+sN)*N = T*N
        N = trefftz_normal
        T = numpy.array([self._trefftz_dist, 0, 0], dtype="complex")
        for ind in xrange(num_y - 1):
            P = 0.5 * mesh[1, ind + 0, :] + 0.5 * mesh[1, ind + 1, :]
            s = numpy.dot(T-P, N)
            trefftz_points[ind] = P + s * N
        '''

        normals = params['normals']
        for ind in xrange(num_y - 1):
            self.new_normals[ind] = normals[ind] - numpy.dot(normals[ind], trefftz_normal) \
                                    * trefftz_normal / norm(trefftz_normal)
        _assemble_AIC_mtx(self.mtx, params['def_mesh'], self.new_normals,
                          trefftz_points, params['b_pts'])

        self.velocities = -numpy.dot(self.mtx, params['circulations']) / params['v']
        unknowns['CDi'] = 1. / params['S_ref'] / params['v'] * \
                         numpy.sum(params['circulations'] * self.velocities * params['widths'])

    def linearize(self, params, unknowns, resids):
        """ Jacobian for drag."""

        jac = self.alloc_jacobian()
        fd_jac = self.complex_step_jacobian(params, unknowns, resids,\
                                         fd_params=['def_mesh', 'alpha', 'normals', 'b_pts'])
        jac.update(fd_jac)

        circ = params['circulations'].real
        widths = params['widths'].real
        v = params['v'].real
        S_ref = params['S_ref'].real
        CD = unknowns['CDi'].real
        velocities = self.velocities.real

        jac['CDi', 'v'] = -2 * CD / v
        jac['CDi', 'S_ref'] = -CD / S_ref
        jac['CDi', 'circulations'][0, :] = 1. / S_ref / v * velocities * widths \
                                          - 1. / S_ref / v**2 * self.mtx.T.real.dot(circ * widths)
        jac['CDi', 'widths'][0, :] = 1. / S_ref / v * velocities * circ

        return jac



class TotalLift(Component):

    def __init__(self, CL0):
        super(TotalLift, self).__init__()

        self.add_param('CL1', val=1.)
        self.add_output('CL', val=1.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self.CL0 = CL0

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['CL'] = params['CL1'] + self.CL0

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['CL', 'CL1'] = 1
        return jac



class TotalDrag(Component):

    def __init__(self, CD0):
        super(TotalDrag, self).__init__()

        self.add_param('CDi', val=1.)
        self.add_output('CD', val=1.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self.CD0 = CD0

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['CD'] = params['CDi'] + self.CD0



class WeissingerStates(Group):

    def __init__(self, num_y):
        super(WeissingerStates, self).__init__()

        self.add('wgeom',
                 WeissingerGeometry(num_y),
                 promotes=['*'])
        self.add('circ',
                 WeissingerCirculations(num_y),
                 promotes=['*'])
        self.add('forces',
                 WeissingerForces(num_y),
                 promotes=['*'])



class WeissingerFunctionals(Group):

    def __init__(self, num_y, CL0, CD0):
        super(WeissingerFunctionals, self).__init__()

        '''
        self.add('lift',
                 WeissingerLift(num_y),
                 promotes=['*'])
        self.add('CL1',
                 WeissingerLiftCoeff(num_y),
                 promotes=['*'])
        self.add('CDi',
                 WeissingerDragCoeff(num_y),
                 promotes=['*'])
        '''
        self.add('liftdrag',
                 WeissingerLiftDrag(num_y),
                 promotes=['*'])
        self.add('coeffs',
                 WeissingerCoeffs(num_y),
                 promotes=['*'])
        self.add('CL',
                 TotalLift(CL0),
                 promotes=['*'])
        self.add('CD',
                 TotalDrag(CD0),
                 promotes=['*'])
