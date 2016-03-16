from __future__ import division
import numpy

from openmdao.api import Component
from scipy.linalg import lu_factor, lu_solve



def norm(vec):
    return numpy.sqrt(numpy.sum(vec**2))


def _biot_savart(N, A, B, P, inf=False, rev=False, eps=1e-5):
    """
    Apply Biot-Savart's law to compute v*n
    induced at a control point by a vortex line
    - N[3] : the normal vector
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
    rH = norm(P - A - numpy.dot((B - A), (P - A)) / numpy.dot((B - A), (B - A)) * (B - A)) + eps
    cosA = numpy.dot((P - A), (B - A)) / (rPA * rAB)
    cosB = numpy.dot((P - B), (A - B)) / (rPB * rAB)
    C = numpy.cross(B - P, A - P)
    C /= norm(C)

    if inf:
        vdn = -numpy.dot(N, C) / rH * (cosA + 1) / (4 * numpy.pi)
    else:
        vdn = -numpy.dot(N, C) / rH * (cosA + cosB) / (4 * numpy.pi)

    if rev:
        vdn = -vdn

    return vdn



def _assemble_AIC_mtx(mtx, mesh, normals, points, b_pts):
    """
    Compute the aerodynamic influence coefficient matrix
    either for the circulation linear system or Trefftz-plane drag computation
    - mtx[num_y-1, num_y-1, 3] : derivative of v*n w.r.t. circulation
    - mesh[2, num_y, 3] : contains LE and TE coordinates at each section
    - normals[num_y-1, 3] : normals vectors for the v*n for each control point
    - points[num_y-1, 3] : control points
    - b_pts[num_y, 3] : bound vortex coordinates
    """

    num_y = mesh.shape[1]

    mtx[:, :] = 0.0

    # Loop through control points
    for ind_i in xrange(num_y - 1):
        N = normals[ind_i]
        P = points[ind_i]

        # Loop through elements
        for ind_j in xrange(num_y - 1):
            A = b_pts[ind_j + 0, :]
            B = b_pts[ind_j + 1, :]
            D = mesh[-1, ind_j + 0, :]
            E = mesh[-1, ind_j + 1, :]

            mtx[ind_i, ind_j] += _biot_savart(N, A, B, P, inf=False, rev=False)
            mtx[ind_i, ind_j] += _biot_savart(N, B, E, P, inf=True,  rev=False)
            mtx[ind_i, ind_j] += _biot_savart(N, A, D, P, inf=True,  rev=True)



class WeissingerPreproc(Component):
    """ Computes various geometric properties for Weissinger analysis """

    def __init__(self, n):
        super(WeissingerPreproc, self).__init__()

        self.add_param('mesh', val=numpy.zeros((2, n, 3)))
        self.add_output('normals', val=numpy.zeros((n-1, 3)))
        self.add_output('b_pts', val=numpy.zeros((n, 3)))
        self.add_output('c_pts', val=numpy.zeros((n-1, 3)))
        self.add_output('widths', val=numpy.zeros((n-1)))
        self.add_output('cos_dih', val=numpy.zeros((n-1)))
        self.add_output('S_ref', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def _get_lengths(self, A, B, axis):
        return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['mesh']
        unknowns['b_pts'] = mesh[0, :, :] * .75 + mesh[1, :, :] * .25
        unknowns['c_pts'] = \
                            0.5 * 0.25 * mesh[0, :-1, :] + \
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

    def linearize(self, params, unknowns, resids):
        J = {}
        mesh = params['mesh']

        b_pts_size = numpy.prod(mesh.shape[1:])
        b_pts_eye = numpy.eye(b_pts_size)
        J['b_pts', 'mesh'] = numpy.hstack((.75 * b_pts_eye, .25 * b_pts_eye))

        cols_size = mesh.shape[1] * 6
        rows_size = (mesh.shape[1] - 1) * 3
        row = numpy.zeros((cols_size))
        row[0] = .125
        row[3] = .125
        row[cols_size / 2 + 0] = .375
        row[cols_size / 2 + 3] = .375
        c_pts_mat = numpy.zeros((rows_size, cols_size))
        for i in range(rows_size):
            c_pts_mat[i, :] = numpy.roll(row, i)
        J['c_pts', 'mesh'] = c_pts_mat

        cols_size = numpy.prod(mesh.shape)
        rows_size = mesh.shape[1] - 1
        row = numpy.zeros((cols_size))
        row[1] = -.75
        row[4] = .75
        row[cols_size / 2 + 1] = -.25
        row[cols_size / 2 + 4] = .25
        widths_mat = numpy.zeros((rows_size, cols_size))
        for i in range(rows_size):
            widths_mat[i, :] = numpy.roll(row, i*3)
        J['widths', 'mesh'] = widths_mat

        # TODO:
        # J['normals', 'mesh'] =
        # J['cos_dih', 'mesh'] =
        # J['S_ref', 'mesh'] =

        return J



class WeissingerCirculations(Component):
    """ Defines circulations """

    def __init__(self, n):
        super(WeissingerCirculations, self).__init__()
        self.add_param('v', val=10.)
        self.add_param('alpha', val=3.)
        self.add_param('mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('b_pts', val=numpy.zeros((n, 3)))
        self.add_param('c_pts', val=numpy.zeros((n-1, 3)))
        self.add_state('circulations', val=numpy.zeros((n-1)))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"
        self.fd_options['linearize'] = True # only for circulations

        size = n - 1
        self.num_y = n
        self.mtx = numpy.zeros((size, size), dtype="complex")
        self.rhs = numpy.zeros((size), dtype="complex")

    def _assemble_system(self, params):
        _assemble_AIC_mtx(self.mtx, params['mesh'], params['normals'],
                          params['c_pts'], params['b_pts'])

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

    def solve_linear(self, dumat, drmat, vois, mode=None):

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t = 0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t = 1

        for voi in vois:
            sol_vec[voi].vec[:] = lu_solve(self.lup, -rhs_vec[voi].vec, trans=t)



class WeissingerForces(Component):
    """ Computes section forces """

    def __init__(self, n):
        super(WeissingerForces, self).__init__()

        self.add_param('circulations', val=numpy.zeros((n-1)))
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_param('widths', val=numpy.zeros((n-1)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_output('sec_forces', val=numpy.zeros((n-1, 3)))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

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

    def linearize(sel params, unknowns, resids):
        """ Jacobian for lift."""
        J = {}
        circ = params['circulations']
        rho = params['rho']
        v = params['v']
        widths = params['widths']
        normals = params['normals']

        n = widths.shape[0]
        sec_forces = numpy.array(normals)
        for ind in xrange(3):
            sec_forces[:, ind] *= rho * v * circ * widths
        J['sec_forces', 'v'] = sec_forces.reshape(n*3) / v * 2.
        J['sec_forces', 'rho'] = sec_forces.reshape(n*3) / rho

        forces_circ = numpy.zeros((3*n, n))
        for ix in xrange(n):
            for iy in xrange(3):
                forces_circ[iy + ix*3, ix] = sec_forces[ix, iy] / circ[ix]
        J['sec_forces', 'circulations'] = forces_circ

        forces_widths = numpy.zeros((3*n, n))
        for ix in xrange(n):
            for iy in xrange(3):
                forces_widths[iy + ix*3, ix] = sec_forces[ix, iy] / widths[ix]
        J['sec_forces', 'widths'] = forces_widths

        forces_normals = numpy.zeros((3*n, 3*n))
        for ix in xrange(n):
            for iy in xrange(3):
                forces_normals[iy + ix*3, iy + ix*3] = sec_forces[ix, iy] / normals[ix, iy]
        J['sec_forces', 'normals'] = forces_normals



class WeissingerLift(Component):
    """ Calculates total lift in force units """

    def __init__(self, n):
        super(WeissingerLift, self).__init__()

        self.add_param('cos_dih', val=numpy.zeros((n-1)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('sec_forces', val=numpy.zeros((n-1, 3)))
        self.add_output('L', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        cos_dih = params['cos_dih']
        normals = params['normals']
        sec_forces = params['sec_forces']

        L = sec_forces[:, 2] / normals[:, 2]
        unknowns['L'] = numpy.sum(L.T * cos_dih)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift."""
        J = {}
        cos_dih = params['cos_dih']
        normals = params['normals']
        sec_forces = params['sec_forces']

        forces_circ = numpy.zeros((3*n, n))
        for ix in xrange(n):
            for iy in xrange(3):
                forces_circ[iy + ix*3, ix] = sec_forces[ix, iy] / circ[ix]
        J['sec_forces', 'circulations'] = forces_circ

        forces_widths = numpy.zeros((3*n, n))
        for ix in xrange(n):
            for iy in xrange(3):
                forces_widths[iy + ix*3, ix] = sec_forces[ix, iy] / widths[ix]
        J['sec_forces', 'widths'] = forces_widths

        forces_normals = numpy.zeros((3*n, 3*n))
        for ix in xrange(n):
            for iy in xrange(3):
                forces_normals[iy + ix*3, iy + ix*3] = sec_forces[ix, iy] / normals[ix, iy]
        J['sec_forces', 'normals'] = forces_normals

        return J



class WeissingerLiftCoeff(Component):
    """ Computes lift coefficient """

    def __init__(self, n):
        super(WeissingerLiftCoeff, self).__init__()

        self.add_param('S_ref', val=0.)
        self.add_param('L', val=0.)
        self.add_param('v', val=0.)
        self.add_param('rho', val=0.)
        self.add_output('CL', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        unknowns['CL'] = params['L'] / (0.5*rho*v**2*S_ref)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift."""
        J = {}
        J['CL', 'v'] = 0.
        J['CL', 'rho'] = 0.
        J['CL', 'S_ref'] = 0.
        J['CL', 'L'] = 1. / (0.5*self.rho*self.v**2*params['S_ref'])

        return J



class WeissingerDragCoeff(Component):
    """ Calculates induced drag coefficient """

    def __init__(self, n):
        super(WeissingerDragCoeff, self).__init__()
        self.add_param('v', val=0.)
        self.add_param('circulations', val=numpy.zeros((n-1)))
        self.add_param('alpha', val=3.)
        self.add_param('mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('b_pts', val=numpy.zeros((n, 3)))
        self.add_param('widths', val=numpy.zeros((n-1)))
        self.add_param('S_ref', val=0.)
        self.add_output('CD', val=0.)

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        self._trefftz_dist = 10000.
        self.mtx = numpy.zeros((n - 1, n - 1), dtype="complex")
        self.intersections = numpy.zeros((n - 1, 2, 3), dtype="complex")
        self.new_normals = numpy.zeros((n - 1, 3), dtype="complex")

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['mesh']
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

        normals = params['normals']
        for ind in xrange(num_y - 1):
            self.new_normals[ind] = normals[ind] - numpy.dot(normals[ind], trefftz_normal) * trefftz_normal / norm(trefftz_normal)
        _assemble_AIC_mtx(self.mtx, params['mesh'], self.new_normals,
                          trefftz_points, params['b_pts'])

        velocities = -numpy.dot(self.mtx, params['circulations']) / params['v']
        unknowns['CD'] = 1. / params['S_ref'] / params['v'] * numpy.sum(params['circulations'] * velocities * params['widths'])

    def linearize(self, params, unknowns, resids):
        """ Jacobian for drag."""
        J = {}
        circ = params['circulations']
        v = params['v']
        alpha = params['alpha']
        b_pts = params['b_pts']
        mesh = params['mesh']

        n = mesh.shape[1]
        J['CD', 'v'] = 0.
        J['CD', 'alpha'] = 0.
        J['CD', 'b_pts'] = numpy.zeros((1, n * 3))

        # TODO:
        # J['CD', 'circulations'] =
        # J['CD', 'trefftz_dist'] =  # not sure if this one is needed
        # J['CD', 'mesh'] =
        # J['CD', 'normals'] =
        # J['CD', 'widths'] =
        # J['CD', 'S_ref'] =

        return J
