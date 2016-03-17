from __future__ import division
import numpy

from openmdao.api import Component, Group
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
    rH = norm(P - A - numpy.dot((B - A), (P - A)) / \
              numpy.dot((B - A), (B - A)) * (B - A)) + eps
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

        self.add_param('def_mesh', val=numpy.zeros((2, n, 3)))
        self.add_output('b_pts', val=numpy.zeros((n, 3)))
        self.add_output('c_pts', val=numpy.zeros((n-1, 3)))
        self.add_output('widths', val=numpy.zeros((n-1)))
        self.add_output('cos_dih', val=numpy.zeros((n-1)))
        self.add_output('normals', val=numpy.zeros((n-1, 3)))
        self.add_output('S_ref', val=0.)

        self.fd_options['force_fd'] = True   # Not worth doing manual partials
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

        arange = numpy.arange(3*n)
        bpts_mesh = numpy.zeros((3*n, 6*n))
        bpts_mesh[arange, arange] = 0.75
        bpts_mesh[arange, arange+3*n] = 0.25
        self.bpts_mesh = bpts_mesh
        
        arange = numpy.arange(3*n-3)
        imesh = numpy.arange(6*n).reshape((2, n, 3))
        cpts_mesh = numpy.zeros((3*n-3, 6*n))
        cpts_mesh[arange, imesh[0, :-1, :].flatten()] = 0.5 * 0.25
        cpts_mesh[arange, imesh[1, :-1, :].flatten()] = 0.5 * 0.75
        cpts_mesh[arange, imesh[0,  1:, :].flatten()] = 0.5 * 0.25
        cpts_mesh[arange, imesh[1,  1:, :].flatten()] = 0.5 * 0.75
        self.cpts_mesh = cpts_mesh

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
        self.rhs = numpy.zeros((size), dtype="complex")

    def _assemble_system(self, params):
        _assemble_AIC_mtx(self.mtx, params['def_mesh'], params['normals'],
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
        
        n = self.num_y
        
        jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['normals', 'def_mesh',
                                                    'b_pts', 'c_pts'],
                                         fd_states=[])
                                         
        jac['circulations', 'circulations'] = self.mtx.real

        normals = params['normals'].real
        alpha = params['alpha'].real * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'].real * numpy.array([cosa, 0., sina], dtype="complex")

        jac['circulations', 'v'][:, 0] = -self.rhs.real / params['v'].real

        dv_da = params['v'].real * numpy.array([-sina, 0., cosa]) * numpy.pi / 180.
        jac['circulations', 'alpha'][:, 0] = normals.dot(dv_da)

        return jac

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


class WeissingerLift(Component):
    """ Calculates total lift in force units """

    def __init__(self, n):
        super(WeissingerLift, self).__init__()

        self.add_param('cos_dih', val=numpy.zeros((n-1)))
        self.add_param('normals', val=numpy.zeros((n-1, 3)))
        self.add_param('sec_forces', val=numpy.zeros((n-1, 3)))
        self.add_output('L', val=0.)

#        self.fd_options['force_fd'] = True
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
        self.add_output('CL', val=0.)

        #self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        unknowns['CL'] = L / (0.5*rho*v**2*S_ref)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for lift."""

        jac = self.alloc_jacobian()
        
        S_ref = params['S_ref'].real
        rho = params['rho'].real
        v = params['v'].real
        L = params['L'].real

        jac['CL', 'S_ref'] = -L / (0.5*rho*v**2*S_ref**2)
        jac['CL', 'L'] = 1.0 / (0.5*rho*v**2*S_ref)
        jac['CL', 'v'] = -2 * L / (0.5*rho*v**3*S_ref)
        jac['CL', 'rho'] = -L / (0.5*rho**2*v**2*S_ref)
        
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
        self.add_output('CD', val=0.)

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

        normals = params['normals']
        for ind in xrange(num_y - 1):
            self.new_normals[ind] = normals[ind] - numpy.dot(normals[ind], trefftz_normal) \
                                    * trefftz_normal / norm(trefftz_normal)
        _assemble_AIC_mtx(self.mtx, params['def_mesh'], self.new_normals,
                          trefftz_points, params['b_pts'])

        self.velocities = -numpy.dot(self.mtx, params['circulations']) / params['v']
        unknowns['CD'] = 1. / params['S_ref'] / params['v'] * \
                         numpy.sum(params['circulations'] * self.velocities * params['widths'])

    def linearize(self, params, unknowns, resids):
        """ Jacobian for drag."""

        jac = self.complex_step_jacobian(params, unknowns, resids,\
                                         fd_params=['def_mesh', 'alpha', 'normals', 'b_pts'])

        circ = params['circulations'].real
        widths = params['widths'].real
        v = params['v'].real
        S_ref = params['S_ref'].real
        CD = unknowns['CD'].real
        velocities = self.velocities.real
        
        jac['CD', 'v'] = -2 * CD / v
        jac['CD', 'S_ref'] = -CD / S_ref
        jac['CD', 'circulations'][0, :] = 1. / S_ref / v * velocities * widths \
                                          - 1. / S_ref / v**2 * self.mtx.T.real.dot(circ * widths)
        jac['CD', 'widths'][0, :] = 1. / S_ref / v * velocities * circ
        
        return jac


class WeissingerGroup(Group):

    def __init__(self, num_y):
        super(WeissingerGroup, self).__init__()

        self.add('preproc',
                 WeissingerPreproc(num_y),
                 promotes=['*'])
        self.add('circ',
                 WeissingerCirculations(num_y),
                 promotes=['*'])
        self.add('forces',
                 WeissingerForces(num_y),
                 promotes=['*'])
        self.add('lift',
                 WeissingerLift(num_y),
                 promotes=['*'])
        self.add('CL',
                 WeissingerLiftCoeff(num_y),
                 promotes=['*'])
        self.add('CD',
                 WeissingerDragCoeff(num_y),
                 promotes=['*'])
