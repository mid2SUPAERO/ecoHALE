""" Defines the aerodynamic analysis component using Weissinger's lifting line theory """

from __future__ import division
import numpy

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve
try:
    import lib
    fortran_flag = True
except:
    fortran_flag = False
# fortran_flag = False


def view_mat(mat):
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    print "Cond #:", numpy.linalg.cond(mat)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

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
        v = -C / rH * (cosA + 1)
    else:
        v = -C / rH * (cosA + cosB)

    if rev:
        v = -v

    return v

def _calc_vorticity(A, B, P):
    r1 = P - A
    r2 = P - B

    r1_mag = norm(r1)
    r2_mag = norm(r2)

    return (r1_mag + r2_mag) * numpy.cross(r1, r2) / \
         (r1_mag * r2_mag * (r1_mag * r2_mag + r1.dot(r2)))


def _assemble_AIC_mtx(mtx, full_mesh, aero_ind, points, b_pts, alpha, skip=False):
    """
    Compute the aerodynamic influence coefficient matrix
    either for the ation linear system or Trefftz-plane drag computation
    - mtx[num_y-1, num_y-1, 3] : derivative of v w.r.t. circulation
    - mesh[2, num_y, 3] : contains LE and TE coordinates at each section
    - points[num_y-1, 3] : control points
    - b_pts[num_y, 3] : bound vortex coordinates
    """

    mtx[:, :, :] = 0.0
    cosa = numpy.cos(alpha * numpy.pi / 180.)
    sina = numpy.sin(alpha * numpy.pi / 180.)
    u = numpy.array([cosa, 0, sina])

    for i_surf, row in enumerate(aero_ind):
        nx_, ny_, n_, n_bpts_, n_panels_, i_, i_bpts_, i_panels_ = row.copy()
        n = nx_ * ny_
        mesh = full_mesh[i_:i_+n_, :].reshape(nx_, ny_, 3)
        bpts = b_pts[i_bpts_:i_bpts_+n_bpts_].reshape(nx_-1, ny_, 3)

        for i_points, row in enumerate(aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            pts = points[i_panels:i_panels+n_panels].reshape(nx-1, ny-1, 3)

            small_mat = numpy.zeros((n_panels, n_panels_, 3)).astype("complex")

            if fortran_flag:
                small_mat[:, :, :] = lib.assembleaeromtx_hug_planform(ny, nx, ny_, nx_, alpha, pts, bpts, mesh, skip)
            else:
                # Spanwise loop through horseshoe elements
                for el_j in xrange(ny_ - 1):
                    el_loc_j = el_j * (nx_ - 1)
                    C_te = mesh[-1, el_j + 1, :]
                    D_te = mesh[-1, el_j + 0, :]

                    # Spanwise loop through control points
                    for cp_j in xrange(ny - 1):
                        cp_loc_j = cp_j * (nx - 1)

                        # Chordwise loop through control points
                        for cp_i in xrange(nx - 1):
                            cp_loc = cp_i + cp_loc_j

                            P = pts[cp_i, cp_j]

                            r1 = P - D_te
                            r2 = P - C_te

                            r1_mag = norm(r1)
                            r2_mag = norm(r2)

                            t1 = numpy.cross(u, r2) / (r2_mag * (r2_mag - u.dot(r2)))
                            t3 = numpy.cross(u, r1) / (r1_mag * (r1_mag - u.dot(r1)))

                            trailing = t1 - t3
                            edges = 0

                            # Chordwise loop through horseshoe elements
                            for el_i in reversed(xrange(nx_ - 1)):
                                el_loc = el_i + el_loc_j

                                A = bpts[el_i, el_j + 0, :]
                                B = bpts[el_i, el_j + 1, :]

                                if el_i == nx_ - 2:
                                    C = mesh[-1, el_j + 1, :]
                                    D = mesh[-1, el_j + 0, :]
                                else:
                                    C = bpts[el_i + 1, el_j + 1, :]
                                    D = bpts[el_i + 1, el_j + 0, :]
                                edges += _calc_vorticity(B, C, P)
                                edges += _calc_vorticity(D, A, P)

                                if skip and el_loc == cp_loc:
                                    small_mat[cp_loc, el_loc, :] = trailing + edges
                                else:
                                    bound = _calc_vorticity(A, B, P)
                                    small_mat[cp_loc, el_loc, :] = trailing + edges + bound

            mtx[i_panels:i_panels+n_panels, i_panels_:i_panels_+n_panels_, :] = small_mat

    mtx /= 4 * numpy.pi


class WeissingerGeometry(Component):
    """ Compute various geometric properties for Weissinger analysis """

    def __init__(self, aero_ind):
        super(WeissingerGeometry, self).__init__()

        n_surf = aero_ind.shape[0]
        tot_n = numpy.sum(aero_ind[:, 2])
        tot_bpts = numpy.sum(aero_ind[:, 3])
        tot_panels = numpy.sum(aero_ind[:, 4])
        self.aero_ind = aero_ind

        self.add_param('def_mesh', val=numpy.zeros((tot_n, 3), dtype="complex"))
        self.add_output('b_pts', val=numpy.zeros((tot_bpts, 3), dtype="complex"))
        self.add_output('mid_b', val=numpy.zeros((tot_panels, 3), dtype="complex"))
        self.add_output('c_pts', val=numpy.zeros((tot_panels, 3)))
        self.add_output('widths', val=numpy.zeros((tot_panels)))
        self.add_output('normals', val=numpy.zeros((tot_panels, 3)))
        self.add_output('S_ref', val=numpy.zeros((n_surf)))

    def _get_lengths(self, A, B, axis):
        return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

    def solve_nonlinear(self, params, unknowns, resids):
        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            mesh = params['def_mesh'][i:i+n, :].reshape(nx, ny, 3)

            b_pts = mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25

            mid_b = (b_pts[:, 1:, :] + b_pts[:, :-1, :]) / 2

            c_pts = 0.5 * 0.25 * mesh[:-1, :-1, :] + \
                    0.5 * 0.75 * mesh[1:, :-1, :] + \
                    0.5 * 0.25 * mesh[:-1,  1:, :] + \
                    0.5 * 0.75 * mesh[1:,  1:, :]

            widths = self._get_lengths(b_pts[:, 1:, :], b_pts[:, :-1, :], 2)

            normals = numpy.cross(
                mesh[:-1,  1:, :] - mesh[ 1:, :-1, :],
                mesh[:-1, :-1, :] - mesh[ 1:,  1:, :],
                axis=2)

            norms = numpy.sqrt(numpy.sum(normals**2, axis=2))

            for j in xrange(3):
                normals[:, :, j] /= norms

            unknowns['b_pts'][i_bpts:i_bpts+n_bpts, :] = b_pts.reshape(-1, b_pts.shape[-1])
            unknowns['mid_b'][i_panels:i_panels+n_panels, :] = mid_b.reshape(-1, mid_b.shape[-1])
            unknowns['c_pts'][i_panels:i_panels+n_panels, :] = c_pts.reshape(-1, c_pts.shape[-1])
            unknowns['widths'][i_panels:i_panels+n_panels] = widths.flatten()
            unknowns['normals'][i_panels:i_panels+n_panels, :] = normals.reshape(-1, normals.shape[-1], order='F')
            unknowns['S_ref'][i_surf] = 0.5 * numpy.sum(norms)

    def linearize(self, params, unknowns, resids):
        """ Jacobian for geometry."""

        jac = self.alloc_jacobian()

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['def_mesh'],
                                         fd_unknowns=['widths', 'normals', 'S_ref'],
                                         fd_states=[])

        jac.update(fd_jac)

        i_ny = 0.
        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            for iz, v in zip(((i_bpts+i_ny)*3, (i_bpts+i_ny+ny)*3), (.75, .25)):
                numpy.fill_diagonal(jac['b_pts', 'def_mesh'][i_bpts*3:(n_bpts+i_bpts)*3, iz:], v)

            for iz, v in zip((i*3, (i+1)*3, (i+ny)*3, (ny+i+1)*3), (.125, .125, .375, .375)):
                for ix in range(nx-1):
                    numpy.fill_diagonal(jac['c_pts', 'def_mesh'][(i_panels+ix*(ny-1))*3:(i_panels+(ix+1)*(ny-1))*3, iz+ix*ny*3:], v)

            for iz, v in zip((i*3, (i+1)*3, (i+ny)*3, (ny+i+1)*3), (.375, .375, .125, .125)):
                for ix in range(nx-1):
                    numpy.fill_diagonal(jac['mid_b', 'def_mesh'][(i_panels+ix*(ny-1))*3:(i_panels+(ix+1)*(ny-1))*3, iz+ix*ny*3:], v)

            i_ny += ny

        return jac


class WeissingerCirculations(Component):
    """ Define circulations """

    def __init__(self, aero_ind):
        super(WeissingerCirculations, self).__init__()

        tot_n = numpy.sum(aero_ind[:, 2])
        tot_bpts = numpy.sum(aero_ind[:, 3])
        tot_panels = numpy.sum(aero_ind[:, 4])
        self.aero_ind = aero_ind

        self.add_param('def_mesh', val=numpy.zeros((tot_n, 3), dtype="complex"))
        self.add_param('b_pts', val=numpy.zeros((tot_bpts, 3), dtype="complex"))
        self.add_param('c_pts', val=numpy.zeros((tot_panels, 3), dtype="complex"))
        self.add_param('normals', val=numpy.zeros((tot_panels, 3)))
        self.add_param('v', val=10.)
        self.add_param('alpha', val=3.)
        self.add_state('circulations', val=numpy.zeros((tot_panels), dtype="complex"))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

        self.deriv_options['linearize'] = True # only for circulations

        self.AIC_mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")
        self.mtx = numpy.zeros((tot_panels, tot_panels), dtype="complex")
        self.rhs = numpy.zeros((tot_panels), dtype="complex")

    def _assemble_system(self, params):
        _assemble_AIC_mtx(self.AIC_mtx, params['def_mesh'], self.aero_ind,
                        params['c_pts'], params['b_pts'], params['alpha'])

        self.mtx[:, :] = 0.
        for ind in xrange(3):
            self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T * params['normals'][:, ind]).T

        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'] * numpy.array([cosa, 0., sina], dtype="complex")
        self.rhs[:] = -params['normals'].reshape(-1, params['normals'].shape[-1], order='F').dot(v_inf)

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

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['normals', 'alpha',\
                                                    'def_mesh', 'b_pts', 'c_pts'],
                                         fd_states=[])
        jac.update(fd_jac)

        jac['circulations', 'circulations'] = self.mtx.real

        jac['circulations', 'v'][:, 0] = -self.rhs.real / params['v'].real

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
    """ Define aerodynamic forces acting on each section """

    def __init__(self, aero_ind):
        super(WeissingerForces, self).__init__()

        n_surf = aero_ind.shape[0]
        tot_n = numpy.sum(aero_ind[:, 2])
        tot_bpts = numpy.sum(aero_ind[:, 3])
        tot_panels = numpy.sum(aero_ind[:, 4])
        self.aero_ind = aero_ind

        self.add_param('def_mesh', val=numpy.zeros((tot_n, 3)))
        self.add_param('b_pts', val=numpy.zeros((tot_bpts, 3)))
        self.add_param('mid_b', val=numpy.zeros((tot_panels, 3)))

        self.aero_ind = aero_ind

        self.add_param('circulations', val=numpy.zeros((tot_panels)))
        self.add_param('alpha', val=3.)
        self.add_param('v', val=10.)
        self.add_param('rho', val=3.)
        self.add_output('sec_forces', val=numpy.zeros((tot_panels, 3)))

        self.mtx = numpy.zeros((tot_panels, tot_panels, 3), dtype="complex")

        self.v = numpy.zeros((tot_panels, 3), dtype="complex")


    def solve_nonlinear(self, params, unknowns, resids):
        circ = params['circulations']

        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        _assemble_AIC_mtx(self.mtx, params['def_mesh'], self.aero_ind,
                          params['mid_b'], params['b_pts'], params['alpha'], skip=True)

        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            for ind in xrange(3):
                self.v[:, ind] = self.mtx[:, :, ind].dot(circ)
            self.v[:, 0] += cosa * params['v']
            self.v[:, 2] += sina * params['v']

            b_pts = params['b_pts'][i_bpts:i_bpts+n_bpts, :].reshape(nx-1, ny, 3)

            bound = b_pts[:, 1:, :] - b_pts[:, :-1, :]

            cross = numpy.cross(self.v[i_panels:i_panels+n_panels], bound.reshape(-1, bound.shape[-1], order='F'))

            for ind in xrange(3):
                unknowns['sec_forces'][i_panels:i_panels+n_panels, ind] = (params['rho'] * circ[i_panels:i_panels+n_panels] * cross[:, ind])

    def linearize(self, params, unknowns, resids):
        """ Jacobian for forces."""

        jac = self.alloc_jacobian()

        fd_jac = self.complex_step_jacobian(params, unknowns, resids,
                                         fd_params=['b_pts', 'alpha',
                                                    'circulations', 'v', 'mid_b', 'def_mesh'],
                                         fd_states=[])
        jac.update(fd_jac)

        rho = params['rho'].real
        sec_forces = unknowns['sec_forces'].real

        jac['sec_forces', 'rho'] = sec_forces.flatten() / rho

        return jac


class WeissingerLiftDrag(Component):
    """ Calculate total lift and drag in force units based on section forces """

    def __init__(self, aero_ind):
        super(WeissingerLiftDrag, self).__init__()

        n_surf = aero_ind.shape[0]
        tot_n = numpy.sum(aero_ind[:, 2])
        tot_bpts = numpy.sum(aero_ind[:, 3])
        tot_panels = numpy.sum(aero_ind[:, 4])
        self.aero_ind = aero_ind

        self.add_param('sec_forces', val=numpy.zeros((tot_panels, 3)))
        self.add_param('alpha', val=3.)
        self.add_output('L', val=numpy.zeros((n_surf)))
        self.add_output('D', val=numpy.zeros((n_surf)))

    def solve_nonlinear(self, params, unknowns, resids):
        alpha = params['alpha'] * numpy.pi / 180.
        forces = params['sec_forces']
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            unknowns['L'][i_surf] = numpy.sum(-forces[i_panels:i_panels+n_panels, 0] * sina + forces[i_panels:i_panels+n_panels, 2] * cosa)
            unknowns['D'][i_surf] = numpy.sum( forces[i_panels:i_panels+n_panels, 0] * cosa + forces[i_panels:i_panels+n_panels, 2] * sina)


    def linearize(self, params, unknowns, resids):
        """ Jacobian for forces."""

        jac = self.alloc_jacobian()

        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)

        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            forces = params['sec_forces'][i_panels:n_panels+i_panels].reshape(nx-1, ny-1, 3)

            tmp = numpy.array([-sina, 0, cosa])
            jac['L', 'sec_forces'][i_surf, i_panels*3:i_panels*3+n_panels*3] = numpy.atleast_2d(numpy.tile(tmp, n_panels))
            tmp = numpy.array([cosa, 0, sina])
            jac['D', 'sec_forces'][i_surf, i_panels*3:i_panels*3+n_panels*3] = numpy.atleast_2d(numpy.tile(tmp, n_panels))

            jac['L', 'alpha'][i_surf] = numpy.pi / 180. * numpy.sum(-forces[:, :, 0] * cosa - forces[:, :, 2] * sina)
            jac['D', 'alpha'][i_surf] = numpy.pi / 180. * numpy.sum(-forces[:, :, 0] * sina + forces[:, :, 2] * cosa)

        return jac

class WeissingerCoeffs(Component):
    """ Compute lift and drag coefficients """

    def __init__(self, aero_ind):
        super(WeissingerCoeffs, self).__init__()

        n_surf = aero_ind.shape[0]
        self.aero_ind = aero_ind
        self.add_param('S_ref', val=numpy.zeros((n_surf)))
        self.add_param('L', val=numpy.zeros((n_surf)))
        self.add_param('D', val=numpy.zeros((n_surf)))
        self.add_param('v', val=0.)
        self.add_param('rho', val=0.)
        self.add_output('CL1', val=numpy.zeros((n_surf)))
        self.add_output('CDi', val=numpy.zeros((n_surf)))

    def solve_nonlinear(self, params, unknowns, resids):
        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        D = params['D']
        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            unknowns['CL1'][i_surf] = L[i_surf] / (0.5*rho*v**2*S_ref[i_surf])
            unknowns['CDi'][i_surf] = D[i_surf] / (0.5*rho*v**2*S_ref[i_surf])

    def linearize(self, params, unknowns, resids):
        """ Jacobian for forces."""

        S_ref = params['S_ref']
        rho = params['rho']
        v = params['v']
        L = params['L']
        D = params['D']

        jac = self.alloc_jacobian()

        for i_surf, row in enumerate(self.aero_ind):
            nx, ny, n, n_bpts, n_panels, i, i_bpts, i_panels = row

            jac['CL1', 'D'][i_surf] = 0.
            jac['CDi', 'L'][i_surf] = 0.

            tmp = 0.5*rho*v**2*S_ref
            jac['CL1', 'L'][i_surf, i_surf] = 1. / tmp[i_surf]
            jac['CDi', 'D'][i_surf, i_surf] = 1. / tmp[i_surf]

            tmp = -0.5*rho**2*v**2*S_ref
            jac['CL1', 'rho'][i_surf] = L[i_surf] / tmp[i_surf]
            jac['CDi', 'rho'][i_surf] = D[i_surf] / tmp[i_surf]

            tmp = -0.5*rho*v**2*S_ref**2
            jac['CL1', 'S_ref'][i_surf, i_surf] = L[i_surf] / tmp[i_surf]
            jac['CDi', 'S_ref'][i_surf, i_surf] = D[i_surf] / tmp[i_surf]

            tmp = -0.25*rho*v**3*S_ref
            jac['CL1', 'v'][i_surf] = L[i_surf] / tmp[i_surf]
            jac['CDi', 'v'][i_surf] = D[i_surf] / tmp[i_surf]

        return jac


class TotalLift(Component):
    """ Calculate total lift in force units """

    def __init__(self, CL0, aero_ind):
        super(TotalLift, self).__init__()

        self.n_surf = aero_ind.shape[0]
        self.add_param('CL1', val=numpy.zeros((self.n_surf)))
        self.add_output('CL', val=numpy.zeros((self.n_surf)))
        self.add_output('CL_wing', val=0.)

        self.CL0 = CL0

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['CL'] = params['CL1'] + self.CL0
        unknowns['CL_wing'] = params['CL1'][0]

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['CL', 'CL1'][:] = numpy.eye(self.n_surf)
        jac['CL_wing', 'CL1'][0][0] = 1
        return jac

class TotalDrag(Component):
    """ Calculate total drag in force units """

    def __init__(self, CD0, aero_ind):
        super(TotalDrag, self).__init__()

        self.n_surf = aero_ind.shape[0]
        self.add_param('CDi', val=numpy.zeros((self.n_surf)))
        self.add_output('CD', val=numpy.zeros((self.n_surf)))
        self.add_output('CD_wing', val=0.)

        self.CD0 = CD0

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['CD'] = params['CDi'] + self.CD0
        unknowns['CD_wing'] = unknowns['CD'][0]

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['CD', 'CDi'][:] = numpy.eye(self.n_surf)
        jac['CD_wing', 'CDi'][0][0] = 1
        return jac

class WeissingerStates(Group):
    """ Group that contains the aerodynamic states """

    def __init__(self, aero_ind):
        super(WeissingerStates, self).__init__()

        self.add('wgeom',
                 WeissingerGeometry(aero_ind),
                 promotes=['*'])
        self.add('circ',
                 WeissingerCirculations(aero_ind),
                 promotes=['*'])
        self.add('forces',
                 WeissingerForces(aero_ind),
                 promotes=['*'])



class WeissingerFunctionals(Group):
    """ Group that contains the aerodynamic functionals used to evaluate performance """

    def __init__(self, aero_ind, CL0, CD0):
        super(WeissingerFunctionals, self).__init__()

        self.add('liftdrag',
                 WeissingerLiftDrag(aero_ind),
                 promotes=['*'])
        self.add('coeffs',
                 WeissingerCoeffs(aero_ind),
                 promotes=['*'])
        self.add('CL',
                 TotalLift(CL0, aero_ind),
                 promotes=['*'])
        self.add('CD',
                 TotalDrag(CD0, aero_ind),
                 promotes=['*'])
