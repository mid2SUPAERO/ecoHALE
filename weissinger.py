from __future__ import division
import numpy

from openmdao.api import Component
from scipy.linalg import lu_factor



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

    rPA = numpy.linalg.norm(A - P)
    rPB = numpy.linalg.norm(B - P)
    rAB = numpy.linalg.norm(B - A)
    rH = numpy.linalg.norm(P - A - numpy.dot((B - A), (P - A)) / numpy.dot((B - A), (B - A)) * (B - A)) + eps
    cosA = numpy.dot((P - A), (B - A)) / (rPA * rAB)
    cosB = numpy.dot((P - B), (A - B)) / (rPB * rAB)
    C = numpy.cross(B - P, A - P)
    C /= numpy.linalg.norm(C)

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

    def solve_nonlinear(self, params, unknowns, resids):
        _assemble_AIC_mtx(self.mtx, params['mesh'], params['normals'],
                          params['c_pts'], params['b_pts'])
        
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'] * numpy.array([cosa, 0., sina], dtype="complex") 
        self.rhs[:] = -params['normals'].dot(v_inf)
        
        unknowns['circulations'] = numpy.linalg.solve(self.mtx, self.rhs)

    def apply_nonlinear(self, params, unknowns, resids):
        _assemble_AIC_mtx(self.mtx, params['mesh'], params['normals'],
                          params['c_pts'], params['b_pts'])
        
        alpha = params['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(alpha)
        sina = numpy.sin(alpha)
        v_inf = params['v'] * numpy.array([cosa, 0., sina], dtype="complex") 
        self.rhs[:] = -params['normals'].dot(v_inf)

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
