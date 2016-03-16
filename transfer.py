from __future__ import division
import numpy

from openmdao.api import Component


class TransferDisplacements(Component):
    """ Performs displacement transfer """

    def __init__(self, n, fem_origin=0.35):
        super(TransferDisplacements, self).__init__()

        self.fem_origin = fem_origin

        self.add_param('mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('disp', val=numpy.zeros((n, 6)))
        self.add_output('def_mesh', val=numpy.zeros((2, n, 3)))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):

        mesh = params['mesh']
        num_x, num_y = mesh.shape[:2]
        disp = params['disp']

        w = self.fem_origin
        ref_curve = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

        Smesh = numpy.zeros(mesh.shape, dtype="complex")
        for ind in xrange(num_x):
            Smesh[ind, :, :] = mesh[ind, :, :] - ref_curve

        def_mesh = numpy.zeros(mesh.shape, dtype="complex")
        cos, sin = numpy.cos, numpy.sin
        for ind in xrange(num_y):
            dx, dy, dz, rx, ry, rz = disp[ind, :]

            # 1 eye from the axis rotation matrices
            # -3 eye from subtracting Smesh three times
            T = -2 * numpy.eye(3, dtype="complex")
            T[ 1:,  1:] += [[cos(rx), -sin(rx)], [ sin(rx), cos(rx)]]
            T[::2, ::2] += [[cos(ry),  sin(ry)], [-sin(ry), cos(ry)]]
            T[ :2,  :2] += [[cos(rz), -sin(rz)], [ sin(rz), cos(rz)]]

            def_mesh[:, ind, :] += Smesh[:, ind, :].dot(T)
            def_mesh[:, ind, 0] += dx
            def_mesh[:, ind, 1] += dy
            def_mesh[:, ind, 2] += dz

        unknowns['def_mesh'] = def_mesh + mesh



class TransferLoads(Component):
    """ Performs load transfer """

    def __init__(self, n, fem_origin=0.35):
        super(TransferLoads, self).__init__()

        self.fem_origin = fem_origin

        self.add_param('def_mesh', val=numpy.zeros((2, n, 3)))
        self.add_param('sec_forces', val=numpy.zeros((n - 1, 3), dtype="complex"))
        self.add_output('loads', val=numpy.zeros((n, 6), dtype="complex"))

        self.fd_options['force_fd'] = True
        self.fd_options['form'] = "complex_step"
        self.fd_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['def_mesh']
        sec_forces = params['sec_forces']

        num_x, num_y = mesh.shape[:2]

        w = 0.25
        a_pts = 0.5 * (1-w) * mesh[:-1, :-1, :] + \
                0.5 *   w   * mesh[1:, :-1, :] + \
                0.5 * (1-w) * mesh[:-1,  1:, :] + \
                0.5 *   w   * mesh[1:,  1:, :]

        w = self.fem_origin
        s_pts = 0.5 * (1-w) * mesh[:-1, :-1, :] + \
                0.5 *   w   * mesh[1:, :-1, :] + \
                0.5 * (1-w) * mesh[:-1,  1:, :] + \
                0.5 *   w   * mesh[1:,  1:, :]

        moment = numpy.zeros((num_y - 1, 3), dtype="complex")
        for ind in xrange(num_y - 1):
            r = a_pts[0, ind, :] - s_pts[0, ind, :]
            F = sec_forces[ind, :]
            moment[ind, :] = numpy.cross(r, F)

        loads = numpy.zeros((num_y, 6), dtype="complex")
        loads[:-1, :3] += 0.5 * sec_forces[:, :]
        loads[ 1:, :3] += 0.5 * sec_forces[:, :]
        loads[:-1, 3:] += 0.5 * moment
        loads[ 1:, 3:] += 0.5 * moment

        unknowns['loads'] = loads
