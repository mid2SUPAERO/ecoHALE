""" Defines the transfer component to couple aero and struct analyses """

from __future__ import division
import numpy

from openmdao.api import Component


class TransferDisplacements(Component):
    """ Performs displacement transfer """

    def __init__(self, aero_ind, fem_origin=0.35):
        super(TransferDisplacements, self).__init__()

        n_surf = aero_ind.shape[0]
        tot_n = numpy.sum(aero_ind[:, 2])
        tot_bpts = numpy.sum(aero_ind[:, 3])
        tot_panels = numpy.sum(aero_ind[:, 4])
        self.aero_ind = aero_ind

        self.nx, self.ny = aero_ind[0, 0:2]
        self.n_wing = aero_ind[0, 2]

        self.fem_origin = fem_origin

        self.add_param('mesh', val=numpy.zeros((tot_n, 3), dtype="complex"))
        self.add_param('disp', val=numpy.zeros((self.ny, 6), dtype="complex"))
        self.add_output('def_mesh', val=numpy.zeros((tot_n, 3), dtype="complex"))

        self.deriv_options['type'] = 'cs'
        # self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):

        wing_mesh = params['mesh'][:self.n_wing, :].reshape(self.nx, self.ny, 3).astype("complex")
        mesh = wing_mesh.astype("complex")
        disp = params['disp']

        w = self.fem_origin
        ref_curve = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

        Smesh = numpy.zeros(mesh.shape, dtype="complex")
        for ind in xrange(self.nx):
            Smesh[ind, :, :] = mesh[ind, :, :] - ref_curve

        def_mesh = numpy.zeros(mesh.shape, dtype="complex")
        cos, sin = numpy.cos, numpy.sin
        for ind in xrange(self.ny):
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


        unknowns['def_mesh'] = params['mesh'].astype("complex")
        unknowns['def_mesh'][:self.n_wing, :] = \
            (def_mesh + mesh).reshape(self.n_wing, 3).astype("complex")


class TransferLoads(Component):
    """ Performs load transfer """

    def __init__(self, aero_ind, fem_origin=0.35):
        super(TransferLoads, self).__init__()

        n_surf = aero_ind.shape[0]
        tot_n = numpy.sum(aero_ind[:, 2])
        tot_bpts = numpy.sum(aero_ind[:, 3])
        tot_panels = numpy.sum(aero_ind[:, 4])
        self.aero_ind = aero_ind

        self.nx, self.ny = aero_ind[0, 0:2]
        self.n_wing = aero_ind[0, 2]
        self.n_panels = aero_ind[0, 4]

        self.fem_origin = fem_origin

        self.add_param('def_mesh', val=numpy.zeros((tot_n, 3)))
        self.add_param('sec_forces', val=numpy.zeros((tot_panels, 3), dtype="complex"))
        self.add_output('loads', val=numpy.zeros((self.ny, 6), dtype="complex"))

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

    def solve_nonlinear(self, params, unknowns, resids):
        wing_mesh = params['def_mesh'][:self.n_wing, :].reshape(self.nx, self.ny, 3)
        mesh = wing_mesh
        sec_forces = params['sec_forces'][:self.n_panels, :].reshape(self.nx-1, self.ny-1, 3)
        sec_forces = numpy.sum(sec_forces, axis=0)

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

        moment = numpy.zeros((self.ny - 1, 3), dtype="complex")
        for ind in xrange(self.ny - 1):
            r = a_pts[0, ind, :] - s_pts[0, ind, :]
            F = sec_forces[ind, :]
            moment[ind, :] = numpy.cross(r, F)

        loads = numpy.zeros((self.ny, 6), dtype="complex")
        loads[:-1, :3] += 0.5 * sec_forces[:, :]
        loads[ 1:, :3] += 0.5 * sec_forces[:, :]
        loads[:-1, 3:] += 0.5 * moment
        loads[ 1:, 3:] += 0.5 * moment

        unknowns['loads'] = loads
