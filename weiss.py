from __future__ import division
import numpy
import pylab
from mpl_toolkits.mplot3d import Axes3D

class VLM(object):

    def __init__(self):
        self.alpha = 3.
        self.mesh = None
        self.mtx = None
        self.rho = 1.225
        self.v = 200.
        self.trefftz_dist = 10000.
        self.CL = 0
        self.CD = 0

    def assemble(self):

        def get_lengths(A, B, axis):
            return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

        def biot_savart(N, A, B, P, infinite=False, reverse=False, eps=1e-5):
            rPA = numpy.linalg.norm(A - P)
            rPB = numpy.linalg.norm(B - P)
            rAB = numpy.linalg.norm(B - A)
            rH = numpy.linalg.norm(P - A - numpy.dot((B - A), (P - A)) / numpy.dot((B - A), (B - A)) * (B - A)) + eps
            cosA = numpy.dot((P - A), (B - A)) / (rPA * rAB)
            cosB = numpy.dot((P - B), (A - B)) / (rPB * rAB)
            C = numpy.cross(B - P, A - P)
            C /= numpy.linalg.norm(C)

            if infinite:
                vdn = -numpy.dot(N, C) / rH * (cosA + 1) / (4 * numpy.pi)
            else:
                vdn = -numpy.dot(N, C) / rH * (cosA + cosB) / (4 * numpy.pi)

            if reverse:
                vdn = -vdn

            return vdn

        def calc_mtx_coeffs(mtx, num_y, normals, points, tref=False, rhs=None, v_inf=None):
            # loop through intersection points and calculate coeffs for mtx
            for ind_iy in xrange(num_y - 1):
                if tref:
                    # project N onto the trefftz plane
                    N = normals[-1, ind_iy] - numpy.dot(normals[-1, ind_iy], trefftz_plane) * trefftz_plane / numpy.linalg.norm(trefftz_plane)
                    P = points[ind_iy, :]
                else:
                    N = normals[ind_ix, ind_iy]

                    rhs[ind_iy] = -numpy.dot(N, v_inf)
                    P = points[ind_ix, ind_iy, :]

                for ind_jy in xrange(num_y - 1):
                    A = b_pts[ind_jy + 0, :]
                    B = b_pts[ind_jy + 1, :]
                    D = mesh[-1, ind_jy + 0, :]
                    E = mesh[-1, ind_jy + 1, :]

                    if not tref:
                        mtx[ind_iy, ind_jy] += biot_savart(N, A, B, P)
                    mtx[ind_iy, ind_jy] += biot_savart(N, B, E, P, infinite=True, reverse=False)
                    mtx[ind_iy, ind_jy] += biot_savart(N, A, D, P, infinite=True, reverse=True)
            return mtx

        alpha = self.alpha * numpy.pi / 180.
        v_inf = self.v * numpy.array([numpy.cos(alpha), 0, numpy.sin(alpha)])
        mesh = self.mesh
        num_x, num_y = mesh.shape[:2]
        chords = get_lengths(mesh[0, :, :], mesh[-1, :, :], 1)

        b_pts = mesh[0, :, :] * .75 + mesh[1, :, :] * .25
        c_pts = 0.5 * (mesh[:-1, :-1, :] * .25 + mesh[1:, :-1, :] * .75) + \
                0.5 * (mesh[:-1,  1:, :] * .25 + mesh[1:,  1:, :] * .75)
        widths = get_lengths(b_pts[1:, :], b_pts[:-1, :], 1)
        cos_dih = (b_pts[1:, 1] - b_pts[:-1, 1]) / widths
        normals = numpy.cross(
            mesh[:-1,  1:, :] - mesh[ 1:, :-1, :],
            mesh[:-1, :-1, :] - mesh[ 1:,  1:, :],
            axis=2)
        norms = numpy.sqrt(numpy.sum(normals**2, axis=2))
        for ind in xrange(3):
            normals[:, :, ind] /= norms
        S_ref = 0.5 * numpy.sum(norms)

        size = (num_x - 1) * (num_y - 1)
        mtx = numpy.zeros((size, size))
        rhs = numpy.zeros(size)

        # TODO: GENERALIZE FOR 2-D VLM, NEED TO ADD IND_JX FOR LOOP and UPDATE INDICES FOR RHS AND MTX

        for ind_ix in xrange(num_x - 1):
            mtx = calc_mtx_coeffs(mtx, num_y, normals, c_pts, tref=False, rhs=rhs, v_inf=v_inf)

        sol = numpy.linalg.solve(mtx, rhs)

        sec_forces = numpy.array(normals)
        for ind in xrange(3):
            sec_forces[:, :, ind] *= self.rho * self.v * sol.reshape((num_x - 1, num_y - 1)) * widths
        self.sec_forces = sec_forces

        sec_lift = self.rho * self.v * sol.reshape((num_x - 1, num_y - 1)) * widths * cos_dih
        L = numpy.sum(sec_lift)
        self.L = L
        print "lift", self.L
        CL = L / (0.5*self.rho*self.v**2*S_ref)

        CL = 2. / S_ref / self.v * numpy.sum(sol.reshape((num_x - 1, num_y - 1)) * widths * cos_dih)
        self.CL = CL

        # TODO: DOUBLE-CHECK TREFFTZ PLANE DRAG GEOMETRY AND CALCULATIONS
        #  drag calculations
        trefftz_plane = numpy.array([numpy.cos(alpha), 0, numpy.sin(alpha)])
        trefftz_dist = self.trefftz_dist
        a = [self.trefftz_dist, 0, 0]

        mtx = numpy.zeros((size, size))

        # find intersection points on the trefftz plane
        intersections = numpy.zeros((num_y - 1, 2, 3))
        for ind_jy in xrange(num_y - 1):
            A = mesh[0, ind_jy + 0, :]
            B = mesh[0, ind_jy + 1, :]
            D = mesh[-1, ind_jy + 0, :]
            E = mesh[-1, ind_jy + 1, :]

            t = -numpy.dot(trefftz_plane, (A - a)) / numpy.dot(trefftz_plane, D - A)
            intersections[ind_jy, 0, :] = A + (D - A) * t

            t = -numpy.dot(trefftz_plane, (B - a)) / numpy.dot(trefftz_plane, E - B)
            intersections[ind_jy, 1, :] = B + (E - B) * t

        trefftz_points = (intersections[:, 1, :] + intersections[:, 0, :]) / 2.

        mtx = calc_mtx_coeffs(mtx, num_y, normals, trefftz_points, tref=True)

        vels = -numpy.dot(mtx, sol) / self.v
        CD = 1. / S_ref / self.v * numpy.sum(sol * vels * widths)
        self.CD = CD
        # exit()

        # pylab.plot(numpy.linspace(0, 1, sol.shape[0]), sol)
        # pylab.show()

def plot_wing(span, mesh):
    fig = pylab.figure()
    ax = Axes3D(fig)

    flat_mesh = mesh.reshape(num_y * 2, 3)
    ax.scatter(flat_mesh[:, 0], flat_mesh[:, 1], flat_mesh[:, 2])
    ax.auto_scale_xyz([0, span], [0, span], [0, span])
    pylab.show()

if __name__ == '__main__':
    # verification case from 1997 NASA LaRC paper
    num_y = 21
    span = 232.02
    chord = 39.37

    mesh = numpy.zeros((2, num_y,3))
    for ind_x in xrange(2):
        for ind_y in xrange(num_y):
            mesh[ind_x, ind_y, :] = [ind_x * chord, ind_y / (num_y-1) * span, 0]

    # plot_wing(span, mesh)

    n = 20
    CLs = numpy.zeros((n))
    CDs = numpy.zeros((n))
    alphas = numpy.linspace(-3, 15, n)
    for i, alpha in enumerate(alphas):
        v = VLM()
        v.mesh = mesh
        v.alpha = alpha
        v.assemble()
        CLs[i] = v.CL
        CDs[i] = v.CD + .009364 # extra factor added from paper
    pylab.plot(alphas, CLs)
    pylab.xlabel('$ \\alpha $', fontsize=24)
    pylab.ylabel('$C_L$', fontsize=24)
    pylab.show()

    pylab.plot(CDs, CLs)
    pylab.xlabel('$C_D$', fontsize=24)
    pylab.ylabel('$C_L$', fontsize=24)
    pylab.show()
