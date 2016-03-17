from __future__ import division
import sqlitedict
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

db = sqlitedict.SqliteDict('aerostruct.db', 'openmdao')

cl = []
cd = []
twist = []
mesh = []
# r = []
# t = []

for case_name, case_data in db.iteritems():
    if "metadata" in case_name or "derivs" in case_name:
        continue # don't plot these cases

    cl.append(case_data['Unknowns']['CL'])
    cd.append(case_data['Unknowns']['CD'])
    twist.append(case_data['Unknowns']['twist'])
    mesh.append(case_data['Unknowns']['mesh'])
    # r.append(case_data['Unknowns']['r'])
    # t.append(case_data['Unknowns']['t'])


# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
#
# ax1.plot(cl, lw=2)
# ax1.set_ylabel(r'$C_L$', rotation="horizontal", ha="right", fontsize=15)
#
# ax2.plot(cd, lw=2)
# ax2.set_xlabel('Iteration Number', fontsize=15)
# ax2.set_ylabel(r'$C_D$', rotation="horizontal", ha="right", fontsize=15)
#
# fig, ax = plt.subplots()
# n_cases = len(twist)
#
# for i,(t_vals, m_vals) in enumerate(zip(twist, mesh)):
#     # m_vals[0,:,1].shape, t_vals.shape
#     ax.plot(m_vals[0,:,1], t_vals, lw=2, c=plt.cm.jet(float(i)/n_cases))


mesh0 = mesh[0]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# xs = mesh0[0,:,0]
# ys = mesh0[0,:,1]
# zs = mesh0[0,:,2]
# ax.scatter(xs, ys, zs)
#
# plt.show()

def plot_wing(mesh):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.tight_layout()

    x = mesh[:, :, 0]
    y = mesh[:, :, 1]
    z = mesh[:, :, 2]
    ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k')
    max_dim = numpy.max(mesh0)
    ax.auto_scale_xyz([-max_dim/4, max_dim/4], [0, max_dim/2], [-max_dim/4, max_dim/4])
    ax.set_axis_off()

    n = mesh.shape[1]
    num_circ = 40
    r = numpy.ones(n) * 1
    t = numpy.arange(n) * .2
    fem_origin = 0.35
    chord = mesh[1, 0, 0] - mesh[0, 0, 0]

    p = numpy.linspace(0, 2*numpy.pi, num_circ)
    R, P = numpy.meshgrid(r, p)
    X, Z = R*numpy.cos(P), R*numpy.sin(P)
    X += fem_origin * chord
    Y = numpy.empty(X.shape)
    Y[:] = numpy.linspace(0, max_dim, n)
    colors = numpy.empty(X.shape)
    colors[:, :] = t
    colors = colors / numpy.max(colors)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(colors), linewidth=0, antialiased=False)

    plt.show()

plot_wing(mesh0)
