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
    cd.append(case_data['Unknowns']['CD_i'])
    twist.append(case_data['Unknowns']['twist'])
    mesh.append(case_data['Unknowns']['mesh'])
    # r.append(case_data['Unknowns']['r'])
    # t.append(case_data['Unknowns']['t'])

mesh0 = mesh[-1]

def plot_wing(mesh, r=None, t=None, tube_only=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.tight_layout()
    ax.set_axis_off()

    x = mesh[:, :, 0]
    y = mesh[:, :, 1]
    z = mesh[:, :, 2]
    if not tube_only:
        ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k')
    max_dim = numpy.max(mesh)

    chord = mesh[1, :, 0] - mesh[0, :, 0]

    if r != None and t != None:
        num_circ = 40
        fem_origin = 0.35
        p = numpy.linspace(0, 2*numpy.pi, num_circ)
        R, P = numpy.meshgrid(r, p)
        X, Z = R*numpy.cos(P), R*numpy.sin(P)
        X[:] += fem_origin * chord
        Y = numpy.empty(X.shape)
        Y[:] = numpy.linspace(0, max_dim, n)
        colors = numpy.empty(X.shape)
        colors[:, :] = t
        colors = colors / numpy.max(colors)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.YlOrRd(colors), linewidth=0, antialiased=False)
    ax.auto_scale_xyz([-max_dim/2, max_dim/2], [0, max_dim], [-max_dim/2, max_dim/2])
    plt.show()

n = mesh0.shape[1]

r = numpy.ones(n) * .5
t = numpy.arange(n) * .2

plot_wing(mesh0)
plot_wing(mesh0, r, t, tube_only=True)
plot_wing(mesh0, r, t)
