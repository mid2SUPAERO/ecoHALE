from __future__ import division
import sqlitedict
import numpy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

"""
User-set options:
"""
db_name = 'aerostruct.db'
iteration = -1
show_wing = True
show_tube = True

db = sqlitedict.SqliteDict(db_name, 'openmdao')

cl = []
cd = []
twist = []
mesh = []
r = []
t = []

for case_name, case_data in db.iteritems():
    if "metadata" in case_name or "derivs" in case_name:
        continue # don't plot these cases

    cl.append(case_data['Unknowns']['CL'])
    cd.append(case_data['Unknowns']['CD'])
    twist.append(case_data['Unknowns']['twist'])
    mesh.append(case_data['Unknowns']['mesh'])
    try:
        r.append(case_data['Unknowns']['r'])
        t.append(case_data['Unknowns']['t'])
    except:
        pass

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

    chords = mesh[1, :, 0] - mesh[0, :, 0]

    if r != None and t != None:
        r = numpy.hstack((r, r[-1]))
        t = numpy.hstack((t, t[-1]))
        num_circ = 40
        fem_origin = 0.35
        p = numpy.linspace(0, 2*numpy.pi, num_circ)
        R, P = numpy.meshgrid(r, p)
        X, Z = R*numpy.cos(P), R*numpy.sin(P)
        X[:] += fem_origin * chords
        Y = numpy.empty(X.shape)
        Y[:] = numpy.linspace(0, max_dim, n)
        colors = numpy.empty(X.shape)
        colors[:, :] = t
        colors = colors / numpy.max(colors)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.YlOrRd(colors), linewidth=0, antialiased=False)
    ax.auto_scale_xyz([-max_dim/2, max_dim/2], [0, max_dim], [-max_dim/2, max_dim/2])
    plt.show()

mesh0 = mesh[iteration]

if show_wing and not show_tube:
    plot_wing(mesh0)
if show_tube and not show_wing:
    r0 = r[iteration]
    t0 = t[iteration]
    plot_wing(mesh0, r, t, tube_only=True)
if show_tube and  show_wing:
    r0 = r[iteration]
    t0 = t[iteration]
    plot_wing(mesh0, r, t)
