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
iteration = 0
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

def plot_wing(mesh, iteration):
    az = ax.azim
    el = ax.elev
    dist = ax.dist

    mesh0 = mesh[iteration]

    plt.tight_layout()
    ax.set_axis_off()

    x = mesh0[:, :, 0]
    y = mesh0[:, :, 1]
    z = mesh0[:, :, 2]
    if show_wing:
        ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='k')

    if show_tube:
        r0 = r[iteration]
        t0 = t[iteration]
        r0 = numpy.hstack((r0, r0[-1]))
        t0 = numpy.hstack((t0, t0[-1]))
        n = mesh0.shape[1]
        num_circ = 40
        fem_origin = 0.35
        p = numpy.linspace(0, 2*numpy.pi, num_circ)
        R, P = numpy.meshgrid(r0, p)
        X, Z = R*numpy.cos(P), R*numpy.sin(P)
        chords = mesh0[1, :, 0] - mesh0[0, :, 0]
        X[:] += fem_origin * chords
        Y = numpy.empty(X.shape)
        Y[:] = numpy.linspace(0, mesh0[0, -1, 1], n)
        colors = numpy.empty(X.shape)
        colors[:, :] = t0
        colors = colors / numpy.max(colors)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.YlOrRd(colors), linewidth=0, antialiased=False)
    max_dim = numpy.max(numpy.max(mesh0))
    ax.auto_scale_xyz([-max_dim/4, max_dim/4], [max_dim/4, 3*max_dim/4], [-max_dim/4, max_dim/4])
    ax.set_title("Iteration: {}".format(iteration))

    ax.view_init(elev=el, azim=az) #Reproduce view


curr_pos = iteration

def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(mesh)

    ax.cla()
    plot_wing(mesh, curr_pos)
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.gca(projection='3d')
# max_dim = numpy.max(numpy.max(mesh[0]))
# # ax.auto_scale_xyz([-max_dim/2, max_dim/2], [0, max_dim], [-max_dim/2, max_dim/2])
# ax.set_xlim([-max_dim/2, max_dim/2])
# ax.set_ylim([0, max_dim])
# ax.set_zlim([-max_dim/2, max_dim/2])


iteration = iteration % len(mesh)
plot_wing(mesh, iteration)
plt.show()
