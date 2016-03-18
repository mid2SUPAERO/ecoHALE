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
start_iteration = 0
show_wing = True
show_tube = True

db = sqlitedict.SqliteDict(db_name, 'openmdao')

def _get_lengths(self, A, B, axis):
    return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

cl = []
cd = []
twist = []
mesh = []
r = []
t = []
sec_forces = []
normals = []
cos_dih = []
lift = []
vonmises = []

for case_name, case_data in db.iteritems():
    if "metadata" in case_name or "derivs" in case_name:
        continue # don't plot these cases

    mesh.append(case_data['Unknowns']['mesh'])
    try:
        r.append(case_data['Unknowns']['r'])
        t.append(case_data['Unknowns']['t'])
        vonmises.append(case_data['Unknowns']['vonmises'])
    except:
        pass
    try:
        cl.append(case_data['Unknowns']['CL'])
        cd.append(case_data['Unknowns']['CD'])
        twist.append(case_data['Unknowns']['twist'])
        normals.append(case_data['Unknowns']['normals'])
        cos_dih.append(case_data['Unknowns']['cos_dih'])
        sec_forces.append(case_data['Unknowns']['sec_forces'])
    except:
        pass

if show_wing:
    for i in range(len(sec_forces)):
        L = sec_forces[i][:, 2] / normals[i][:, 2]
        lift.append(L.T * cos_dih[i])

def plot_wing(mesh, iteration):
    az = ax.azim
    el = ax.elev
    dist = ax.dist

    mesh0 = mesh[iteration]


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


def plot_sides(mesh, twist, lift, t, vonmises, curr_pos):
    m_vals = mesh[curr_pos]
    span = (m_vals[0, :, 1] / (m_vals[0, -1, 1]) - 0.5) * 2
    span_diff = ((m_vals[0, :-1, 1] + m_vals[0, 1:, 1])/2 / (m_vals[0, -1, 1]) - 0.5) * 2

    if show_tube:
        thick_vals = t[curr_pos]
        vm_vals = vonmises[curr_pos]

        ax4.plot(span_diff, thick_vals, lw=2, c='b')
        ax4.locator_params(axis='y',nbins=3)
        ax4.locator_params(axis='x',nbins=3)
        ax4.set_ylabel('thickness', rotation="horizontal", ha="right")

        ax5.plot(span_diff, vm_vals, lw=2, c='b')
        ax5.locator_params(axis='y',nbins=3)
        ax5.locator_params(axis='x',nbins=3)
        ax5.set_ylabel('von mises', rotation="horizontal", ha="right")

    if show_wing:
        t_vals = twist[curr_pos]
        l_vals = lift[curr_pos]

        ax2.plot(span, t_vals, lw=2, c='b')
        ax2.locator_params(axis='y',nbins=3)
        ax2.locator_params(axis='x',nbins=3)
        ax2.set_ylabel('twist', rotation="horizontal", ha="right")

        ax3.plot(span_diff, l_vals, lw=2, c='b')
        ax3.locator_params(axis='y',nbins=3)
        ax3.locator_params(axis='x',nbins=3)
        ax3.set_ylabel('lift', rotation="horizontal", ha="right")




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

    if show_wing:
        ax2.cla()
        ax3.cla()
    if show_tube:
        ax4.cla()
        ax5.cla()

    plot_sides(mesh, twist, lift, t, vonmises, curr_pos)

    fig.canvas.draw()

curr_pos = start_iteration % len(mesh)

fig = plt.figure(figsize=(12, 6))
ax = plt.subplot2grid((4,8), (0,0), rowspan=4, colspan=4, projection='3d')
n_cases = len(twist)

if show_wing and not show_tube:
    ax2 = plt.subplot2grid((4,8), (0,4), rowspan=2, colspan=4)
    ax3 = plt.subplot2grid((4,8), (2,4), rowspan=2, colspan=4)
if show_tube and not show_wing:
    ax4 = plt.subplot2grid((4,8), (0,4), rowspan=2, colspan=4)
    ax5 = plt.subplot2grid((4,8), (2,4), rowspan=2, colspan=4)
if show_wing and show_tube:
    ax2 = plt.subplot2grid((4,8), (0,4), colspan=4)
    ax3 = plt.subplot2grid((4,8), (1,4), colspan=4)
    ax4 = plt.subplot2grid((4,8), (2,4), colspan=4)
    ax5 = plt.subplot2grid((4,8), (3,4), colspan=4)


fig.canvas.mpl_connect('key_press_event', key_event)
plot_wing(mesh, curr_pos)
plot_sides(mesh, twist, lift, t, vonmises, curr_pos)
plt.tight_layout()
plt.show()
