from __future__ import division
import sqlitedict
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def _get_lengths(self, A, B, axis):
    return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

db = sqlitedict.SqliteDict('weissinger.db', 'openmdao')

twist = []
mesh = []
sec_forces = []
normals = []
cos_dih = []
lift = []

for case_name, case_data in db.iteritems():
    if "metadata" in case_name or "derivs" in case_name:
        continue # don't plot these cases
    n = case_data['Unknowns']['mesh'].shape[1]

    # only grab one half of the wing
    n21 = n/2
    sec_forces.append(case_data['Unknowns']['sec_forces'][n21:, :])
    twist.append(case_data['Unknowns']['twist'][n21:])
    normals.append(case_data['Unknowns']['normals'][n21:, :])
    cos_dih.append(case_data['Unknowns']['cos_dih'][n21:])
    mesh.append(case_data['Unknowns']['mesh'][:, n21:, :])

for i in range(len(sec_forces)):
    L = sec_forces[i][:, 2] / normals[i][:, 2]
    lift.append(L.T * cos_dih[i])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

n_cases = len(twist)

for i, (t_vals, l_vals, m_vals) in enumerate(zip(twist, lift, mesh)):
    span = (m_vals[0, :, 1] / (m_vals[0, -1, 1]) - 0.5) * 2
    span_diff = ((m_vals[0, :-1, 1] + m_vals[0, 1:, 1])/2 / (m_vals[0, -1, 1]) - 0.5) * 2

    ax1.plot(span, t_vals, lw=2, c=plt.cm.jet(i/n_cases))
    ax2.plot(span_diff, l_vals, lw=2, c=plt.cm.jet(i/n_cases))

    ax2.set_xlabel('normalized span')
    ax1.set_ylabel('twist')
    ax2.set_ylabel('lift')

plt.show()
