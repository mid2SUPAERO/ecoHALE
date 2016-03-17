import sqlitedict

import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

db = sqlitedict.SqliteDict('weissinger.db', 'openmdao')


cl = []
cd = []
twist = []
mesh = []

for case_name, case_data in db.iteritems(): 
    if "metadata" in case_name or "derivs" in case_name: 
        continue # adon't plot these cases

    cl.append(case_data['Unknowns']['CL'])
    cd.append(case_data['Unknowns']['CD'])
    twist.append(case_data['Unknowns']['twist'])
    mesh.append(case_data['Unknowns']['mesh'])


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.plot(cl, lw=2)
ax1.set_ylabel(r'$C_L$', rotation="horizontal", ha="right", fontsize=15)

ax2.plot(cd, lw=2)
ax2.set_xlabel('Iteration Number', fontsize=15)
ax1.set_ylabel(r'$C_D$', rotation="horizontal", ha="right", fontsize=15)

fig, ax = plt.subplots()
n_cases = len(twist)

for i,(t_vals, m_vals) in enumerate(zip(twist, mesh)): 
    # m_vals[0,:,1].shape, t_vals.shape
    ax.plot(m_vals[0,:,1], t_vals, lw=2, c=plt.cm.jet(float(i)/n_cases))


mesh0 = mesh[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = mesh0[0,:,0]
ys = mesh0[0,:,1]
zs = mesh0[0,:,2]
ax.scatter(xs, ys, zs)

plt.show()