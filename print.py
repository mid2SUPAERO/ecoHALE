from __future__ import division
import sqlitedict
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



db = sqlitedict.SqliteDict('aerostruct.db', 'openmdao')

variables = ['CL', 'CD', 'alpha', 'failure', 'fuelburn', 'eq_con']

for name in variables:
    print name,
print

for case_name, case_data in db.iteritems():
    if "metadata" in case_name or "derivs" in case_name:
        continue # don't plot these cases

    if 1:
        print case_name,
        for name in variables:
            print case_data['Unknowns'][name],
        print

    save = case_data

exit()



variables = ['v', 'circulations', 'alpha', 'def_mesh', 'normals', 'b_pts', 'widths', 'S_ref', 'CD']
for name in variables:
    print name
    print save['Unknowns'][name]

params = {}
for name in variables:
    params[name] = save['Unknowns'][name]
unknowns = {}
resids = {}

from weissinger import WeissingerDragCoeff

import pylab

alphas = numpy.linspace(-20, 25, 20)
CDs = numpy.zeros((20))
for i, alpha in enumerate(alphas):
    params['alpha'] = alpha
    w = WeissingerDragCoeff(5)
    w.solve_nonlinear(params, unknowns, resids)
    CDs[i] = unknowns['CD'].real

    print 'alpha:', alpha, 'CD:', unknowns['CD'].real


pylab.plot(alphas, CDs)
pylab.xlabel('$ \\alpha $', fontsize=24)
pylab.ylabel('$C_D$', fontsize=24)
pylab.show()

#w = WeissingerDragCoeff(5)
#w.solve_nonlinear(params, unknowns, resids)
#print unknowns

print
print
print

from weiss import VLM

n = 20
CLs = numpy.zeros((n))
CDs = numpy.zeros((n))
alphas = numpy.linspace(-3, 15, n)
for i, alpha in enumerate(alphas):
    v = VLM()
    v.mesh = save['Unknowns']['def_mesh']
    v.alpha = alpha
    v.v = save['Unknowns']['v']*1e3
    v.assemble()
    CLs[i] = v.CL
    CDs[i] = v.CD# + .009364 # extra factor added from paper
    print v.CD

'''
import pylab
pylab.plot(alphas, CLs)
pylab.xlabel('$ \\alpha $', fontsize=24)
pylab.ylabel('$C_L$', fontsize=24)
pylab.show()

pylab.plot(CDs, CLs)
pylab.xlabel('$C_D$', fontsize=24)
pylab.ylabel('$C_L$', fontsize=24)
pylab.show()
'''

pylab.plot(alphas, CDs)
pylab.xlabel('$ \\alpha $', fontsize=24)
pylab.ylabel('$C_D$', fontsize=24)
pylab.show()
