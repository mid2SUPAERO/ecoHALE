""" Example script to produce drag polar
    Flat rectangular wing from NASA report """

from __future__ import division
import numpy
import sys
import warnings
warnings.filterwarnings("ignore")

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder, pyOptSparseDriver, profile
from geometry import GeometryMesh, mesh_gen, LinearInterp
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
from openmdao.devtools.partition_tree_n2 import view_tree

# Define the aircraft properties
execfile('CRM.py')

num_x = 5
num_y = 121
num_twist = 3
span = 232.02
chord = 39.37
mesh = numpy.zeros((num_x, num_y, 3))
ny2 = (num_y + 1) / 2
half_wing = numpy.zeros((ny2))
beta = numpy.linspace(0, numpy.pi/2, ny2)

# mixed spacing with w as a weighting factor
cosine = .5 * numpy.cos(beta)**1 #  cosine spacing
uniform = numpy.linspace(0, .5, ny2)[::-1] #  uniform spacing
amt_of_cos = .5
half_wing = cosine * amt_of_cos + (1 - amt_of_cos) * uniform
full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * span

for ind_x in xrange(num_x):
    for ind_y in xrange(num_y):
        mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, full_wing[ind_y], 0] # straight elliptical spacing

root = Group()

des_vars = [
    ('twist', numpy.zeros(num_twist)),
    ('span', span),
    ('v', v),
    ('alpha', alpha),
    ('rho', rho),
    ('disp', numpy.zeros((num_y, 6)))
]

root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['*'])
root.add('mesh',
         GeometryMesh(mesh, num_twist),
         promotes=['*'])
root.add('def_mesh',
         TransferDisplacements(num_x, num_y),
         promotes=['*'])
root.add('weissingerstates',
         WeissingerStates(num_x, num_y),
         promotes=['*'])
root.add('weissingerfuncs',
         WeissingerFunctionals(num_x, num_y, CL0, CD0,  num_twist),
         promotes=['*'])

prob = Problem()
prob.root = root

prob.setup()

alpha_start = -3.
alpha_stop = 14
num_alpha = 18.

a_list = []
CL_list = []
CD_list = []

print
for alpha in numpy.linspace(alpha_start, alpha_stop, num_alpha):
    prob['alpha'] = alpha
    prob.run_once()
    print 'alpha', prob['alpha'], "; CL", prob['CL'], "; CD", prob['CD'], "; num", num_y
    a_list.append(alpha)
    CL_list.append(prob['CL'])
    CD_list.append(prob['CD'] + 0.009364)

nasa_data = numpy.loadtxt('nasa_prandtl.csv', delimiter=',', skiprows=1)
nasa_data = nasa_data[nasa_data[:, 1].argsort()]

import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(a_list, CL_list)
ax1.set_xlabel('alpha')
ax1.set_ylabel('CL')
ax2.plot(CD_list, CL_list, label='ours')
ax2.plot(nasa_data[:, 0], nasa_data[:, 1], label='nasa')
plt.legend(loc=0)
ax2.set_xlabel('CD')
ax2.set_ylabel('CL')
plt.show()
