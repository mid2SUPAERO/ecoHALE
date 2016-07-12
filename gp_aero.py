from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, SqliteRecorder
from geometry import GeometryMesh, mesh_gen, LinearInterp
from transfer import TransferDisplacements, TransferLoads
from weissinger import WeissingerStates, WeissingerFunctionals
#from model_helpers import view_tree

# # Create the mesh with 2 inboard points and 3 outboard points
# mesh = mesh_gen(n_points_inboard=2, n_points_outboard=3)
# num_y = mesh.shape[1]
#
# if 1:
#     num_x = 2
#     num_y = 11
#     span = 10.
#     chord = 1.
#     mesh = numpy.zeros((num_x, num_y, 3))
#     for ind_x in xrange(num_x):
#         for ind_y in xrange(num_y):
#             mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord, ind_y / (num_y - 1) * span, 0]

def aero(**kwargs):

    print "before: ",kwargs
    if not kwargs:
        from gp_setup import setup
        kwargs = setup()

    mesh = kwargs['mesh']
    num_x = kwargs['num_x']
    num_y = kwargs['num_y']
    des_vars = kwargs['des_vars']

    print "after: ",mesh,num_x,num_y,des_vars
    disp = numpy.zeros((num_y, 6))  # for display?

    root = Group()

    root.add('des_vars',
             IndepVarComp(des_vars),
             # explicitly list design variables
             promotes=['twist','span','v','alpha','rho','disp','r','t'])
    root.add('mesh',
             GeometryMesh(mesh,num_x), # changes mesh given span, sweep, twist, and des_vars
             promotes=['*'])
    root.add('def_mesh',
             TransferDisplacements(num_y),
             promotes=['*'])
    root.add('weissingerstates',
             WeissingerStates(num_y),
             promotes=['*'])
    root.add('loads',
             TransferLoads(num_y),
             # explicitly list variables
             promotes=['def_mesh','sec_forces','loads'])

    prob = Problem()
    prob.root = root

    prob.setup()

    # prob.run_once()
    prob.run()
    print 'Aero Complete'
    print prob['loads']
    print prob['def_mesh']
    return

if __name__ == "__main__":
    aero(sys.argv[1:])
