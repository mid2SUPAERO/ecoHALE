from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group
from geometry import GeometryMesh, Bspline, gen_mesh, get_inds
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals
from b_spline import get_bspline_mtx
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

def aero(kwargs):

    #print "before: ",kwargs
    if not kwargs:
        from gp_setup import setup
        kwargs = setup()

    # Unpack variables
    mesh = kwargs.get('mesh')
    num_x = kwargs.get('num_x')
    num_y = kwargs.get('num_y')
    span = kwargs.get('span')
    twist_cp = kwargs.get('twist_cp')
    thickness_cp = kwargs.get('thickness_cp')
    v = kwargs.get('v')
    alpha = kwargs.get('alpha')
    rho = kwargs.get('rho')
    r = kwargs.get('r')
    t = kwargs.get('t')
    aero_ind = kwargs.get('aero_ind')
    fem_ind = kwargs.get('fem_ind')
    num_thickness = kwargs.get('num_thickness')
    num_twist = kwargs.get('num_twist')
    sweep = kwargs.get('sweep')
    taper = kwargs.get('taper')
    disp = kwargs.get('disp')

    # print "after: ",mesh,num_x,num_y,des_vars

    # Define Jacobians for b-spline controls
    tot_n_fem = numpy.sum(fem_ind[:, 0])
    num_surf = fem_ind.shape[0]
    jac_twist = get_bspline_mtx(num_twist, num_y)
    jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

    disp = numpy.zeros((num_y, 6))  # for display?

    # Set additional mesh parameters
    dihedral = 0.  # dihedral angle in degrees
    sweep = 0.  # shearing sweep angle in degrees
    taper = 1.  # taper ratio

    # Define the design variables
    des_vars = [
        ('twist_cp', numpy.zeros(num_twist)),
        ('dihedral', dihedral),
        ('sweep', sweep),
        ('span', span),
        ('taper', taper),
        ('v', v),
        ('alpha', alpha),
        ('rho', rho),
        ('disp', numpy.zeros((tot_n_fem, 6))),
        ('aero_ind', aero_ind),
        ('fem_ind', fem_ind)
    ]

    root = Group()

    root.add('des_vars',
             IndepVarComp(des_vars),
             # explicitly list design variables
             promotes=['twist_cp','span','v','alpha','rho','disp'])
    root.add('twist_bsp',
             Bspline('twist_cp', 'twist', jac_twist),
             promotes=['*'])
    root.add('thickness_bsp',
             Bspline('thickness_cp', 'thickness', jac_thickness),
             promotes=['*'])
    # root.add('tube',
    #          MaterialsTube(fem_ind),
    #          promotes=['*'])
    root.add('mesh',
             GeometryMesh(mesh, aero_ind), # changes mesh given span, sweep, twist, and des_vars
             promotes=['*'])
    root.add('def_mesh',
             TransferDisplacements(aero_ind, fem_ind),
             promotes=['*'])
    root.add('VLMstates',
             VLMStates(aero_ind),
             promotes=['*'])
    root.add('loads',
             TransferLoads(aero_ind, fem_ind),
             # explicitly list variables
             promotes=['def_mesh','sec_forces','loads'])

    prob = Problem()
    prob.root = root

    prob.setup()

    # prob.run_once()
    prob.run()
    print 'Aero Complete'
    print "Loads:"
    print prob['loads']
    print ''
    print 'Def_Mesh:'
    print prob['def_mesh']
    return

if __name__ == "__main__":
    aero(sys.argv[1:])
