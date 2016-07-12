from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group
from geometry import GeometryMesh, Bspline, gen_mesh, get_inds
from transfer import TransferLoads
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

def aero(def_mesh, **kwargs):

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
    dihedral = kwargs.get('dihedral')

    # print "after: ",mesh,num_x,num_y,des_vars

    # Define Jacobians for b-spline controls
    tot_n_fem = numpy.sum(fem_ind[:, 0])
    num_surf = fem_ind.shape[0]
    jac_twist = get_bspline_mtx(num_twist, num_y)
    jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

    disp = numpy.zeros((num_y, 6))  # for display?

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
        ('def_mesh', def_mesh)
    ]

    root = Group()

    root.add('des_vars',
             IndepVarComp(des_vars),
             promotes=['twist_cp','span','v','alpha','rho','disp','dihedral','def_mesh'])
    # root.add('twist_bsp',  # What is this doing?
    #          Bspline('twist_cp', 'twist', jac_twist),
    #          promotes=['*'])
    # root.add('thickness_bsp',    # What is this doing?
    #          Bspline('thickness_cp', 'thickness', jac_thickness),
    #          promotes=['*'])
    # root.add('tube',
    #          MaterialsTube(fem_ind),
    #          promotes=['*'])

    root.add('VLMstates',
             VLMStates(aero_ind),
             promotes=[
                'def_mesh','b_pts','mid_b','c_pts','widths','normals','S_ref', # VLMGeometry
                'alpha','circulations','v',  # VLMCirculations
                'rho','sec_forces'           # VLMForces
             ])
    root.add('loads',
             TransferLoads(aero_ind, fem_ind),
             promotes=['def_mesh','sec_forces','loads'])

    prob = Problem()
    prob.root = root
    prob.setup()
    prob.run()

    # print 'Aero Complete'
    # print "Loads:"
    # print prob['loads']
    # print ''
    # print 'Def_Mesh:'
    # print prob['def_mesh']

    return prob['loads']  # Output the Loads matrix

if __name__ == "__main__":
    aero(sys.argv[1:])
