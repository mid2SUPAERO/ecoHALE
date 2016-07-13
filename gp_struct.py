# Structures Module

from __future__ import division
import numpy
import sys

from openmdao.api import IndepVarComp, Problem, Group
from geometry import GeometryMesh, Bspline, gen_mesh, get_inds
from transfer import TransferDisplacements
from materials import MaterialsTube
from spatialbeam import SpatialBeamStates
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

def struct(loads, **kwargs):

    # #print "before: ",kwargs
    # if not kwargs:
    #     from gp_setup import setup
    #     kwargs = setup()

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
    E = kwargs.get('E')
    G = kwargs.get('G')
    stress = kwargs.get('stress')
    mrho = kwargs.get('mrho')
    tot_n_fem = kwargs.get('tot_n_fem')
    num_surf = kwargs.get('num_surf')
    jac_twist = kwargs.get('jac_twist')
    jac_thickness = kwargs.get('jac_thickness')

    # Define the design variables
    des_vars = [
        ('twist_cp', twist_cp),
        ('thickness_cp', thickness_cp),
        ('r', r),
        # ('dihedral', dihedral),
        # ('sweep', sweep),
        # ('span', span),
        # ('taper', taper),
        # ('v', v),
        # ('alpha', alpha),
        # ('rho', rho),
        # ('disp', disp),
        ('loads', loads)
    ]

    # print '^^^loads^^^'
    # print loads

    root = Group()

    root.add('des_vars',
             IndepVarComp(des_vars),
            #  promotes=['thickness_cp','r','loads'])
            promotes=['*'])

    # root.add('twist_bsp',  # What is this doing?
    #          Bspline('twist_cp', 'twist', jac_twist),
    #          promotes=['*'])
    root.add('twist_bsp',
             Bspline('twist_cp', 'twist', jac_twist),
             promotes=['*'])
    root.add('thickness_bsp',    # What is this doing?
             Bspline('thickness_cp', 'thickness', jac_thickness),
            #  promotes=['thickness'])
            promotes=['*'])
    root.add('mesh',
             GeometryMesh(mesh, aero_ind),
             promotes=['*'])
    root.add('tube',
             MaterialsTube(fem_ind),
            #  promotes=['r','thickness','A','Iy','Iz','J'])
            promotes=['*'])
    root.add('spatialbeamstates',
             SpatialBeamStates(aero_ind, fem_ind, E, G),
            #  promotes=[
            #     'mesh', # ComputeNodes
            #     'A','Iy','Iz','J','loads', # SpatialBeamFEM
            #     'disp' # SpatialBeamDisp
            #  ])
            promotes=['*'])
    root.add('transferdisp',
             TransferDisplacements(aero_ind, fem_ind),
            #  promotes=['mesh','disp','def_mesh'])
            promotes=['*'])

    prob = Problem()
    prob.root = root
    prob.setup()
    prob.run()

    def_mesh = prob['def_mesh']

    print "2.5 --- from struct... def_mesh"
    print def_mesh

    return def_mesh  # Output the def_mesh matrix

if __name__ == "__main__":
    struct(sys.argv[1:])
