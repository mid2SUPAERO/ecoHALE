# Setup the aerostruct data using OpenMdDAO

# In Matlab code, add this before calling Python functions:
#   if count(py.sys.path,'') == 0
#       insert(py.sys.path,int32(0),'');
#   end

from __future__ import division
import numpy
import sys
import os
from openmdao.api import IndepVarComp, Problem, Group
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from transfer import TransferDisplacements
from b_spline import get_bspline_mtx
from spatialbeam import radii

def setup(num_inboard=2, num_outboard=3):

    # Define the aircraft properties
    from CRM import span, v, alpha, rho

    # Create the mesh with 2 inboard points and 3 outboard points.
    # This will be mirrored to produce a mesh with 7 spanwise points,
    # or 6 spanwise panels
    mesh = gen_crm_mesh(n_points_inboard=2, n_points_outboard=3, num_x=2)
    num_x, num_y = mesh.shape[:2]
    num_twist = numpy.max([int((num_y - 1) / 5), 5])

    r = radii(mesh)

    # Set the number of thickness control points and the initial thicknesses
    num_thickness = num_twist
    t = r / 10

    mesh = mesh.reshape(-1, mesh.shape[-1])
    aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
    fem_ind = [num_y]
    aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

    # Set additional mesh parameters
    dihedral = 0.  # dihedral angle in degrees
    sweep = 0.  # shearing sweep angle in degrees
    taper = 1.  # taper ratio

    # Initial displacements of zero
    tot_n_fem = numpy.sum(fem_ind[:, 0])
    disp = numpy.zeros((tot_n_fem, 6))

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
        ('disp', disp),
        ('aero_ind', aero_ind),
        ('fem_ind', fem_ind)
    ]

    root = Group()

    root.add('des_vars',
         IndepVarComp(des_vars),
         promotes=['twist_cp','span','v','alpha','rho','disp','dihedral'])
    root.add('mesh',  # This component is needed, otherwise resulting loads matrix is NaN
         GeometryMesh(mesh, aero_ind), # changes mesh given span, sweep, twist, and des_vars
         promotes=['span','sweep','dihedral','twist','taper','mesh'])
    root.add('def_mesh',
         TransferDisplacements(aero_ind, fem_ind),
         promotes=['mesh','disp','def_mesh'])

    prob = Problem()
    prob.root = root
    prob.setup()
    prob.run()

    # Output the def_mesh for the aero modules
    def_mesh = prob['def_mesh']

    # Other variables needed for aero and struct modules
    kwargs = {
        'mesh': mesh,
        'num_x': num_x,
        'num_y': num_y,
        'span': span,
        'twist_cp': numpy.zeros(num_twist),
        'thickness_cp': numpy.ones(num_thickness)*numpy.max(t),
        'v': v,
        'alpha': alpha,
        'rho': rho,
        'r': r,
        't': t,
        'aero_ind': aero_ind,
        'fem_ind': fem_ind,
        'num_thickness': num_thickness,
        'num_twist': num_twist,
        'sweep': sweep,
        'taper': taper,
        'dihedral': dihedral
    }

    return def_mesh, kwargs

if __name__ == '__main__':
    setup()
