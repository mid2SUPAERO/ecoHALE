# Setup the aerostruct data using OpenMdDAO

# In Matlab code, add this before calling Python functions:
#   if count(py.sys.path,'') == 0
#       insert(py.sys.path,int32(0),'');
#   end

from __future__ import division
import numpy
import sys
import os
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
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
        'num_twist': num_twist
    }

    return kwargs

if __name__ == '__main__':
    setup()
