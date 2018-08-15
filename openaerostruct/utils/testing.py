from openmdao.api import Problem, Group, IndepVarComp, view_model

from six import iteritems
from numpy.testing import assert_almost_equal
from openmdao.utils.assert_utils import assert_rel_error
import numpy as np
from openaerostruct.geometry.utils import generate_mesh


def view_mat(mat1, mat2,key,tol=1e-10):
    """
    Helper function used to visually examine matrices. It plots mat1 and mat2 side by side,
    and shows the difference between the two.

    Parameters
    ----------
    mat1 : ndarray
        The Jacobian approximated by openMDAO
    mat2 : ndarray
        The Jacobian computed by compute_partials
    key : str
        The name of the tuple (of, wrt) for which the Jacobian is computed
    tol : float (Optional) 
        The tolerance, below which the two numbers are considered the same for
        plotting purposes.

    Returns
    -------
    CDw : float
        Wave drag coefficient for the lifting surface computed using the 
        Korn equation
    """
    import matplotlib.pyplot as plt
    if len(mat1.shape) > 2:
        mat1 = np.sum(mat1, axis=2)
    if len(mat2.shape) > 2:
        mat2 = np.sum(mat2, axis=2)
    vmin = np.amin(np.hstack((mat1.flatten(),mat2.flatten())))
    vmax = np.amax(np.hstack((mat1.flatten(),mat2.flatten())))
    if vmax-vmin < tol: # add small difference for plotting if both values are the same
        vmin = vmin - tol
        vmax = vmax + tol

    fig, ax = plt.subplots(ncols=3,figsize=(12,6))
    ax[0].imshow(mat1.real, interpolation='none',vmin=vmin,vmax=vmax)
    ax[0].set_title('Approximated Jacobian')

    im = ax[1].imshow(mat2.real, interpolation='none',vmin=vmin,vmax=vmax)
    fig.colorbar(im, orientation='horizontal',ax=ax[0:2].ravel().tolist())
    ax[1].set_title('User-Defined Jacobian')

    diff = mat2.real - mat1.real
    if np.max(np.abs(diff).flatten()) < tol: # add small difference for plotting if diff is small
        vmin = -1*tol
        vmax = tol
    im2 = ax[2].imshow(diff, interpolation='none', vmin=vmin,vmax=vmax)
    fig.colorbar(im2, orientation='horizontal',ax=ax[2],aspect=10)
    ax[2].set_title('Difference')
    plt.suptitle(key)
    plt.show()

def run_test(obj, comp, tol=1e-5, complex_flag=False):
    prob = Problem()
    prob.model.add_subsystem('comp', comp)
    prob.setup(force_alloc_complex=complex_flag)

    prob.run_model()
    check = prob.check_partials(compact_print=True)
    for key, subjac in iteritems(check[list(check.keys())[0]]):
        if subjac['magnitude'].fd > 1e-6:
            assert_rel_error(obj, subjac['rel error'].forward, 0., tol)
            assert_rel_error(obj, subjac['rel error'].reverse, 0., tol)
        elif np.isnan(subjac['magnitude'].fd):
            raise ValueError('Derivative magnitude is NaN')

def get_default_surfaces():
    # Create a dictionary to store options about the mesh
    mesh_dict = {'num_y' : 7,
                 'num_x' : 2,
                 'wing_type' : 'CRM',
                 'symmetry' : True,
                 'num_twist_cp' : 5}

    # Generate the aerodynamic mesh based on the previous dictionary
    mesh, twist_cp = generate_mesh(mesh_dict)

    wing_dict = {'name' : 'wing',
                 'num_y' : 4,
                 'num_x' : 2,
                 'symmetry' : True,
                 'S_ref_type' : 'wetted',
                 'CL0' : 0.1,
                 'CD0' : 0.1,
                 'mesh' : mesh,

                 # Airfoil properties for viscous drag calculation
                 'k_lam' : 0.05,         # percentage of chord with laminar
                                         # flow, used for viscous drag
                 't_over_c' : 0.15,      # thickness over chord ratio (NACA0015)
                 'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                         # thickness
                 'with_viscous' : True,  # if true, compute viscous drag
                 'with_wave' : True, # if true, computes wave drag
                 'fem_model_type' : 'tube',

                 # Structural values are based on aluminum 7075
                 'E' : 70.e9,            # [Pa] Young's modulus of the spar
                 'G' : 30.e9,            # [Pa] shear modulus of the spar
                 'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
                 'mrho' : 3.e3,          # [kg/m^3] material density
                 'fem_origin' : 0.35,    # normalized chordwise location of the spar
                 't_over_c' : 0.15,      # maximum airfoil thickness
                 'wing_weight_ratio' : 2.,

                 }

    # Create a dictionary to store options about the mesh
    mesh_dict = {'num_y' : 5,
                 'num_x' : 3,
                 'wing_type' : 'rect',
                 'symmetry' : False}

    # Generate the aerodynamic mesh based on the previous dictionary
    mesh = generate_mesh(mesh_dict)

    tail_dict = {'name' : 'tail',
                 'num_y' : 5,
                 'num_x' : 3,
                 'symmetry' : False,
                 'mesh' : mesh}

    surfaces = [wing_dict, tail_dict]

    return surfaces
