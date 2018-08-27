from __future__ import division, print_function
import numpy as np
from scipy.linalg import lu_factor, lu_solve

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

def norm(vec, axis=None):
    return np.sqrt(np.sum(vec**2, axis=axis))

def unit(vec):
    return vec / norm(vec)

def norm_d(vec):
    vec_d = vec/norm(vec)
    return vec_d

def unit_d(vec):
    n_d = norm_d(vec)
    normvec = norm(vec)
    vec_d = np.outer((-vec/(normvec*normvec)),n_d)+1/normvec*np.eye(len(vec))

    return vec_d

# This is a limited cross product definition for 3 vectors
def cross_d(a,b):
    if not isinstance(a, np.ndarray) :
        a = np.array(a)
        if a.shape!=(3,):
            raise ValueError('a must be a (3,) nd array')
    if not isinstance(b, np.ndarray):
        b = np.array(b)
        if b.shape!=(3,):
            raise ValueError('b must be a (3,) nd array')

    dcda = np.zeros([3,3])
    dcdb = np.zeros([3,3])

    dcda[0,1]=b[2]
    dcda[0,2]=-b[1]
    dcda[1,0]=-b[2]
    dcda[1,2]=b[0]
    dcda[2,0]=b[1]
    dcda[2,1]=-b[0]

    dcdb[0,1]=-a[2]
    dcdb[0,2]=a[1]
    dcdb[1,0]=a[2]
    dcdb[1,2]=-a[0]
    dcdb[2,0]=-a[1]
    dcdb[2,1]=a[0]

    return dcda,dcdb


def radii(mesh, t_c=0.15):

    """
    Obtain the radii of the FEM element based on local chord.
    """
    vectors = mesh[-1, :, :] - mesh[0, :, :]
    chords = np.sqrt(np.sum(vectors**2, axis=1))
    mean_chords = 0.5 * chords[:-1] + 0.5 * chords[1:]
    return t_c * mean_chords / 2.

def _assemble_system(nodes, A, J, Iy, Iz,
                     K_a, K_t, K_y, K_z,
                     cons, E, G, x_gl, T,
                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                     const2, const_y, const_z, n, size, K):

    """
    Assemble the structural stiffness matrix based on 6 degrees of freedom
    per element.

    Can be run in Fortran or Python code depending on the flags used.
    """

    # Fortran
    if fortran_flag:
        K = OAS_API.oas_api.assemblestructmtx(nodes, A, J, Iy, Iz,
                                     K_a, K_t, K_y, K_z,
                                     cons, E, G, x_gl, T,
                                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                                     const2, const_y, const_z)

    # Python
    else:
        K[:] = 0.

        # Loop over each element
        for ielem in range(n-1):

            # Obtain the element nodes
            P0 = nodes[ielem, :]
            P1 = nodes[ielem+1, :]

            x_loc = unit(P1 - P0)
            y_loc = unit(np.cross(x_loc, x_gl))
            z_loc = unit(np.cross(x_loc, y_loc))

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            for ind in range(4):
                T_elem[3*ind:3*ind+3, 3*ind:3*ind+3] = T

            L = norm(P1 - P0)
            EA_L = E * A[ielem] / L
            GJ_L = G * J[ielem] / L
            EIy_L3 = E * Iy[ielem] / L**3
            EIz_L3 = E * Iz[ielem] / L**3

            K_a[:, :] = EA_L * const2
            K_t[:, :] = GJ_L * const2

            K_y[:, :] = EIy_L3 * const_y
            K_y[1, :] *= L
            K_y[3, :] *= L
            K_y[:, 1] *= L
            K_y[:, 3] *= L

            K_z[:, :] = EIz_L3 * const_z
            K_z[1, :] *= L
            K_z[3, :] *= L
            K_z[:, 1] *= L
            K_z[:, 3] *= L

            K_elem[:] = 0
            K_elem += S_a.T.dot(K_a).dot(S_a)
            K_elem += S_t.T.dot(K_t).dot(S_t)
            K_elem += S_y.T.dot(K_y).dot(S_y)
            K_elem += S_z.T.dot(K_z).dot(S_z)

            res = T_elem.T.dot(K_elem).dot(T_elem)

            in0, in1 = ielem, ielem+1

            # Populate the full matrix with stiffness
            # contributions from each node
            K[6*in0:6*in0+6, 6*in0:6*in0+6] += res[:6, :6]
            K[6*in1:6*in1+6, 6*in0:6*in0+6] += res[6:, :6]
            K[6*in0:6*in0+6, 6*in1:6*in1+6] += res[:6, 6:]
            K[6*in1:6*in1+6, 6*in1:6*in1+6] += res[6:, 6:]

        # Include a scaled identity matrix in the rows and columns
        # corresponding to the structural constraints.
        # Hardcoded 1 constraint for now.
        for ind in range(1):
            for k in range(6):
                K[-6+k, 6*cons+k] = 1.e9
                K[6*cons+k, -6+k] = 1.e9

    return K
