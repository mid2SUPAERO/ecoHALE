""" Define the structural analysis component using spatial beam theory.

Each FEM element has 6-DOF; translation in the x, y, and z direction and
rotation about the x, y, and z-axes.

"""

from __future__ import division, print_function
import numpy
numpy.random.seed(123)

from openmdao.api import Component, Group, LinearSystem
from scipy.linalg import lu_factor, lu_solve

try:
    import OAS_API
    # a
    # Make sure we don't use Fortran here; temporary for assignment
    fortran_flag = True
except:
    fortran_flag = False

def view_mat(mat):
    """ Helper function used to visually examine matrices. """
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()


def norm(vec):
    return numpy.sqrt(numpy.sum(vec**2))


def unit(vec):
    return vec / norm(vec)


def radii(mesh, t_c=0.15):
    """ Obtain the radii of the FEM component based on chord. """
    vectors = mesh[-1, :, :] - mesh[0, :, :]
    chords = numpy.sqrt(numpy.sum(vectors**2, axis=1))
    chords = 0.5 * chords[:-1] + 0.5 * chords[1:]
    return t_c * chords


def _assemble_system(nodes, A, J, Iy, Iz, loads,
                     K_a, K_t, K_y, K_z,
                     cons, E, G, x_gl, T,
                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                     const2, const_y, const_z, n, size, K, rhs):

    """
    Assemble the structural stiffness matrix based on 6 degrees of freedom
    per element.

    Can be run in dense Fortran or dense
    Python code depending on the flags used. Currently, dense Fortran
    seems to be the fastest version across many matrix sizes.

    """

    # Populate the right-hand side of the linear system using the
    # prescribed or computed loads
    rhs[:] = 0.0
    rhs[:6*n] = loads.reshape(n*6)
    rhs[numpy.abs(rhs) < 1e-6] = 0.

    # Dense Fortran
    if fortran_flag:
        K = OAS_API.oas_api.assemblestructmtx(nodes, A, J, Iy, Iz,
                                     K_a, K_t, K_y, K_z,
                                     cons, E, G, x_gl, T,
                                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                                     const2, const_y, const_z)

    # Dense Python
    else:

        K[:] = 0.

        # Loop over each element
        for ielem in xrange(n-1):

            # Obtain the element nodes
            P0 = nodes[ielem, :]
            P1 = nodes[ielem+1, :]

            x_loc = unit(P1 - P0)
            y_loc = unit(numpy.cross(x_loc, x_gl))
            z_loc = unit(numpy.cross(x_loc, y_loc))

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            for ind in xrange(4):
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
        for ind in xrange(1):
            for k in xrange(6):
                K[-6+k, 6*cons+k] = 1.e9
                K[6*cons+k, -6+k] = 1.e9

    return K, rhs


class AssembleK(Component):
    """
    Compute the displacements and rotations by solving the linear system
    using the structural stiffness matrix.

    Parameters
    ----------
    A[ny-1] : array_like
        Areas for each FEM element.
    Iy[ny-1] : array_like
        Mass moment of inertia around the y-axis for each FEM element.
    Iz[ny-1] : array_like
        Mass moment of inertia around the z-axis for each FEM element.
    J[ny-1] : array_like
        Polar moment of inertia for each FEM element.
    nodes[ny, 3] : array_like
        Flattened array with coordinates for each FEM node.
    loads[ny, 6] : array_like
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    disp_aug[6*(ny+1)] : array_like
        Augmented displacement array. Obtained by solving the system
        K * disp_aug = rhs, where rhs is a flattened version of loads.

    """

    def __init__(self, surface, cg_x=5):
        super(AssembleK, self).__init__()

        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'central'

        self.ny = surface['num_y']

        self.size = size = 6 * self.ny + 6

        self.add_param('A', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('Iy', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('Iz', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('J', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('nodes', val=numpy.zeros((self.ny, 3), dtype="complex"))
        self.add_param('loads', val=numpy.zeros((self.ny, 6), dtype="complex"))

        self.add_output('K', val=numpy.zeros((size, size), dtype="complex"))
        self.add_output('rhs', val=numpy.zeros((size), dtype="complex"))

        self.E = surface['E']
        self.G = surface['G']
        self.cg_x = cg_x

        self.const2 = numpy.array([
            [1, -1],
            [-1, 1],
        ], dtype='complex')
        self.const_y = numpy.array([
            [12, -6, -12, -6],
            [-6, 4, 6, 2],
            [-12, 6, 12, 6],
            [-6, 2, 6, 4],
        ], dtype='complex')
        self.const_z = numpy.array([
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4],
        ], dtype='complex')
        self.x_gl = numpy.array([1, 0, 0], dtype='complex')

        self.K_elem = numpy.zeros((12, 12), dtype='complex')
        self.T_elem = numpy.zeros((12, 12), dtype='complex')
        self.T = numpy.zeros((3, 3), dtype='complex')

        self.K = numpy.zeros((size, size), dtype='complex')
        self.rhs = numpy.zeros(size, dtype='complex')

        self.K_a = numpy.zeros((2, 2), dtype='complex')
        self.K_t = numpy.zeros((2, 2), dtype='complex')
        self.K_y = numpy.zeros((4, 4), dtype='complex')
        self.K_z = numpy.zeros((4, 4), dtype='complex')

        self.S_a = numpy.zeros((2, 12), dtype='complex')
        self.S_a[(0, 1), (0, 6)] = 1.

        self.S_t = numpy.zeros((2, 12), dtype='complex')
        self.S_t[(0, 1), (3, 9)] = 1.

        self.S_y = numpy.zeros((4, 12), dtype='complex')
        self.S_y[(0, 1, 2, 3), (2, 4, 8, 10)] = 1.

        self.S_z = numpy.zeros((4, 12), dtype='complex')
        self.S_z[(0, 1, 2, 3), (1, 5, 7, 11)] = 1.

    def solve_nonlinear(self, params, unknowns, resids):

        # Find constrained nodes based on closeness to specified cg point
        nodes = params['nodes']
        dist = nodes - numpy.array([self.cg_x, 0, 0])
        idx = (numpy.linalg.norm(dist, axis=1)).argmin()
        self.cons = idx

        loads = params['loads']

        self.K, self.rhs = \
            _assemble_system(params['nodes'],
                             params['A'], params['J'], params['Iy'],
                             params['Iz'], loads, self.K_a, self.K_t,
                             self.K_y, self.K_z, self.cons,
                             self.E, self.G, self.x_gl, self.T, self.K_elem,
                             self.S_a, self.S_t, self.S_y, self.S_z,
                             self.T_elem, self.const2, self.const_y,
                             self.const_z, self.ny, self.size,
                             self.K, self.rhs)

        unknowns['K'] = self.K
        unknowns['rhs'] = self.rhs

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        # Find constrained nodes based on closeness to specified cg point
        nodes = params['nodes']
        dist = nodes - numpy.array([self.cg_x, 0, 0])
        idx = (numpy.linalg.norm(dist, axis=1)).argmin()
        self.cons = idx

        loads = params['loads']

        A = params['A']
        J = params['J']
        Iy = params['Iy']
        Iz = params['Iz']

        if mode == 'fwd':
            K, Kd = OAS_API.oas_api.assemblestructmtx_d(nodes, dparams['nodes'], A, dparams['A'],
                                         J, dparams['J'], Iy, dparams['Iy'],
                                         Iz, dparams['Iz'],
                                         self.K_a, self.K_t, self.K_y, self.K_z,
                                         self.cons, self.E, self.G, self.x_gl, self.T,
                                         self.K_elem, self.S_a, self.S_t, self.S_y, self.S_z, self.T_elem,
                                         self.const2, self.const_y, self.const_z)

            dresids['K'] += Kd
            dresids['rhs'][:-6] += dparams['loads'].reshape(-1)

        if mode == 'rev':
            nodesb, Ab, Jb, Iyb, Izb = OAS_API.oas_api.assemblestructmtx_b(nodes, A, J, Iy, Iz,
                                self.K_a, self.K_t, self.K_y, self.K_z,
                                self.cons, self.E, self.G, self.x_gl, self.T,
                                self.K_elem, self.S_a, self.S_t, self.S_y, self.S_z, self.T_elem,
                                self.const2, self.const_y, self.const_z, self.K, dresids['K'])

            dparams['nodes'] += nodesb
            dparams['A'] += Ab
            dparams['J'] += Jb
            dparams['Iy'] += Iyb
            dparams['Iz'] += Izb

            dparams['loads'] += dresids['rhs'][:-6].reshape(-1, 6)


class SpatialBeamFEM(Component):
    """
    Compute the displacements and rotations by solving the linear system
    using the structural stiffness matrix.

    Parameters
    ----------
    A[ny-1] : array_like
        Areas for each FEM element.
    Iy[ny-1] : array_like
        Mass moment of inertia around the y-axis for each FEM element.
    Iz[ny-1] : array_like
        Mass moment of inertia around the z-axis for each FEM element.
    J[ny-1] : array_like
        Polar moment of inertia for each FEM element.
    nodes[ny, 3] : array_like
        Flattened array with coordinates for each FEM node.
    loads[ny, 6] : array_like
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    disp_aug[6*(ny+1)] : array_like
        Augmented displacement array. Obtained by solving the system
        K * disp_aug = rhs, where rhs is a flattened version of loads.

    """

    def __init__(self, size):
        super(SpatialBeamFEM, self).__init__()

        self.add_param('K', val=numpy.zeros((size, size), dtype="complex"))
        self.add_param('rhs', val=numpy.zeros((size), dtype="complex"))
        self.add_state('disp_aug', val=numpy.zeros((size), dtype="complex"))

        self.size = size

        # cache
        self.lup = None
        self.rhs_cache = None

    def solve_nonlinear(self, params, unknowns, resids):
        """ Use numpy to solve Ax=b for x.
        """

        # lu factorization for use with solve_linear
        self.lup = lu_factor(params['K'])

        unknowns['disp_aug'] = lu_solve(self.lup, params['rhs'])
        resids['disp_aug'] = params['K'].dot(unknowns['disp_aug']) - params['rhs']

    def apply_nonlinear(self, params, unknowns, resids):
        """Evaluating residual for given state."""

        resids['disp_aug'] = params['K'].dot(unknowns['disp_aug']) - params['rhs']

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Apply the derivative of state variable with respect to
        everything."""

        if mode == 'fwd':

            if 'disp_aug' in dunknowns:
                dresids['disp_aug'] += params['K'].dot(dunknowns['disp_aug'])
            if 'K' in dparams:
                dresids['disp_aug'] += dparams['K'].dot(unknowns['disp_aug'])
            if 'rhs' in dparams:
                dresids['disp_aug'] -= dparams['rhs']

        elif mode == 'rev':

            if 'disp_aug' in dunknowns:
                dunknowns['disp_aug'] += params['K'].T.dot(dresids['disp_aug'])
            if 'K' in dparams:
                dparams['K'] += numpy.outer(unknowns['disp_aug'], dresids['disp_aug']).T
            if 'rhs' in dparams:
                dparams['rhs'] -= dresids['disp_aug']

    def solve_linear(self, dumat, drmat, vois, mode=None):
        """ LU backsubstitution to solve the derivatives of the linear system."""

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t=0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t=1

        if self.rhs_cache is None:
            self.rhs_cache = numpy.zeros((self.size, ))
        rhs = self.rhs_cache

        for voi in vois:
            rhs[:] = rhs_vec[voi]['disp_aug']

            sol = lu_solve(self.lup, rhs, trans=t)

            sol_vec[voi]['disp_aug'] = sol[:]


class SpatialBeamDisp(Component):
    """
    Select displacements from augmented vector.

    The solution to the linear system has additional results due to the
    constraints on the FEM model. The displacements from this portion of
    the linear system are not needed, so we select only the relevant
    portion of the displacements for further calculations.

    Parameters
    ----------
    disp_aug[6*(ny+1)] : array_like
        Augmented displacement array. Obtained by solving the system
        K * disp_aug = rhs, where rhs is a flattened version of loads.

    Returns
    -------
    disp[6*ny] : array_like
        Actual displacement array formed by truncating disp_aug.

    """

    def __init__(self, surface):
        super(SpatialBeamDisp, self).__init__()

        self.ny = surface['num_y']

        self.add_param('disp_aug', val=numpy.zeros(((self.ny+1)*6), dtype='complex'))
        self.add_output('disp', val=numpy.zeros((self.ny, 6), dtype='complex'))

    def solve_nonlinear(self, params, unknowns, resids):
        # Obtain the relevant portions of disp_aug and store the reshaped
        # displacements in disp
        unknowns['disp'] = params['disp_aug'][:-6].reshape((-1, 6))

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        n = self.ny * 6
        jac['disp', 'disp_aug'][:n, :n] = numpy.eye((n))
        return jac


class ComputeNodes(Component):
    """
    Compute FEM nodes based on aerodynamic mesh.

    The FEM nodes are placed at 0.35*chord, or based on the fem_origin value.

    Parameters
    ----------
    mesh[nx, ny, 3] : array_like
        Array defining the nodal points of the lifting surface.

    Returns
    -------
    nodes[ny, 3] : array_like
        Flattened array with coordinates for each FEM node.

    """

    def __init__(self, surface):
        super(ComputeNodes, self).__init__()

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.fem_origin = surface['fem_origin']

        self.add_param('mesh', val=numpy.zeros((self.nx, self.ny, 3), dtype=complex))
        self.add_output('nodes', val=numpy.zeros((self.ny, 3), dtype=complex))

    def solve_nonlinear(self, params, unknowns, resids):
        w = self.fem_origin
        mesh = params['mesh']

        unknowns['nodes'] = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        w = self.fem_origin

        n = self.ny * 3
        jac['nodes', 'mesh'][:n, :n] = numpy.eye(n) * (1-w)
        jac['nodes', 'mesh'][:n, -n:] = numpy.eye(n) * w

        return jac


class SpatialBeamEnergy(Component):
    """ Compute strain energy.

    Parameters
    ----------
    disp[ny, 6] : array_like
        Actual displacement array formed by truncating disp_aug.
    loads[ny, 6] : array_like
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Returns
    -------
    energy : float
        Total strain energy of the structural component.

    """

    def __init__(self, surface):
        super(SpatialBeamEnergy, self).__init__()

        ny = surface['num_y']

        self.add_param('disp', val=numpy.zeros((ny, 6), dtype=complex))
        self.add_param('loads', val=numpy.zeros((ny, 6), dtype=complex))
        self.add_output('energy', val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['energy'] = numpy.sum(params['disp'] * params['loads'])

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        jac['energy', 'disp'][0, :] = params['loads'].real.flatten()
        jac['energy', 'loads'][0, :] = params['disp'].real.flatten()
        return jac

class SpatialBeamWeight(Component):
    """ Compute total weight.

    Parameters
    ----------
    A[ny-1] : array_like
        Areas for each FEM element.
    nodes[ny, 3] : array_like
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    weight : float
        Total weight of the structural component."""

    def __init__(self, surface):
        super(SpatialBeamWeight, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']

        self.add_param('A', val=numpy.zeros((self.ny - 1), dtype=complex))
        self.add_param('nodes', val=numpy.zeros((self.ny, 3), dtype=complex))
        self.add_output('weight', val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        A = params['A']
        nodes = params['nodes']

        # Calculate the volume and weight of the total structure
        volume = numpy.sum(numpy.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1) * A)

        volume = 0.
        for i in range(self.ny-1):
            diff = (nodes[i+1, :] - nodes[i, :])**2
            diff_sum = numpy.sum(diff)
            diff_norm = numpy.sqrt(diff_sum) * A[i]
            volume = volume + diff_norm

        weight = volume * self.surface['mrho'] * 9.81

        if self.surface['symmetry']:
            weight *= 2.

        unknowns['weight'] = weight

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        A = params['A']
        nodes = params['nodes']

        # First we will solve for dweight_dA
        # Calculate the volume and weight of the total structure
        elem_lengths = numpy.linalg.norm(nodes[1:, :] - nodes[:-1, :], axis=1).reshape(1, -1)

        # Multiply by the material density and force of gravity
        dweight_dA = elem_lengths * self.surface['mrho'] * 9.81

        # Account for symmetry
        if self.surface['symmetry']:
            dweight_dA *= 2.

        # Save the result to the jacobian dictionary
        jac['weight', 'A'] = dweight_dA

        # Next, we will compute the derivative of weight wrt nodes.
        # Here we're using results from AD to compute the derivative
        # Initialize the reverse seeds.
        nodesb = numpy.zeros(nodes.shape)
        volumeb = 1.

        # Compute the norms and other backwards seeds
        norms = numpy.linalg.norm(nodes[1:, :] - nodes[:1, :], axis=1)
        diff_sumb = (A * 0.5 * norms * volumeb).reshape(-1, 1)
        tempb = 2*(nodes[1:, :] - nodes[:-1, :]) * diff_sumb

        # Sum these results to the nodesb seeds
        nodesb[1:, :] += tempb
        nodesb[:-1, :] -= tempb

        # Apply the multipliers for material properties and symmetry
        nodesb *= self.surface['mrho'] * 9.81

        if self.surface['symmetry']:
            nodesb *= 2.

        # Store the flattened array in the jacobian dictionary
        jac['weight', 'nodes'] = nodesb.reshape(1, -1)

        return jac

class SpatialBeamVonMisesTube(Component):
    """ Compute the von Mises stress in each element.

    Parameters
    ----------
    r[ny-1] : array_like
        Radii for each FEM element.
    nodes[ny, 3] : array_like
        Flattened array with coordinates for each FEM node.
    disp[ny, 6] : array_like
        Displacements of each FEM node.

    Returns
    -------
    vonmises[ny-1, 2] : array_like
        von Mises stress magnitudes for each FEM element.

    """

    def __init__(self, surface):
        super(SpatialBeamVonMisesTube, self).__init__()

        self.ny = surface['num_y']

        self.add_param('nodes', val=numpy.zeros((self.ny, 3),
                       dtype="complex"))
        self.add_param('r', val=numpy.zeros((self.ny - 1),
                       dtype="complex"))
        self.add_param('disp', val=numpy.zeros((self.ny, 6),
                       dtype="complex"))

        self.add_output('vonmises', val=numpy.zeros((self.ny-1, 2),
                        dtype="complex"))

        if not fortran_flag:
            self.deriv_options['type'] = 'cs'
            self.deriv_options['form'] = 'central'

        self.E = surface['E']
        self.G = surface['G']

        self.T = numpy.zeros((3, 3), dtype='complex')
        self.x_gl = numpy.array([1, 0, 0], dtype='complex')
        self.t = 0

    def solve_nonlinear(self, params, unknowns, resids):
        r = params['r']
        disp = params['disp']
        nodes = params['nodes']
        vonmises = unknowns['vonmises']
        T = self.T
        E = self.E
        G = self.G
        x_gl = self.x_gl

        if fortran_flag:
            vm = OAS_API.oas_api.calc_vonmises(nodes, r, disp, E, G, x_gl)
            unknowns['vonmises'] = vm

        else:

            num_elems = self.ny - 1
            for ielem in xrange(self.ny-1):

                P0 = nodes[ielem, :]
                P1 = nodes[ielem+1, :]
                L = norm(P1 - P0)

                x_loc = unit(P1 - P0)
                y_loc = unit(numpy.cross(x_loc, x_gl))
                z_loc = unit(numpy.cross(x_loc, y_loc))

                T[0, :] = x_loc
                T[1, :] = y_loc
                T[2, :] = z_loc

                u0x, u0y, u0z = T.dot(disp[ielem, :3])
                r0x, r0y, r0z = T.dot(disp[ielem, 3:])
                u1x, u1y, u1z = T.dot(disp[ielem+1, :3])
                r1x, r1y, r1z = T.dot(disp[ielem+1, 3:])

                tmp = numpy.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
                sxx0 = E * (u1x - u0x) / L + E * r[ielem] / L * tmp
                sxx1 = E * (u0x - u1x) / L + E * r[ielem] / L * tmp
                sxt = G * r[ielem] * (r1x - r0x) / L

                vonmises[ielem, 0] = numpy.sqrt(sxx0**2 + sxt**2)
                vonmises[ielem, 1] = numpy.sqrt(sxx1**2 + sxt**2)

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        r = params['r'].real
        disp = params['disp'].real
        nodes = params['nodes'].real
        vonmises = unknowns['vonmises'].real
        E = self.E
        G = self.G
        x_gl = self.x_gl

        if mode == 'fwd':
            _, a = OAS_API.oas_api.calc_vonmises_d(nodes, dparams['nodes'], r, dparams['r'], disp, dparams['disp'], E, G, x_gl)
            dresids['vonmises'] += a

        if mode == 'rev':
            a, b, c = OAS_API.oas_api.calc_vonmises_b(nodes, r, disp, E, G, x_gl, vonmises, dresids['vonmises'])
            dparams['nodes'] += a
            dparams['r'] += b
            dparams['disp'] += c

class SpatialBeamFailureKS(Component):
    """
    Aggregate failure constraints from the structure.

    To simplify the optimization problem, we aggregate the individual
    elemental failure constraints using a Kreisselmeier-Steinhauser (KS)
    function.

    The KS function produces a smoother constraint than using a max() function
    to find the maximum point of failure, which produces a better-posed
    optimization problem.

    Parameters
    ----------
    vonmises[ny-1, 2] : array_like
        von Mises stress magnitudes for each FEM element.

    Returns
    -------
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Used to simplify the optimization problem by
        reducing the number of constraints.

    """

    def __init__(self, surface, rho=10):
        super(SpatialBeamFailureKS, self).__init__()

        self.ny = surface['num_y']

        self.add_param('vonmises', val=numpy.zeros((self.ny-1, 2), dtype=complex))
        self.add_output('failure', val=0.)

        self.sigma = surface['stress']
        self.rho = rho

    def solve_nonlinear(self, params, unknowns, resids):
        sigma = self.sigma
        rho = self.rho
        vonmises = params['vonmises']

        fmax = numpy.max(vonmises/sigma - 1)

        nlog, nsum, nexp = numpy.log, numpy.sum, numpy.exp
        ks = 1 / rho * nlog(nsum(nexp(rho * (vonmises/sigma - 1 - fmax))))
        unknowns['failure'] = fmax + ks

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        vonmises = params['vonmises']
        sigma = self.sigma
        rho = self.rho

        if mode == 'fwd':

            vonmisesd = dparams['vonmises']

            fmax = numpy.max(vonmises / sigma - 1)
            i, j = numpy.where((vonmises/sigma - 1)==fmax)
            i, j = i[0], j[0]
            fmaxd = vonmisesd[i, j] / sigma

            arg1d = rho*(vonmisesd/sigma-fmaxd)
            arg1 = rho*(vonmises/sigma-1-fmax)
            arg3d = numpy.sum(arg1d * numpy.exp(arg1))
            arg3 = numpy.sum(numpy.exp(arg1))

            ksd = arg3d/(rho*arg3)
            ks = 1/rho*numpy.log(arg3)

            dresids['failure'] += fmaxd + ksd
            failure = fmax + ks

        if mode == 'rev':

            fmax = numpy.max(vonmises / sigma - 1)
            i, j = numpy.where((vonmises/sigma - 1)==fmax)
            i, j = i[0], j[0]

            ksb = dresids['failure']

            tempb0 = ksb / (rho * numpy.sum(numpy.exp(rho * (vonmises/sigma - fmax - 1))))
            tempb = numpy.exp(rho*(vonmises/sigma-fmax-1))*rho*tempb0
            fmaxb = ksb - numpy.sum(tempb)

            dparams['vonmises'] = tempb / sigma
            dparams['vonmises'][i, j] += fmaxb / sigma


class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def __init__(self, surface):
        super(SpatialBeamStates, self).__init__()

        size = 6 * surface['num_y'] + 6

        self.add('nodes',
                 ComputeNodes(surface),
                 promotes=['*'])
        self.add('assembly',
                 AssembleK(surface),
                 promotes=['*'])
        self.add('fem',
                 SpatialBeamFEM(size),
                 promotes=['*'])
        self.add('disp',
                 SpatialBeamDisp(surface),
                 promotes=['*'])


class SpatialBeamFunctionals(Group):
    """ Group that contains the spatial beam functionals used to evaluate
    performance. """

    def __init__(self, surface):
        super(SpatialBeamFunctionals, self).__init__()

        self.add('energy',
                 SpatialBeamEnergy(surface),
                 promotes=['*'])
        self.add('weight',
                 SpatialBeamWeight(surface),
                 promotes=['*'])
        self.add('vonmises',
                 SpatialBeamVonMisesTube(surface),
                 promotes=['*'])
        self.add('failure',
                 SpatialBeamFailureKS(surface),
                 promotes=['*'])
