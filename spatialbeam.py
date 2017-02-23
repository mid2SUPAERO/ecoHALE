""" Define the structural analysis component using spatial beam theory.

Each FEM element has 6-DOF; translation in the x, y, and z direction and
rotation about the x, y, and z-axes.

"""

from __future__ import division, print_function
import numpy
numpy.random.seed(123)

from openmdao.api import Component, Group
from scipy.linalg import lu_factor, lu_solve

try:
    import OAS_API
    fortran_flag = True
except:
    fortran_flag = False


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
                     elem_IDs, cons,
                     E, G, x_gl, T,
                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                     const2, const_y, const_z, n, size, K, rhs):

    """
    Assemble the structural stiffness matrix based on 6 degrees of freedom
    per element.

    Can be run in dense Fortran or dense
    Python code depending on the flags used. Currently, dense Fortran
    seems to be the fastest version across many matrix sizes.

    """

    size = 6 * n + 6
    num_cons = 1

    num_elems = elem_IDs.shape[0]
    E_vec = E*numpy.ones(num_elems)
    G_vec = G*numpy.ones(num_elems)

    # Populate the right-hand side of the linear system using the
    # prescribed or computed loads
    rhs[:] = 0.0
    rhs[:6*n] = loads.reshape(n*6)
    rhs[numpy.abs(rhs) < 1e-6] = 0.

    # Dense Fortran
    if fortran_flag:
        K, x = OAS_API.oas_api.assemblestructmtx(nodes, A, J, Iy, Iz,
                                     K_a, K_t, K_y, K_z,
                                     elem_IDs+1, cons,
                                     E_vec, G_vec, x_gl, T,
                                     K_elem, S_a, S_t, S_y, S_z, T_elem,
                                     const2, const_y, const_z, loads)

    # Dense Python
    else:

        num_nodes = num_elems + 1

        K[:] = 0.

        # Loop over each element
        for ielem in xrange(num_elems):
            # Obtain the element nodes
            P0 = nodes[elem_IDs[ielem, 0], :]
            P1 = nodes[elem_IDs[ielem, 1], :]

            x_loc = unit(P1 - P0)
            y_loc = unit(numpy.cross(x_loc, x_gl))
            z_loc = unit(numpy.cross(x_loc, y_loc))

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            for ind in xrange(4):
                T_elem[3*ind:3*ind+3, 3*ind:3*ind+3] = T

            L = norm(P1 - P0)
            EA_L = E_vec[ielem] * A[ielem] / L
            GJ_L = G_vec[ielem] * J[ielem] / L
            EIy_L3 = E_vec[ielem] * Iy[ielem] / L**3
            EIz_L3 = E_vec[ielem] * Iz[ielem] / L**3

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

            in0, in1 = elem_IDs[ielem, :]

            # Populate the full matrix with stiffness
            # contributions from each node
            K[6*in0:6*in0+6, 6*in0:6*in0+6] += res[:6, :6]
            K[6*in1:6*in1+6, 6*in0:6*in0+6] += res[6:, :6]
            K[6*in0:6*in0+6, 6*in1:6*in1+6] += res[:6, 6:]
            K[6*in1:6*in1+6, 6*in1:6*in1+6] += res[6:, 6:]

        # Include a scaled identity matrix in the rows and columns
        # corresponding to the structural constraints
        for ind in xrange(num_cons):
            for k in xrange(6):
                K[-6+k, 6*cons+k] = 1.e9
                K[6*cons+k, -6+k] = 1.e9

    # Check to solve on the Python level if not done on the Fortran level
    if not fortran_flag:
        x = numpy.linalg.solve(K, rhs)

    return K, x, rhs


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

    def __init__(self, surface, cg_x=5):
        super(SpatialBeamFEM, self).__init__()

        self.surface = surface
        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.size = size = 6 * self.ny + 6

        self.add_param('A', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('Iy', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('Iz', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('J', val=numpy.zeros((self.ny - 1), dtype="complex"))
        self.add_param('nodes', val=numpy.zeros((self.ny, 3), dtype="complex"))
        self.add_param('loads', val=numpy.zeros((self.ny, 6), dtype="complex"))
        self.add_state('disp_aug', val=numpy.zeros((size), dtype="complex"))

        # self.deriv_options['type'] = 'cs'
        # self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"
        self.deriv_options['linearize'] = True  # only for FEM

        self.E = surface['E']
        self.G = surface['G']
        self.cg_x = cg_x

        elem_IDs = numpy.zeros((self.ny - 1, 2), int)
        arange = numpy.arange(self.ny-1)
        elem_IDs[:, 0] = arange
        elem_IDs[:, 1] = arange + 1
        self.elem_IDs = elem_IDs

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

        num_nodes = self.ny - 1
        num_cons = 1

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
        name = self.surface['name']

        # Find constrained nodes based on closeness to specified cg point
        nodes = params['nodes']
        dist = nodes - numpy.array([self.cg_x, 0, 0])
        idx = (numpy.linalg.norm(dist, axis=1)).argmin()
        self.cons = idx

        loads = params['loads']

        self.K, self.x, self.rhs = \
            _assemble_system(params['nodes'],
                             params['A'], params['J'], params['Iy'],
                             params['Iz'], loads, self.K_a, self.K_t,
                             self.K_y, self.K_z, self.elem_IDs, self.cons,
                             self.E, self.G, self.x_gl, self.T, self.K_elem,
                             self.S_a, self.S_t, self.S_y, self.S_z,
                             self.T_elem, self.const2, self.const_y,
                             self.const_z, self.ny, self.size,
                             self.K, self.rhs)

        unknowns['disp_aug'] = self.x


    def apply_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        loads = params['loads']
        self.K, _, self.rhs = \
            _assemble_system(params['nodes'],
                             params['A'], params['J'], params['Iy'],
                             params['Iz'], loads, self.K_a, self.K_t,
                             self.K_y, self.K_z, self.elem_IDs, self.cons,
                             self.E, self.G, self.x_gl, self.T, self.K_elem,
                             self.S_a, self.S_t, self.S_y, self.S_z,
                             self.T_elem, self.const2, self.const_y,
                             self.const_z, self.ny, self.size,
                             self.K, self.rhs)

        disp_aug = unknowns['disp_aug']
        resids['disp_aug'] = self.K.dot(disp_aug) - self.rhs

    def linearize(self, params, unknowns, resids):
        """ Jacobian for disp."""

        name = self.surface['name']
        jac = self.alloc_jacobian()
        if fortran_flag:
            fd_jac = self.fd_jacobian(params, unknowns, resids,
                                                fd_params=['A', 'Iy', 'Iz', 'J',
                                                           'nodes', 'loads'],
                                                fd_states=[])
            jac.update(fd_jac)
        else:
            cs_jac = self.complex_step_jacobian(params, unknowns, resids,
                                                fd_params=['A', 'Iy', 'Iz', 'J',
                                                           'nodes', 'loads'],
                                                fd_states=[])
            jac.update(cs_jac)
        jac['disp_aug', 'disp_aug'] = self.K.real

        self.lup = lu_factor(self.K.real)

        return jac

    def solve_linear(self, dumat, drmat, vois, mode=None):

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t = 0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t = 1

        for voi in vois:
            sol_vec[voi].vec[:] = \
                lu_solve(self.lup, rhs_vec[voi].vec, trans=t)


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

        # self.deriv_options['type'] = 'cs'

    def solve_nonlinear(self, params, unknowns, resids):
        # Obtain the relevant portions of disp_aug and store the reshaped
        # displacements in disp
        unknowns['disp'] = params['disp_aug'][:-6].reshape((self.ny, 6))

    def linearize(self, params, unknowns, resids):
        jac = self.alloc_jacobian()
        n = self.ny * 6
        jac['disp', 'disp_aug'] = numpy.hstack((numpy.eye((n)), numpy.zeros((n, 6))))
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

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']
        self.fem_origin = surface['fem_origin']

        self.add_param('mesh', val=numpy.zeros((self.nx, self.ny, 3), dtype=complex))
        self.add_output('nodes', val=numpy.zeros((self.ny, 3), dtype=complex))

    def solve_nonlinear(self, params, unknowns, resids):
        w = self.fem_origin
        name = self.surface['name']
        mesh = params['mesh']

        unknowns['nodes'] = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

    def linearize(self, params, unknowns, resids):
        jac = self.complex_step_jacobian(params, unknowns, resids,
                                            fd_params=['mesh'],
                                            fd_states=[])
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

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']

        self.add_param('disp', val=numpy.zeros((self.ny, 6), dtype=complex))
        self.add_param('loads', val=numpy.zeros((self.ny, 6), dtype=complex))
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
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('A', val=numpy.zeros((self.ny - 1), dtype=complex))
        self.add_param('nodes', val=numpy.zeros((self.ny, 3), dtype=complex))
        self.add_output('weight', val=0.)

        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'

        elem_IDs = numpy.zeros((self.ny - 1, 2), int)
        arange = numpy.arange(self.ny-1)
        elem_IDs[:, 0] = arange
        elem_IDs[:, 1] = arange + 1
        self.elem_IDs = elem_IDs

    def solve_nonlinear(self, params, unknowns, resids):
        A = params['A']
        nodes = params['nodes']
        num_elems = self.elem_IDs.shape[0]

        # Calculate the volume and weight of the total structure
        volume = 0.
        for ielem in xrange(num_elems):
            in0, in1 = self.elem_IDs[ielem, :]
            P0 = nodes[in0, :]
            P1 = nodes[in1, :]
            L = norm(P1 - P0)
            volume += L * A[ielem]

        weight = volume * self.surface['mrho'] * 9.81

        if self.surface['symmetry']:
            weight *= 2.

        unknowns['weight'] = weight

class SpatialBeamVonMisesTube(Component):
    """ Compute the max von Mises stress in each element.

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

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

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

        elem_IDs = numpy.zeros((self.ny - 1, 2), int)
        arange = numpy.arange(self.ny-1)
        elem_IDs[:, 0] = arange
        elem_IDs[:, 1] = arange + 1

        self.elem_IDs = elem_IDs
        self.E = surface['E']
        self.G = surface['G']

        self.T = numpy.zeros((3, 3), dtype='complex')
        self.x_gl = numpy.array([1, 0, 0], dtype='complex')
        self.t = 0

    def solve_nonlinear(self, params, unknowns, resids):
        elem_IDs = self.elem_IDs
        name = self.surface['name']
        r = params['r']
        disp = params['disp']
        nodes = params['nodes']
        vonmises = unknowns['vonmises']
        T = self.T
        E = self.E
        G = self.G
        x_gl = self.x_gl

        if fortran_flag:
            vm = OAS_API.oas_api.calc_vonmises(elem_IDs+1, nodes, r, disp, E, G, x_gl)
            unknowns['vonmises'] = vm

        else:

            num_elems = elem_IDs.shape[0]
            for ielem in xrange(num_elems):
                in0, in1 = elem_IDs[ielem, :]

                P0 = nodes[in0, :]
                P1 = nodes[in1, :]
                L = norm(P1 - P0)

                x_loc = unit(P1 - P0)
                y_loc = unit(numpy.cross(x_loc, x_gl))
                z_loc = unit(numpy.cross(x_loc, y_loc))

                T[0, :] = x_loc
                T[1, :] = y_loc
                T[2, :] = z_loc

                u0x, u0y, u0z = T.dot(disp[in0, :3])
                r0x, r0y, r0z = T.dot(disp[in0, 3:])
                u1x, u1y, u1z = T.dot(disp[in1, :3])
                r1x, r1y, r1z = T.dot(disp[in1, 3:])

                tmp = numpy.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
                sxx0 = E * (u1x - u0x) / L + E * r[ielem] / L * tmp
                sxx1 = E * (u0x - u1x) / L + E * r[ielem] / L * tmp
                sxt = G * r[ielem] * (r1x - r0x) / L

                vonmises[ielem, 0] = numpy.sqrt(sxx0**2 + sxt**2)
                vonmises[ielem, 1] = numpy.sqrt(sxx1**2 + sxt**2)

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        name = self.surface['name']
        elem_IDs = self.elem_IDs
        r = params['r'].real
        disp = params['disp'].real
        nodes = params['nodes'].real
        vonmises = unknowns['vonmises'].real
        E = self.E
        G = self.G
        x_gl = self.x_gl

        if mode == 'fwd':
            _, a = OAS_API.oas_api.calc_vonmises_d(elem_IDs+1, nodes, dparams['nodes'], r, dparams['r'], disp, dparams['disp'], E, G, x_gl)
            dresids['vonmises'] += a

        if mode == 'rev':
            a, b, c = OAS_API.oas_api.calc_vonmises_b(elem_IDs+1, nodes, r, disp, E, G, x_gl, vonmises, dresids['vonmises'])
            dparams['nodes'] += a
            dparams['r'] += b
            dparams['disp'] += c

        ### DOT PRODUCT TEST ###
        # nodesd = numpy.random.random_sample(nodes.shape)
        # rd = numpy.random.random_sample(r.shape)
        # dispd = numpy.random.random_sample(disp.shape)
        #
        # nodesd_copy = nodesd.copy()
        # rd_copy = rd.copy()
        # dispd_copy = dispd.copy()
        #
        # vonmises, vonmisesd = OAS_API.oas_api.calc_vonmises_d(elem_IDs+1, nodes, nodesd, r, rd, disp, dispd, E, G, x_gl)
        #
        # vonmisesb = numpy.random.random_sample(vonmises.shape)
        # vonmisesb_copy = vonmisesb.copy()
        #
        # nodesb, rb, dispb = OAS_API.oas_api.calc_vonmises_b(elem_IDs+1, nodes, r, disp, E, G, x_gl, vonmises, vonmisesb)
        #
        # dotprod = 0.
        # dotprod += numpy.sum(nodesd_copy*nodesb)
        # dotprod += numpy.sum(rd_copy*rb)
        # dotprod += numpy.sum(dispd_copy*dispb)
        # dotprod -= numpy.sum(vonmisesd*vonmisesb_copy)
        # print
        # print 'SHOULD BE ZERO:', dotprod
        # print

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

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('vonmises', val=numpy.zeros((self.ny-1, 2), dtype=complex))
        self.add_output('failure', val=0.)

        self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'

        self.sigma = surface['stress']
        self.rho = rho

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        sigma = self.sigma
        rho = self.rho
        vonmises = params['vonmises']

        fmax = numpy.max(vonmises/sigma - 1)

        nlog, nsum, nexp = numpy.log, numpy.sum, numpy.exp
        ks = 1 / rho * nlog(nsum(nexp(rho * (vonmises/sigma - 1 - fmax))))
        unknowns['failure'] = fmax + ks


class SpatialBeamStates(Group):
    """ Group that contains the spatial beam states. """

    def __init__(self, surface):
        super(SpatialBeamStates, self).__init__()

        self.add('nodes',
                 ComputeNodes(surface),
                 promotes=['*'])
        self.add('fem',
                 SpatialBeamFEM(surface),
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
