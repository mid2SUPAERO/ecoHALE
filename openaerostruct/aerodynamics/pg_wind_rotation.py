from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


class RotateToWindFrame(ExplicitComponent):
    """
    Rotate the VLM Geometry from the standard aerodynamic to the wind frame.
    In the wind frame the freestream will be along the x-axis.

    This transformation is given by the following rotation matrix:
         -      -     -                           -     -      -
        | x_wind |   | cosb*cosa, -sinb, cosb*sina |   | x_aero |
        | y_wind | = | sinb*cosa,  cosb, sinb*sina | . | y_aero |
        | z_wind |   | -sina,         0,      cosa |   | z_aero |
         -      -     -                           -     -       -

    Where "a" is the angle of attack and "b" is the sideslip angle.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in aero
        frame.
    bound_vecs[num_eval_points, 3] : numpy array
        The vectors representing the bound vortices for each panel in the
        problem.
        This array contains points for all lifting surfaces in the problem.
    coll_pts[num_eval_points, 3] : numpy array
        The xyz coordinates of the collocation points used in the VLM analysis.
        This array contains points for all lifting surfaces in the problem.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in aero frame, computed as the cross of
        the two diagonals from the mesh points.
    rotational_velocities[num_eval_points, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces.
        This array contains points for all lifting surfaces in the problem.
    alpha : float
        Angle of attack in degrees.
    beta : float
        Sideslip angle in degrees.

    Returns
    -------
    def_mesh_w_frame[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in wind
        frame.
    bound_vecs_w_frame[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices in wind frame.
    coll_pts_w_frame[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed in wind frame.
    normals_w_frame[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in wind frame.
    v_rot_w_frame[nx-1, ny-1, 3] : numpy array
        Velocity component at collecation points due to rotational velocity in
        wind frame.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        surfaces = self.options['surfaces']
        rotational = self.options['rotational']

        # Loop through all the surfaces to determine the total number
        # of evaluation points.
        num_eval_points = 0
        for surface in self.options['surfaces']:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            num_eval_points += (nx - 1) * (ny - 1)

        self.add_input('alpha', val=0., units='rad')
        self.add_input('beta', val=0., units='rad')
        self.add_input('coll_pts', shape=(num_eval_points, 3), units='m')
        self.add_input('bound_vecs', shape=(num_eval_points, 3), units='m')
        if rotational:
            self.add_input('rotational_velocities', shape=(num_eval_points, 3), units='m/s')

        self.add_output('coll_pts_w_frame', shape=(num_eval_points, 3), units='m')
        self.add_output('bound_vecs_w_frame', shape=(num_eval_points, 3), units='m')
        if rotational:
            self.add_output('rotational_velocities_w_frame', shape=(num_eval_points, 3), units='m/s')

        for surface in self.options['surfaces']:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface['name']

            mesh_name = '{}_def_mesh'.format(name)
            self.add_input(mesh_name, shape=(nx, ny, 3), units='m')

            normals_name = '{}_normals'.format(name)
            self.add_input(normals_name, shape=(nx - 1, ny - 1, 3))

            mesh_name = '{}_def_mesh_w_frame'.format(name)
            self.add_output(mesh_name, shape=(nx, ny, 3), units='m')

            normals_name = '{}_normals_w_frame'.format(name)
            self.add_output(normals_name, shape=(nx - 1, ny - 1, 3))

        # We'll compute all of sensitivities associated with angle of attack and
        # sideslip number through complex-step. Since it's a scalar this is
        # pretty cheap.
        self.declare_partials('*', 'alpha', method='cs')
        self.declare_partials('*', 'beta', method='cs')

    def compute(self, inputs, outputs):
        rotational = self.options['rotational']

        alpha = inputs['alpha']
        beta = inputs['beta']

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        # Define aero->wind rotation matrix
        Tw = np.array([[cosb*cosa, -sinb, cosb*sina],
                       [sinb*cosa,  cosb, sinb*sina],
                       [-sina,         0, cosa     ]], alpha.dtype)

        # Use einsum for fast vectorized matrix multiplication
        outputs['bound_vecs_w_frame'] = np.einsum('lk,jk->jl', Tw, inputs['bound_vecs'])
        outputs['coll_pts_w_frame'] = np.einsum('lk,jk->jl', Tw, inputs['coll_pts'])
        if rotational:
            outputs['rotational_velocities_w_frame'] = \
                np.einsum('lk,jk->jl', Tw, inputs['rotational_velocities'])

        for surface in self.options['surfaces']:
            name = surface['name']

            wrt_name = '{}_def_mesh'.format(name)
            of_name = '{}_def_mesh_w_frame'.format(name)
            outputs[of_name] = np.einsum('lk,ijk->ijl', Tw, inputs[wrt_name])

            wrt_name = '{}_normals'.format(name)
            of_name = '{}_normals_w_frame'.format(name)
            outputs[of_name] = np.einsum('lk,ijk->ijl', Tw, inputs[wrt_name])


class RotateFromWindFrame(ExplicitComponent):
    """
    Rotate the aerodynamic sectional and nodal force vectors from the wind to
    the standard aerodynamic frame. This is the reverse operation of the
    RotateToWindFrame component.

    This transformation is given by the following rotation matrix:
         -        -     -                           -     -        -
        | F_x_aero |   | cosb*cosa, sinb*cosa, -sina |   | F_x_wind |
        | F_y_aero | = | -sinb,          cosb,     0 | . | F_y_wind |
        | F_z_aero |   | cosb*sina, sinb*sina,  cosa |   | F_z_wind |
         -        -     -                           -     -        -

    Where "a" is the angle of attack and "b" is the sideslip angle.

    Parameters
    ----------
    sec_forces_w_frame[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in wind frame.
    node_forces_w_frame[nx, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in wind frame.

    alpha : float
        Angle of attack in degrees.
    beta : float
        Sideslip angle in degrees.

    Returns
    -------
    sec_forces[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in aero frame.
    node_forces[nx-1, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in aero frame.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        self.surface = surface = self.options['surface']

        [nx, ny, _] = surface['mesh'].shape
        name = surface['name']

        self.add_input('sec_forces_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_input('node_forces_w_frame', val=np.zeros((nx, ny, 3)), units='N')
        self.add_input('alpha', val=0., units='rad')
        self.add_input('beta', val=0., units='rad')

        self.add_output('sec_forces', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_output('node_forces', val=np.zeros((nx, ny, 3)), units='N')

        # We'll compute all of sensitivities associated with angle of attack and
        # sideslip number through complex-step. Since it's a scalar this is
        # pretty cheap.
        self.declare_partials('*', 'alpha', method='cs')
        self.declare_partials('*', 'beta', method='cs')

    def compute(self, inputs, outputs):

        alpha = inputs['alpha']
        beta = inputs['beta']
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        # Define aero->wind rotation matrix
        Tw = np.array([[cosb*cosa, -sinb, cosb*sina],
                       [sinb*cosa,  cosb, sinb*sina],
                       [-sina,         0, cosa     ]], alpha.dtype)
        # wind->aero rotation matrix is given by transpose

        # Use einsum for fast vectorized matrix multiplication
        outputs['sec_forces'] = np.einsum('lk,ijk->ijl', Tw.T, inputs['sec_forces_w_frame'])
        outputs['node_forces'] = np.einsum('lk,ijk->ijl', Tw.T, inputs['node_forces_w_frame'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        alpha = inputs['alpha']
        beta = inputs['beta']
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        Tw = np.array([[cosb*cosa, -sinb, cosb*sina],
                       [sinb*cosa,  cosb, sinb*sina],
                       [-sina,         0, cosa     ]], alpha.dtype)
        if mode == 'fwd':
            if 'sec_forces' in d_outputs and 'sec_forces_w_frame' in d_inputs:
                d_outputs['sec_forces'] += np.einsum('lk,ijk->ijl', Tw.T, d_inputs['sec_forces_w_frame'])
            if 'node_forces' in d_outputs and 'node_forces_w_frame' in d_inputs:
                d_outputs['node_forces'] += np.einsum('lk,ijk->ijl', Tw.T, d_inputs['node_forces_w_frame'])

        if mode == 'rev':
            if 'sec_forces' in d_outputs and 'sec_forces_w_frame' in d_inputs:
                d_inputs['sec_forces_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_outputs['sec_forces'])
            if 'node_forces' in d_outputs and 'node_forces_w_frame' in d_inputs:
                d_inputs['node_forces_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_outputs['node_forces'])