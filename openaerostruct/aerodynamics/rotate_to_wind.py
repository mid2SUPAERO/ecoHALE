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
    bound_vecs[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord in
        aero frame.
    coll_pts[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed in aero frame. Used to set up the linear system.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in aero frame, computed as the cross of
        the two diagonals from the mesh points.
    v_rot[nx-1, ny-1, 3] : numpy array
        Velocity component at collecation points due to rotational velocity  in
        aero frame.

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

        # Loop through all the surfaces to determine the total number
        # of evaluation points.
        for surface in self.options['surfaces']:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            num_eval_points += (nx - 1) * (ny - 1)

        self.add_input('alpha', val=0., units='rad')
        self.add_input('beta', val=0., units='rad')
        self.add_input('coll_pts', shape=(num_eval_points, 3), units='m')
        self.add_input('bound_vecs', shape=(num_eval_points, 3), units='m')

        for surface in self.options['surfaces']:
            mesh = surface['mesh']
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface['name']

            # Take in a deformed mesh for each surface.
            mesh_name = '{}_def_mesh'.format(name)
            self.add_input(mesh_name, shape=(nx, ny, 3), units='m')

            normals_name = '{}_normals'.format(name)
            self.add_input(normals_name, shape=(nx - 1, ny - 1, 3))

            velocities_name = '{}_velocities'.format(eval_name)
            self.add_output(velocities_name, shape=(num_eval_points, 3), units='m/s')

        #self.add_input('def_mesh', val=np.zeros((nx, ny, 3)), units='m')
        #self.add_input('bound_vecs', val=np.zeros((nx-1, ny, 3)), units='m')
        #self.add_input('coll_pts', val=np.zeros((nx-1, ny-1, 3)), units='m')
        @self.add_input('normals', val=np.zeros((nx-1, ny-1, 3)))
        self.add_input('v_rot', val=np.zeros((nx-1, ny-1, 3)), units='m/s')

        self.add_output('def_mesh_w_frame', val=np.zeros((nx, ny, 3)), units='m')
        self.add_output('bound_vecs_w_frame', val=np.zeros((nx-1, ny, 3)), units='m')
        self.add_output('coll_pts_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='m')
        self.add_output('normals_w_frame', val=np.zeros((nx-1, ny-1, 3)))
        self.add_output('v_rot_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='m/s')

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

        # Use einsum for fast vectorized matrix multiplication
        outputs['def_mesh_w_frame'] = np.einsum('lk,ijk->ijl', Tw, inputs['def_mesh'])
        outputs['bound_vecs_w_frame'] = np.einsum('lk,ijk->ijl', Tw, inputs['bound_vecs'])
        outputs['coll_pts_w_frame'] = np.einsum('lk,ijk->ijl', Tw, inputs['coll_pts'])
        outputs['normals_w_frame'] = np.einsum('lk,ijk->ijl', Tw, inputs['normals'])
        outputs['v_rot_w_frame'] = np.einsum('lk,ijk->ijl', Tw, inputs['v_rot'])


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
            if 'def_mesh_w_frame' in d_outputs and 'def_mesh' in d_inputs:
                d_outputs['def_mesh_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_inputs['def_mesh'])
            if 'bound_vecs_w_frame' in d_outputs and 'bound_vecs' in d_inputs:
                d_outputs['bound_vecs_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_inputs['bound_vecs'])
            if 'coll_pts_w_frame' in d_outputs and 'coll_pts' in d_inputs:
                d_outputs['coll_pts_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_inputs['coll_pts'])
            if 'normals_w_frame' in d_outputs and 'normals' in d_inputs:
                d_outputs['normals_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_inputs['normals'])
            if 'v_rot_w_frame' in d_outputs and 'v_rot' in d_inputs:
                d_outputs['v_rot_w_frame'] += np.einsum('lk,ijk->ijl', Tw, d_inputs['v_rot'])

        if mode == 'rev':
            if 'def_mesh_w_frame' in d_outputs and 'def_mesh' in d_inputs:
                d_inputs['def_mesh'] += np.einsum('lk,ijk->ijl', Tw.T, d_outputs['def_mesh_w_frame'])
            if 'bound_vecs_w_frame' in d_outputs and 'bound_vecs' in d_inputs:
                d_inputs['bound_vecs'] += np.einsum('lk,ijk->ijl', Tw.T, d_outputs['bound_vecs_w_frame'])
            if 'coll_pts_w_frame' in d_outputs and 'coll_pts' in d_inputs:
                d_inputs['coll_pts'] += np.einsum('lk,ijk->ijl', Tw.T, d_outputs['coll_pts_w_frame'])
            if 'normals_w_frame' in d_outputs and 'normals' in d_inputs:
                d_inputs['normals'] += np.einsum('lk,ijk->ijl', Tw.T, d_outputs['normals_w_frame'])
            if 'v_rot_w_frame' in d_outputs and 'v_rot' in d_inputs:
                d_inputs['v_rot'] += np.einsum('lk,ijk->ijl', Tw.T, d_outputs['v_rot_w_frame'])
