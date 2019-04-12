from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


class ScaleToPrandtlGlauert(ExplicitComponent):
    """
    Scale the wind frame coordinates to get the Prandtl-Glauert transformed
    geometry.

    The Prandtl glauert transformation is defined as below:
        Coordinates
        -----------------------
        x_pg = x_wind
        y_pg = B*y_wind
        z_pg = B*z_wind

        Normals
        -----------------------
        n_x_pg = B*n_x_wind
        n_y_pg = n_y_wind
        n_z_pg = n_z_wind

        Perturbation velocities
        -----------------------
        v_x_pg = B^2*v_x_wind
        v_y_pg = B*v_y_wind
        v_z_pg = B*v_z_wind

    where B = sqrt(1 - M^2).
    Note: The freestream velocity remains untransformed and therefore is not
    included in this component.

    Parameters
    ----------
    def_mesh_w_frame[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in wind frame.
    bound_vecs_w_frame[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices in wind frame.
    coll_pts_w_frame[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed in wind frame.
    normals_w_frame[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in wind frame.
    v_rot_w_frame[nx-1, ny-1, 3] : numpy array
        Velocity component at collocation points due to rotational velocity in wind frame.

    M : float
        Freestream Mach number.

    Returns
    -------
    def_mesh_pg[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in PG frame.
    bound_vecs_pg[nx-1, ny, 3] : numpy array
        Bound points in PG frame.
    coll_pts_pg[nx-1, ny-1, 3] : numpy array
        Collocation points in PG frame.
    normals_pg[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in PG frame.
    v_rot_pg[nx-1, ny-1, 3] : numpy array
        Velocity component at collocation points due to rotational velocity in PG frame.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        self.surface = surface = self.options['surface']

        [nx, ny, _] = surface['mesh'].shape
        name = surface['name']

        self.add_input('def_mesh_w_frame', val=np.zeros((nx, ny, 3)), units='m')
        self.add_input('bound_vecs_w_frame', val=np.zeros((nx-1, ny, 3)), units='m')
        self.add_input('coll_pts_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='m')
        self.add_input('normals_w_frame', val=np.zeros((nx-1, ny-1, 3)))
        self.add_input('v_rot_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='m/s')

        self.add_output('def_mesh_pg', val=np.zeros((nx, ny, 3)), units='m')
        self.add_output('bound_vecs_pg', val=np.zeros((nx-1, ny, 3)), units='m')
        self.add_output('coll_pts_pg', val=np.zeros((nx-1, ny-1, 3)), units='m')
        self.add_output('normals_pg', val=np.zeros((nx-1, ny-1, 3)))
        self.add_output('v_rot_pg', val=np.zeros((nx-1, ny-1, 3)), units='m/s')

        self.add_input('MachNumber', val=0.)

        # We'll compute all of sensitivities associated with Mach number through
        # complex-step. Since it's a scalar this is pretty cheap.
        self.declare_partials('*', 'MachNumber', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['MachNumber']
        betaPG = np.sqrt(1 - M**2)

        outputs['def_mesh_pg'] = inputs['def_mesh_w_frame']
        outputs['def_mesh_pg'][:,:,1] *= betaPG
        outputs['def_mesh_pg'][:,:,2] *= betaPG

        outputs['bound_vecs_pg'] = inputs['bound_vecs_w_frame']
        outputs['bound_vecs_pg'][:,:,1] *= betaPG
        outputs['bound_vecs_pg'][:,:,2] *= betaPG

        outputs['coll_pts_pg'] = inputs['coll_pts_w_frame']
        outputs['coll_pts_pg'][:,:,1] *= betaPG
        outputs['coll_pts_pg'][:,:,2] *= betaPG

        outputs['normals_pg'] = inputs['normals_w_frame']
        outputs['normals_pg'][:,:,0] *= betaPG

        outputs['v_rot_pg'] = inputs['v_rot_w_frame']
        outputs['v_rot_pg'][:,:,0] *= betaPG**2
        outputs['v_rot_pg'][:,:,1] *= betaPG
        outputs['v_rot_pg'][:,:,2] *= betaPG

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        M = inputs['MachNumber']
        betaPG = np.sqrt(1 - M**2)

        if mode == 'fwd':
            if 'def_mesh_pg' in d_outputs and 'def_mesh_w_frame' in d_inputs:
                d_outputs['def_mesh_pg'][:,:,0] += 1.0 * d_inputs['def_mesh_w_frame'][:,:,0]
                d_outputs['def_mesh_pg'][:,:,1] += betaPG * d_inputs['def_mesh_w_frame'][:,:,1]
                d_outputs['def_mesh_pg'][:,:,2] += betaPG * d_inputs['def_mesh_w_frame'][:,:,2]

            if 'bound_vecs_pg' in d_outputs and 'bound_vecs_w_frame' in d_inputs:
                d_outputs['bound_vecs_pg'][:,:,0] += 1.0 * d_inputs['bound_vecs_w_frame'][:,:,0]
                d_outputs['bound_vecs_pg'][:,:,1] += betaPG * d_inputs['bound_vecs_w_frame'][:,:,1]
                d_outputs['bound_vecs_pg'][:,:,2] += betaPG * d_inputs['bound_vecs_w_frame'][:,:,2]

            if 'coll_pts_pg' in d_outputs and 'coll_pts_w_frame' in d_inputs:
                d_outputs['coll_pts_pg'][:,:,0] += 1.0 * d_inputs['coll_pts_w_frame'][:,:,0]
                d_outputs['coll_pts_pg'][:,:,1] += betaPG * d_inputs['coll_pts_w_frame'][:,:,1]
                d_outputs['coll_pts_pg'][:,:,2] += betaPG * d_inputs['coll_pts_w_frame'][:,:,2]

            if 'normals_pg' in d_outputs and 'normals_w_frame' in d_inputs:
                d_outputs['normals_pg'][:,:,0] += betaPG * d_inputs['normals_w_frame'][:,:,0]
                d_outputs['normals_pg'][:,:,1] += 1.0 * d_inputs['normals_w_frame'][:,:,1]
                d_outputs['normals_pg'][:,:,2] += 1.0 * d_inputs['normals_w_frame'][:,:,2]

            if 'v_rot_pg' in d_outputs and 'v_rot_w_frame' in d_inputs:
                d_outputs['v_rot_pg'][:,:,0] += betaPG**2 * d_inputs['v_rot_w_frame'][:,:,0]
                d_outputs['v_rot_pg'][:,:,1] += betaPG * d_inputs['v_rot_w_frame'][:,:,1]
                d_outputs['v_rot_pg'][:,:,2] += betaPG * d_inputs['v_rot_w_frame'][:,:,2]

        if mode == 'rev':
            if 'def_mesh_pg' in d_outputs and 'def_mesh_w_frame' in d_inputs:
                d_inputs['def_mesh_w_frame'][:,:,0] += 1.0 * d_outputs['def_mesh_pg'][:,:,0]
                d_inputs['def_mesh_w_frame'][:,:,1] += betaPG * d_outputs['def_mesh_pg'][:,:,1]
                d_inputs['def_mesh_w_frame'][:,:,2] += betaPG * d_outputs['def_mesh_pg'][:,:,2]

            if 'bound_vecs_pg' in d_outputs and 'bound_vecs_w_frame' in d_inputs:
                d_inputs['bound_vecs_w_frame'][:,:,0] += 1.0 * d_outputs['bound_vecs_pg'][:,:,0]
                d_inputs['bound_vecs_w_frame'][:,:,1] += betaPG * d_outputs['bound_vecs_pg'][:,:,1]
                d_inputs['bound_vecs_w_frame'][:,:,2] += betaPG * d_outputs['bound_vecs_pg'][:,:,2]

            if 'coll_pts_pg' in d_outputs and 'coll_pts_w_frame' in d_inputs:
                d_inputs['coll_pts_w_frame'][:,:,0] += 1.0 * d_outputs['coll_pts_pg'][:,:,0]
                d_inputs['coll_pts_w_frame'][:,:,1] += betaPG * d_outputs['coll_pts_pg'][:,:,1]
                d_inputs['coll_pts_w_frame'][:,:,2] += betaPG * d_outputs['coll_pts_pg'][:,:,2]

            if 'normals_pg' in d_outputs and 'normals_w_frame' in d_inputs:
                d_inputs['normals_w_frame'][:,:,0] += betaPG * d_outputs['normals_pg'][:,:,0]
                d_inputs['normals_w_frame'][:,:,1] += 1.0 * d_outputs['normals_pg'][:,:,1]
                d_inputs['normals_w_frame'][:,:,2] += 1.0 * d_outputs['normals_pg'][:,:,2]

            if 'v_rot_pg' in d_outputs and 'v_rot_w_frame' in d_inputs:
                d_inputs['v_rot_w_frame'][:,:,0] += betaPG**2 * d_outputs['v_rot_pg'][:,:,0]
                d_inputs['v_rot_w_frame'][:,:,1] += betaPG * d_outputs['v_rot_pg'][:,:,1]
                d_inputs['v_rot_w_frame'][:,:,2] += betaPG * d_outputs['v_rot_pg'][:,:,2]


class ScaleFromPrandtlGlauert(ExplicitComponent):
    """
    Scale the Prandtl-Glauert transformed forces to get the physical forces
    Prandtl-Glauert transformed geometry.

    The inverse Prandtl-Glauert transformation for forces is defined as below:
        F_x_wind = F_x_pg/B^4
        F_y_wind = F_y_pg/B^3
        F_z_wind = F_z_pg/B^3

    where B = sqrt(1 - M^2).

    Parameters
    ----------
    sec_forces_pg[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in PG domain.
    node_forces_pg[nx, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in PG domain.

    M : float
        Freestream Mach number.

    Returns
    -------
    sec_forces_w_frame[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in wind frame.
    node_forces_w_frame[nx, ny, 3] : numpy array
        Equivilent force vector on each panel (lattice) node in wind frame.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        self.surface = surface = self.options['surface']

        [nx, ny, _] = surface['mesh'].shape
        name = surface['name']

        self.add_input('sec_forces_pg', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_input('node_forces_pg', val=np.zeros((nx, ny, 3)), units='N')

        self.add_output('sec_forces_w_frame', val=np.zeros((nx-1, ny-1, 3)), units='N')
        self.add_output('node_forces_w_frame', val=np.zeros((nx, ny, 3)), units='N')

        self.add_input('MachNumber', val=0.)

        # We'll compute all of sensitivities associated with Mach number through
        # complex-step. Since it's a scalar this is pretty cheap.
        self.declare_partials('*', 'MachNumber', method='cs')

    def compute(self, inputs, outputs):
        M = inputs['MachNumber']
        betaPG = np.sqrt(1 - M**2)

        outputs['sec_forces_w_frame'] = inputs['sec_forces_pg']
        outputs['sec_forces_w_frame'][:,:,0] /= betaPG**4
        outputs['sec_forces_w_frame'][:,:,1] /= betaPG**3
        outputs['sec_forces_w_frame'][:,:,2] /= betaPG**3

        outputs['node_forces_w_frame'] = inputs['node_forces_pg']
        outputs['node_forces_w_frame'][:,:,0] /= betaPG**4
        outputs['node_forces_w_frame'][:,:,1] /= betaPG**3
        outputs['node_forces_w_frame'][:,:,2] /= betaPG**3

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        M = inputs['MachNumber']
        betaPG = np.sqrt(1 - M**2)

        if mode == 'fwd':
            if 'sec_forces_w_frame' in d_outputs and 'sec_forces_pg' in d_inputs:
                d_outputs['sec_forces_w_frame'][:,:,0] += d_inputs['sec_forces_pg'][:,:,0] / betaPG**4
                d_outputs['sec_forces_w_frame'][:,:,1] += d_inputs['sec_forces_pg'][:,:,1] / betaPG**3
                d_outputs['sec_forces_w_frame'][:,:,2] += d_inputs['sec_forces_pg'][:,:,2] / betaPG**3

            if 'node_forces_w_frame' in d_outputs and 'node_forces_pg' in d_inputs:
                d_outputs['node_forces_w_frame'][:,:,0] += d_inputs['node_forces_pg'][:,:,0] / betaPG**4
                d_outputs['node_forces_w_frame'][:,:,1] += d_inputs['node_forces_pg'][:,:,1] / betaPG**3
                d_outputs['node_forces_w_frame'][:,:,2] += d_inputs['node_forces_pg'][:,:,2] / betaPG**3

        if mode == 'rev':
            if 'sec_forces_w_frame' in d_outputs and 'sec_forces_pg' in d_inputs:
                d_inputs['sec_forces_pg'][:,:,0] += d_outputs['sec_forces_w_frame'][:,:,0] / betaPG**4
                d_inputs['sec_forces_pg'][:,:,1] += d_outputs['sec_forces_w_frame'][:,:,1] / betaPG**3
                d_inputs['sec_forces_pg'][:,:,2] += d_outputs['sec_forces_w_frame'][:,:,2] / betaPG**3

            if 'node_forces_w_frame' in d_outputs and 'node_forces_pg' in d_inputs:
                d_inputs['node_forces_pg'][:,:,0] += d_outputs['node_forces_w_frame'][:,:,0] / betaPG**4
                d_inputs['node_forces_pg'][:,:,1] += d_outputs['node_forces_w_frame'][:,:,1] / betaPG**3
                d_inputs['node_forces_pg'][:,:,2] += d_outputs['node_forces_w_frame'][:,:,2] / betaPG**3