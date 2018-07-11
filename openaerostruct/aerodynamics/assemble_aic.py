from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.aerodynamics.utils import _assemble_AIC_mtx, _assemble_AIC_mtx_b, \
    _assemble_AIC_mtx_d

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
except:
    fortran_flag = False

data_type = float


class AssembleAIC(ExplicitComponent):
    """
    Compute the circulations based on the AIC matrix and the panel velocities.
    Note that the flow tangency condition is enforced at the 3/4 chord point.
    There are multiple versions of the first four inputs with one
    for each surface defined.
    Each of these four inputs has the name of the surface prepended on the
    actual input name.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.
    b_pts[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.

    v : float
        Freestream air velocity in m/s.
    alpha : float
        Angle of attack in degrees.

    Returns
    -------
    AIC[(nx-1)*(ny-1), (nx-1)*(ny-1)] : numpy array
        The aerodynamic influence coefficient matrix. Solving the linear system
        of AIC * circulations = n * v gives us the circulations for each of the
        horseshoe vortices.
    rhs[(nx-1)*(ny-1)] : numpy array
        The right-hand-side of the linear system that yields the circulations.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        self.surfaces = surfaces = self.options['surfaces']

        tot_panels = 0
        for surface in surfaces:
            self.surface = surface
            ny = surface['num_y']
            nx = surface['num_x']
            name = surface['name']

            self.add_input(name + '_def_mesh', val=np.zeros((nx, ny, 3)), units='m')
            self.add_input(name + '_b_pts', val=np.zeros((nx-1, ny, 3)), units='m')
            self.add_input(name + '_c_pts', val=np.zeros((nx-1, ny-1, 3)), units='m')
            self.add_input(name + '_normals', val=np.zeros((nx-1, ny-1, 3)))
            tot_panels += (nx - 1) * (ny - 1)

        self.tot_panels = tot_panels

        self.add_input('v', val=1., units='m/s')
        self.add_input('alpha', val=1.)

        self.add_output('AIC', val=np.ones((tot_panels, tot_panels), dtype=data_type), units='1/m')
        self.add_output('rhs', val=np.ones((tot_panels), dtype=data_type), units='m/s')

        self.AIC_mtx = np.zeros((tot_panels, tot_panels, 3),
                                   dtype=data_type)
        self.mtx = np.zeros((tot_panels, tot_panels),
                                   dtype=data_type)

        self.declare_partials('*', '*')

        if not fortran_flag:
            self.declare_partials('*', '*', method='fd')

        # for surface in self.surfaces:
        #     name = surface['name']
        #     self.declare_partials('rhs', name + '_def_mesh', dependent=False)

    def compute(self, inputs, outputs):

        # Actually assemble the AIC matrix
        _assemble_AIC_mtx(self.AIC_mtx, inputs, self.surfaces)

        # Construct an flattened array with the normals of each surface in order
        # so we can do the normals with velocities to set up the right-hand-side
        # of the system.
        flattened_normals = np.zeros((self.tot_panels, 3), dtype=data_type)
        i = 0
        for surface in self.surfaces:
            name = surface['name']
            num_panels = (surface['num_x'] - 1) * (surface['num_y'] - 1)
            flattened_normals[i:i+num_panels, :] = inputs[name + '_normals'].reshape(-1, 3, order='F')
            i += num_panels

        # Construct a matrix that is the AIC_mtx dotted by the normals at each
        # collocation point. This is used to compute the circulations
        self.mtx[:, :] = 0.
        for ind in range(3):
            self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T *
                flattened_normals[:, ind]).T

        # Obtain the freestream velocity direction and magnitude by taking
        # alpha into account
        alpha = inputs['alpha'][0] * np.pi / 180.
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        v_inf = inputs['v'][0] * np.array([cosa, 0., sina], dtype=data_type)

        # Populate the right-hand side of the linear system with the
        # expected velocities at each collocation point
        outputs['rhs'] = -flattened_normals.\
            reshape(-1, flattened_normals.shape[-1], order='F').dot(v_inf)

        outputs['AIC'] = self.mtx


    if fortran_flag:
        if 0:
            def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
                if mode == 'fwd':

                    AIC_mtxd = np.zeros(self.AIC_mtx.shape)

                    # Actually assemble the AIC matrix
                    _assemble_AIC_mtx_d(AIC_mtxd, inputs, d_inputs, self.surfaces)

                    # Construct an flattened array with the normals of each surface in order
                    # so we can do the normals with velocities to set up the right-hand-side
                    # of the system.
                    flattened_normals = np.zeros((self.tot_panels, 3))
                    flattened_normalsd = np.zeros((self.tot_panels, 3))
                    i = 0
                    for surface in self.surfaces:
                        name = surface['name']
                        num_panels = (surface['num_x'] - 1) * (surface['num_y'] - 1)
                        flattened_normals[i:i+num_panels, :] = \
                            inputs[name + '_normals'].reshape(-1, 3, order='F')
                        if name+'_normals' in d_inputs:
                            flattened_normalsd[i:i+num_panels, :] = \
                                d_inputs[name + '_normals'].reshape(-1, 3, order='F')
                        else:
                            flattened_normalsd[i:i+num_panels, :] = 0.
                        i += num_panels

                    # Construct a matrix that is the AIC_mtx dotted by the normals at each
                    # collocation point. This is used to compute the circulations
                    self.mtx[:, :] = 0.
                    for ind in range(3):
                        self.mtx[:, :] += (AIC_mtxd[:, :, ind].T *
                            flattened_normals[:, ind]).T
                        self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T *
                            flattened_normalsd[:, ind]).T

                    # Obtain the freestream velocity direction and magnitude by taking
                    # alpha into account
                    alpha = inputs['alpha'][0] * np.pi / 180.
                    if 'alpha' in d_inputs:
                        alphad = d_inputs['alpha'][0] * np.pi / 180.
                    else:
                        alphad = 0.
                    cosa = np.cos(alpha)
                    sina = np.sin(alpha)
                    cosad = -sina * alphad
                    sinad = cosa * alphad

                    freestream_direction = np.array([cosa, 0., sina])
                    v_inf = inputs['v'][0] * freestream_direction
                    if 'v' in d_inputs:
                        v_infd = d_inputs['v'][0] * freestream_direction
                    else:
                        v_infd = np.zeros((3))
                    v_infd += inputs['v'][0] * np.array([cosad, 0., sinad])

                    # Populate the right-hand side of the linear system with the
                    # expected velocities at each collocation point
                    d_outputs['rhs'] = -flattened_normalsd.\
                        reshape(-1, 3, order='F').dot(v_inf)
                    d_outputs['rhs'] += -flattened_normals.\
                        reshape(-1, 3, order='F').dot(v_infd)

                    d_outputs['AIC'] = self.mtx

                if mode == 'rev':

                    # Construct an flattened array with the normals of each surface in order
                    # so we can do the normals with velocities to set up the right-hand-side
                    # of the system.
                    flattened_normals = np.zeros((self.tot_panels, 3))
                    i = 0
                    for surface in self.surfaces:
                        name = surface['name']
                        num_panels = (surface['num_x'] - 1) * (surface['num_y'] - 1)
                        flattened_normals[i:i+num_panels, :] = \
                            inputs[name + '_normals'].reshape(-1, 3, order='F')
                        i += num_panels

                    AIC_mtxb = np.zeros((self.tot_panels, self.tot_panels, 3))
                    flattened_normalsb = np.zeros(flattened_normals.shape)
                    for ind in range(3):
                        AIC_mtxb[:, :, ind] = (d_outputs['AIC'].T * flattened_normals[:, ind]).T
                        flattened_normalsb[:, ind] += \
                            np.sum(self.AIC_mtx[:, :, ind].real * d_outputs['AIC'], axis=1).T

                    # Actually assemble the AIC matrix
                    _assemble_AIC_mtx_b(AIC_mtxb, inputs, d_inputs, self.surfaces)

                    # Obtain the freestream velocity direction and magnitude by taking
                    # alpha into account
                    alpha = inputs['alpha'][0] * np.pi / 180.
                    cosa = np.cos(alpha)
                    sina = np.sin(alpha)
                    arr = np.array([cosa, 0., sina])
                    v_inf = inputs['v'][0] * arr

                    fn = flattened_normals
                    fnb = np.zeros(fn.shape)
                    rhsb = d_outputs['rhs']

                    v_infb = 0.
                    for ind in reversed(range(self.tot_panels)):
                        fnb[ind, :] -= v_inf * rhsb[ind]
                        v_infb -= fn[ind, :] * rhsb[ind]

                    if 'v' in d_inputs:
                        d_inputs['v'] += sum(arr * v_infb)
                    arrb = inputs['v'] * v_infb
                    alphab = np.cos(alpha) * arrb[2]
                    alphab -= np.sin(alpha) * arrb[0]
                    alphab *= np.pi / 180.

                    if 'alpha' in d_inputs:
                        d_inputs['alpha'] += alphab

                    i = 0
                    for surface in self.surfaces:
                        name = surface['name']
                        nx = surface['num_x']
                        ny = surface['num_y']
                        num_panels = (nx - 1) * (ny - 1)
                        if name+'_normals' in d_inputs:
                            d_inputs[name + '_normals'] += \
                                flattened_normalsb[i:i+num_panels, :].reshape(nx-1, ny-1, 3, order='F')
                            d_inputs[name + '_normals'] += \
                                fnb[i:i+num_panels, :].reshape(nx-1, ny-1, 3, order='F')
                        i += num_panels

        else:
            def compute_partials(self, inputs, partials):

                for param in inputs:

                    d_inputs = {}
                    d_inputs[param] = inputs[param].copy()
                    d_outputs = {}

                    if isinstance(d_inputs[param], np.ndarray):
                        for j, val in enumerate(np.array(d_inputs[param]).flatten()):
                            d_in_b = np.array(d_inputs[param]).flatten()
                            d_in_b[:] = 0.
                            d_in_b[j] = 1.
                            d_inputs[param] = d_in_b.reshape(d_inputs[param].shape)

                            AIC_mtxd = np.zeros(self.AIC_mtx.shape)

                            # Actually assemble the AIC matrix
                            _assemble_AIC_mtx_d(AIC_mtxd, inputs, d_inputs, self.surfaces)

                            # Construct an flattened array with the normals of each surface in order
                            # so we can do the normals with velocities to set up the right-hand-side
                            # of the system.
                            flattened_normals = np.zeros((self.tot_panels, 3))
                            flattened_normalsd = np.zeros((self.tot_panels, 3))
                            i = 0
                            for surface in self.surfaces:
                                name = surface['name']
                                num_panels = (surface['num_x'] - 1) * (surface['num_y'] - 1)
                                flattened_normals[i:i+num_panels, :] = \
                                    inputs[name + '_normals'].reshape(-1, 3, order='F')
                                if name+'_normals' in d_inputs:
                                    flattened_normalsd[i:i+num_panels, :] = \
                                        d_inputs[name + '_normals'].reshape(-1, 3, order='F')
                                else:
                                    flattened_normalsd[i:i+num_panels, :] = 0.
                                i += num_panels

                            # Construct a matrix that is the AIC_mtx dotted by the normals at each
                            # collocation point. This is used to compute the circulations
                            self.mtx[:, :] = 0.
                            for ind in range(3):
                                self.mtx[:, :] += (AIC_mtxd[:, :, ind].T *
                                    flattened_normals[:, ind]).T
                                self.mtx[:, :] += (self.AIC_mtx[:, :, ind].T *
                                    flattened_normalsd[:, ind]).T

                            # Obtain the freestream velocity direction and magnitude by taking
                            # alpha into account
                            alpha = inputs['alpha'][0] * np.pi / 180.
                            if 'alpha' in d_inputs:
                                alphad = d_inputs['alpha'][0] * np.pi / 180.
                            else:
                                alphad = 0.
                            cosa = np.cos(alpha)
                            sina = np.sin(alpha)
                            cosad = -sina * alphad
                            sinad = cosa * alphad

                            freestream_direction = np.array([cosa, 0., sina])
                            v_inf = inputs['v'][0] * freestream_direction
                            if 'v' in d_inputs:
                                v_infd = d_inputs['v'][0] * freestream_direction
                            else:
                                v_infd = np.zeros((3))
                            v_infd += inputs['v'][0] * np.array([cosad, 0., sinad])

                            # Populate the right-hand side of the linear system with the
                            # expected velocities at each collocation point
                            d_outputs['rhs'] = -flattened_normalsd.\
                                reshape(-1, 3, order='F').dot(v_inf)
                            d_outputs['rhs'] += -flattened_normals.\
                                reshape(-1, 3, order='F').dot(v_infd)

                            d_outputs['AIC'] = self.mtx

                            partials['AIC', param][:, j] = d_outputs['AIC'].flatten()
                            if 'mesh' not in param:
                                partials['rhs', param][:, j] = d_outputs['rhs'].flatten()

                    else:
                        d_inputs[param] = 1.

                        partials['AIC', param] = d_outputs['AIC'].flatten()
                        partials['rhs', param] = d_outputs['rhs'].flatten()
