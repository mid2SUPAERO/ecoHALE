from __future__ import division, print_function

import numpy as np
from scipy.sparse import coo_matrix, identity

from openmdao.api import ExplicitComponent


def _skew(vector):
    out = np.array([[0, -vector[2], vector[1]],\
                    [vector[2], 0, -vector[0]],\
                    [-vector[1], vector[0], 0]])
    return out


class LoadTransfer(ExplicitComponent):
    """
    Perform aerodynamic load transfer.

    Apply the computed sectional forces on the aerodynamic surfaces to
    obtain the deformed mesh FEM loads.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Flattened array defining the lifting surfaces after deformation.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant when more than one chordwise
        panel).

    Returns
    -------
    LoadsA[ny,3] : numpy array  = loads[:,:3]
    LoadsB[ny,3]: nympy array   = loads[:,3:]
        Flattened array containing the loads applied on the FEM component,
        computed from the sectional forces.

    Loads[ny, 6] : numpy array = [LoadsA,LoadsB]

    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.nx = nx = surface['mesh'].shape[0]
        self.ny = ny = surface['mesh'].shape[1]


        if surface['fem_model_type'] == 'tube':
            self.fem_origin = surface['fem_origin']
        else:
            y_upper = surface['data_y_upper']
            x_upper = surface['data_x_upper']
            y_lower = surface['data_y_lower']

            fem_origin = (x_upper[0]  * (y_upper[0]  - y_lower[0]) +
                          x_upper[-1] * (y_upper[-1] - y_lower[-1])) / \
                        ((y_upper[0]  -  y_lower[0]) + (y_upper[-1] - y_lower[-1]))

            # For some reason, surface data is complex in some tests.
            self.fem_origin = np.float(fem_origin)

        self.w1 = w1 = 0.25
        self.w2 = w2 = self.fem_origin

        self.add_input('def_mesh', val=np.zeros((nx, ny, 3)), units='m')
        self.add_input('sec_forces', val=np.zeros((nx-1, ny-1, 3)), units='N')

        self.add_output('loads', val=np.zeros((self.ny, 6)), units='N') ## WARNING!!! UNITS ARE A MIXTURE OF N & N*m
        # Well, technically the units of this load array are mixed.
        # The first 3 indices are N and the last 3 are N*m.

        # Derivatives

        # First, the direct loads wrt sec_forces terms.
        base_row = np.array([0, 1, 2, 6, 7, 8])
        base_col = np.array([0, 1, 2, 0, 1, 2])
        row = np.tile(base_row, ny-1) + np.repeat(6*np.arange(ny-1), 6)
        col = np.tile(base_col, ny-1) + np.repeat(3*np.arange(ny-1), 6)
        rows1 = np.tile(row, nx-1)
        cols1 = np.tile(col, nx-1) + np.repeat(3*(ny-1)*np.arange(nx-1), 6*(ny-1))

        # Then, the term from the cross product.
        base_row = np.array([3, 3, 3, 4, 4, 4, 5, 5, 5])
        base_col = np.tile(np.array([0, 1, 2]), 3)
        row = np.tile(base_row, ny-1) + np.repeat(6*np.arange(ny-1), 9)
        col = np.tile(base_col, ny-1) + np.repeat(3*np.arange(ny-1), 9)
        row1 = np.tile(row, nx-1)
        col1 = np.tile(col, nx-1) + np.repeat(3*(ny-1)*np.arange(nx-1), 9*(ny-1))
        rows2 = np.tile(row1, 2) + np.repeat(np.array([0, 6]), 9*(nx-1)*(ny-1))
        cols2 = np.tile(col1, 2)

        rows = np.concatenate([rows1, rows2])
        cols = np.concatenate([cols1, cols2])

        self.declare_partials(of='loads', wrt='sec_forces', rows=rows, cols=cols)

        # ------- OLD ---------
        # Below we create each individual partial derivative


        # --------------------------------dloadsB__dsec_forces-------------------------------------
        #dloadsB__dmoment
        rowsA=np.zeros((3*(ny-1)*2))
        colsA=np.zeros((3*(ny-1)*2))
        data=0.5*np.ones((3*(ny-1)*2))
        for col_idx in range(3*(ny-1)):
            rowsA[col_idx*2] = col_idx
            rowsA[col_idx*2+1] = col_idx+3
            colsA[col_idx*2] = col_idx
            colsA[col_idx*2+1] = col_idx
        self.dloadsB__dmoment = coo_matrix((data, (rowsA, colsA)), shape=(3*ny,3*(ny-1)))

        # dmoment__dsec_forces
        diff = np.ones((nx-1, ny-1, 3))
        rows = np.zeros((9*(ny-1)*(nx-1)))
        cols = np.zeros((9*(ny-1)*(nx-1)))
        data = np.ones((9*(ny-1)*(nx-1)))
        tmp_row = np.array([0,0,0,1,1,1,2,2,2])
        tmp_col = np.array([0,1,2,0,1,2,0,1,2])
        # setup rows and cols
        for nxitr in range(nx-1):
            for nyitr in range(ny-1):
                index = 9*(nyitr+(ny-1)*nxitr)
                rows[index:index+9] = tmp_row+nyitr*3.
                cols[index:index+9] = tmp_col+nyitr*3.+3*(ny-1)*nxitr
                data[index:index+9] = -_skew(diff[nxitr,nyitr,:]).flatten()
        dmoment__dsec_forces = coo_matrix((data, (rows,cols)), shape=(3*(ny-1), 3*(ny-1)*(nx-1)))

        # dloadsB__dsec_forces
        dloadsB__dsec_forces = self.dloadsB__dmoment * dmoment__dsec_forces
        dloadsB__dsec_forces = dloadsB__dsec_forces.tocoo() # force the system to be a coo_matrix

        # --------------------------------dloadsB__ddef_mesh-------------------------------------
        # setup sparce partials for dloadsB__ddef_mesh where:

        # dloadsB__dmoment
        rows = np.zeros((3*(ny-1)*2))
        cols = np.zeros((3*(ny-1)*2))
        data = np.ones((3*(ny-1)*2))
        tmp_row = np.array([0,3])
        tmp_col = np.array([0,0])
        for i in range(3*(ny-1)):
            rows[i*2:(1+i)*2] = i+tmp_row
            cols[i*2:(1+i)*2] = i+tmp_col
        self.dloadsB__dmoment = 0.5*coo_matrix((data, (rows,cols)), shape=(3*ny, 3*(ny-1)))

        # dmoment__ddiff
        # this will need to be recalculated in def_partials b/c it's a function of sec_forces
        rows = np.zeros((9*(ny-1)*(nx-1)))
        cols = np.zeros((9*(ny-1)*(nx-1)))
        data = np.zeros((9*(ny-1)*(nx-1)))
        tmp_row = np.array([0,0,0,1,1,1,2,2,2])
        tmp_col = np.array([0,1,2,0,1,2,0,1,2])
        tmp_sec_forces = np.ones((nx-1,ny-1,3))
        # setup rows and cols
        for nxitr in range(nx-1):
            for nyitr in range(ny-1):
                index = 9*(nyitr+(ny-1)*nxitr)
                rows[index:index+9] = tmp_row+nyitr*3.
                cols[index:index+9] = tmp_col+nyitr*3.+3*(ny-1)*nxitr
                data[index:index+9] = -_skew(tmp_sec_forces[nxitr,nyitr,:]).flatten()
        dmoment__ddiff = coo_matrix((data, (rows,cols)), shape=(3*(ny-1), 3*(ny-1)*(nx-1)))

        # ddiff__ds_pts
        tmp_ddiff__ds_pts = np.zeros((3*(ny-1)*(nx-1),3*(ny-1)))
        for xiter in range(nx-1):
            tmp_ddiff__ds_pts[xiter*3*(ny-1):(1+xiter)*3*(ny-1),0:3*(ny-1)] = np.eye((3*(ny-1)))
        ddiff__ds_pts = coo_matrix(tmp_ddiff__ds_pts)

        # ds_pts__ddef_mesh
        tmp_ds_pts__ddef_mesh = np.zeros((3*(ny-1),3*(ny)*(nx)))
        block = np.zeros((3*(ny-1),3*ny))
        block[:,0:-3] = np.eye((3*(ny-1)))
        block[:,3:] = block[:,3:] + np.eye((3*(ny-1)))
        tmp_ds_pts__ddef_mesh[:,0:3*ny] = 0.5*(1-w2)*block
        tmp_ds_pts__ddef_mesh[:,3*ny*(nx-1):] = tmp_ds_pts__ddef_mesh[:,3*ny*(nx-1):] + 0.5 * w2 * block
        ds_pts__ddef_mesh = coo_matrix(tmp_ds_pts__ddef_mesh)

        # ddiff__da_pts
        ddiff__da_pts = identity((3*(ny-1)*(nx-1)))

        # da_pts__ddef_mesh
        tmp_da_pts__ddef_mesh = np.zeros((3*(ny-1)*(nx-1),3*ny*nx))
        block2 = np.zeros((3*(ny-1),6*ny))
        block2[:,:-3*ny] = 0.5 * (1-w1) * block
        block2[:,3*ny:] = 0.5 * w1 * block
        down = 3*(ny-1)
        over = 3*ny
        for xiter in range(nx-1):
            tmp_da_pts__ddef_mesh[xiter*down:xiter*down+3*(ny-1),over*xiter:over*xiter+6*ny] = block2
        da_pts__ddef_mesh = coo_matrix(tmp_da_pts__ddef_mesh)

        # to shorten future calculation times we calculate two other partials
        # ddiff__ddef_mesh
        self.ddiff__ddef_mesh = ddiff__da_pts * da_pts__ddef_mesh - ddiff__ds_pts * ds_pts__ddef_mesh
        self.ddiff__ddef_mesh = self.ddiff__ddef_mesh.tocoo()

        # dloadsB__ddef_mesh
        dloadsB__ddef_mesh = self.dloadsB__dmoment * dmoment__ddiff * self.ddiff__ddef_mesh
        dloadsB__ddef_mesh = dloadsB__ddef_mesh.tocoo()

        #---------------------------------dloads__ddef_mesh
        #dloads__dloadsA
        blk = np.array([0,1,2])
        rows = np.zeros((ny*3))
        data = np.ones((ny*3))
        cols = np.linspace(0,ny*3-1, ny*3)
        for yiter in range(ny):
            rows[yiter*3:(1+yiter)*3] = blk+6*yiter

        #dloads__dloadsB
        rows = rows + 3
        self.dloads__dloadsB = coo_matrix((data, (rows, cols)), shape=(ny*6, ny*3))

        dloads__ddef_mesh = self.dloads__dloadsB * dloadsB__ddef_mesh
        dloads__ddef_mesh = dloads__ddef_mesh.tocoo()
        rows = dloads__ddef_mesh.row
        cols = dloads__ddef_mesh.col

        self.declare_partials(of='loads', wrt='def_mesh', rows=rows, cols=cols)

        # -------------------------------- Check Partial Options-------------------------------------
        self.set_check_partial_options('*', method='cs', step=1e-40)

    def compute(self, inputs, outputs):
        mesh = inputs['def_mesh'] #[nx, ny, 3]
        sec_forces = inputs['sec_forces']

        # Compute the aerodynamic centers at the quarter-chord point of each panel
        # a_pts [nx-1, ny-1, 3]
        a_pts = 0.5 * (1-self.w1) * mesh[:-1, :-1, :] + \
                0.5 *   self.w1   * mesh[1:, :-1, :] + \
                0.5 * (1-self.w1) * mesh[:-1,  1:, :] + \
                0.5 *   self.w1   * mesh[1:,  1:, :]

        # Compute the structural midpoints based on the fem_origin location
        # s_pts [ny-1, 3]
        s_pts = 0.5 * (1-self.w2) * mesh[0, :-1, :] + \
                0.5 *   self.w2   * mesh[-1, :-1, :] + \
                0.5 * (1-self.w2) * mesh[0,  1:, :] + \
                0.5 *   self.w2   * mesh[-1,  1:, :]

        # Find the moment arm between the aerodynamic centers of each panel
        # and the FEM elements
        # diff [nx-1, ny-1, 3]
        moment = 0.5 * np.sum(np.cross(a_pts - s_pts, sec_forces), axis=0)

        # Only need to zero out the part that is assigned via +=
        outputs['loads'][-1, :] = 0.

        # Compute the loads based on the xyz forces and the computed moments
        sec_forces_sum = 0.5 * np.sum(sec_forces, axis=0)
        outputs['loads'][:-1, :3] = sec_forces_sum
        outputs['loads'][1:, :3] += sec_forces_sum

        outputs['loads'][:-1, 3:] = moment
        outputs['loads'][1:, 3:] += moment

    def compute_partials(self, inputs, partials):
        mesh = inputs['def_mesh']
        sec_forces = inputs['sec_forces']
        ny = self.ny
        nx = self.nx

        # -------------------------------dloadsB__dsec_forces--------------------------------------
        # dmoment__dsec_forces
        # Compute the aerodynamic centers at the quarter-chord point of each panel
        a_pts = 0.5 * (1-self.w1) * mesh[:-1, :-1, :] + \
                0.5 *   self.w1   * mesh[1:, :-1, :] + \
                0.5 * (1-self.w1) * mesh[:-1,  1:, :] + \
                0.5 *   self.w1   * mesh[1:,  1:, :]

        # Compute the structural midpoints based on the fem_origin location
        s_pts = 0.5 * (1-self.w2) * mesh[0, :-1, :] + \
                0.5 *   self.w2   * mesh[-1, :-1, :] + \
                0.5 * (1-self.w2) * mesh[0,  1:, :] + \
                0.5 *   self.w2   * mesh[-1,  1:, :]

        # Derivatives across the cross product.
        diff = a_pts - s_pts

        dmom_dsec = np.zeros((nx-1, ny-1, 3, 3))
        dmom_dsec[:, :, 0, 1] = -diff[:, :, 2] * 0.5
        dmom_dsec[:, :, 0, 2] = diff[:, :, 1] * 0.5
        dmom_dsec[:, :, 1, 0] = diff[:, :, 2] * 0.5
        dmom_dsec[:, :, 1, 2] = -diff[:, :, 0] * 0.5
        dmom_dsec[:, :, 2, 0] = -diff[:, :, 1] * 0.5
        dmom_dsec[:, :, 2, 1] = diff[:, :, 0] * 0.5

        n1 = 6*(ny-1)*(nx-1)
        partials['loads', 'sec_forces'][:n1] = 0.5

        n2 = n1 + n1*3//2
        dmom_dsec = dmom_dsec.flatten()
        partials['loads', 'sec_forces'][n1:n2] = dmom_dsec
        partials['loads', 'sec_forces'][n2:] = dmom_dsec

        rows = np.zeros((9*(ny-1)*(nx-1)))
        cols = np.zeros((9*(ny-1)*(nx-1)))
        data = np.zeros((9*(ny-1)*(nx-1)))
        tmp_row = np.array([0,0,0,1,1,1,2,2,2])
        tmp_col = np.array([0,1,2,0,1,2,0,1,2])
        # setup rows and cols
        for nxitr in range(nx-1):
            for nyitr in range(ny-1):
                index = 9*(nyitr+(ny-1)*nxitr)
                rows[index:index+9] = tmp_row+nyitr*3.
                cols[index:index+9] = tmp_col+nyitr*3.+3*(ny-1)*nxitr
                data[index:index+9] = _skew(diff[nxitr,nyitr,:]).flatten()
        dmoment__dsec_forces = coo_matrix((data, (rows,cols)), shape=(3*(ny-1), 3*(ny-1)*(nx-1)))

        dloadsB__dsec_forces = self.dloadsB__dmoment * dmoment__dsec_forces
        dloadsB__dsec_forces = dloadsB__dsec_forces.tocoo()

        # --------------------------------dloadsB__ddef_mesh-------------------------------------
        # dmoment__ddiff must to be calulated each time b/c it's a function of sec_forces
        rows = np.zeros((9*(ny-1)*(nx-1)))
        cols = np.zeros((9*(ny-1)*(nx-1)))
        data = np.zeros((9*(ny-1)*(nx-1)))
        tmp_row = np.array([0,0,0,1,1,1,2,2,2])
        tmp_col = np.array([0,1,2,0,1,2,0,1,2])
        # setup rows and cols
        for nxitr in range(nx-1):
            for nyitr in range(ny-1):
                index = 9*(nyitr+(ny-1)*nxitr)
                rows[index:index+9] = tmp_row+nyitr*3.
                cols[index:index+9] = tmp_col+nyitr*3.+3*(ny-1)*nxitr
                data[index:index+9] = -_skew(sec_forces[nxitr,nyitr,:]).flatten()
        dmoment__ddiff = coo_matrix((data, (rows,cols)), shape=(3*(ny-1), 3*(ny-1)*(nx-1)))


        # multiply the matrix with those previous calculated in setup
        dloadsB__ddef_mesh = self.dloadsB__dmoment * dmoment__ddiff * self.ddiff__ddef_mesh
        dloadsB__ddef_mesh = dloadsB__ddef_mesh.tocoo()

        dloads__ddef_mesh = self.dloads__dloadsB * dloadsB__ddef_mesh
        dloads__ddef_mesh = dloads__ddef_mesh.tocoo()
        partials['loads','def_mesh'] = dloads__ddef_mesh.data

