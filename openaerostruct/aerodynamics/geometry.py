from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


np.random.seed(314)

class VLMGeometry(ExplicitComponent):
    """ Compute various geometric properties for VLM analysis.

    parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    b_pts[nx-1, ny, 3] : numpy array
        Bound points for the horseshoe vortices, found along the 1/4 chord.
    c_pts[nx-1, ny-1, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed. Used to set up the linear system.
    widths[ny-1] : numpy array
        The spanwise widths of each individual panel.
    lengths[ny] : numpy array
        The chordwise length of the entire airfoil following the camber line.
    chords[ny] : numpy array
        The chordwise distance between the leading and trailing edges.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.
    S_ref : float
        The reference area of the lifting surface.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = ny = surface['num_y']
        self.nx = nx = surface['num_x']

        self.add_input('def_mesh', val=np.zeros((nx, ny, 3)), units='m')

        self.add_output('b_pts', val=np.random.random((nx-1, ny, 3)), units='m')
        self.add_output('c_pts', val=np.zeros((nx-1, ny-1, 3)), units='m')
        self.add_output('widths', val=np.ones((ny-1)), units='m')
        self.add_output('cos_sweep', val=np.zeros((ny-1)), units='m')
        self.add_output('lengths', val=np.zeros((ny)), units='m')
        self.add_output('chords', val=np.zeros((ny)), units='m')
        self.add_output('normals', val=np.zeros((nx-1, ny-1, 3)))
        self.add_output('S_ref', val=1., units='m**2')

        self.declare_partials('*', '*')

        size = (nx-1) * ny * 3
        base = np.arange(size)
        rows = np.tile(base, 2)
        cols = rows + np.repeat([0, ny*3], len(base))
        val = np.empty((2*size, ))
        val[:size] = 0.75
        val[size:] = 0.25

        self.declare_partials('b_pts', 'def_mesh', rows=rows, cols=cols, val=val)

        size = (nx-1) * (ny-1) * 3
        base = np.arange(size)
        rows = np.tile(base, 4)
        cbase = np.tile(base + np.repeat(3*np.arange(nx-1), 3*(ny-1)), 4)
        cols = cbase + np.repeat([0, 3, ny*3, (ny+1)*3], len(base))
        val = np.empty((4*size, ))
        val[:2*size] = 0.125
        val[2*size:] = 0.375

        self.declare_partials('c_pts', 'def_mesh', rows=rows, cols=cols, val=val)

        size = ny - 1
        base = np.arange(size)
        rows = np.tile(base, 12)
        col = np.tile(3*base, 6) + np.repeat(np.arange(6), len(base))
        cols = np.tile(col, 2) + np.repeat([0, (nx-1)*ny*3], len(col))

        self.declare_partials('widths', 'def_mesh', rows=rows, cols=cols)

        rows = np.tile(base, 8)
        col = np.tile(3*base, 4) + np.repeat([1, 2, 4, 5], len(base))
        cols = np.tile(col, 2) + np.repeat([0, (nx-1)*ny*3], len(col))

        self.declare_partials('cos_sweep', 'def_mesh', rows=rows, cols=cols)

        size = ny
        base = np.arange(size)
        rows = np.tile(base, nx * 3)
        col = np.tile(3*base, 3) + np.repeat(np.arange(3), len(base))
        cols = np.tile(col, nx) + np.repeat(3*ny*np.arange(nx), len(col))

        self.declare_partials('lengths', 'def_mesh', rows=rows, cols=cols)

        rows = np.tile(base, 6)
        col = np.tile(3*base, 3) + np.repeat(np.arange(3), len(base))
        cols = np.tile(col, 2) + np.repeat([0, (nx-1)*ny*3], len(col))

        self.declare_partials('chords', 'def_mesh', rows=rows, cols=cols)

        self.declare_partials('S_ref', 'def_mesh', method='cs')

    def compute(self, inputs, outputs):
        mesh = inputs['def_mesh']

        # Compute the bound points at quarter-chord
        b_pts = mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25

        # Compute the collocation points at the midpoints of each
        # panel's 3/4 chord line
        c_pts = 0.5 * 0.25 * mesh[:-1, :-1, :] + \
                0.5 * 0.75 * mesh[1:, :-1, :] + \
                0.5 * 0.25 * mesh[:-1,  1:, :] + \
                0.5 * 0.75 * mesh[1:,  1:, :]

        # Compute the widths of each panel at the quarter-chord line
        quarter_chord = 0.25 * mesh[-1] + 0.75 * mesh[0]
        widths = np.linalg.norm(quarter_chord[1:, :] - quarter_chord[:-1, :], axis=1)

        # Compute the numerator of the cosine of the sweep angle of each panel
        # (we need this for the viscous drag dependence on sweep, and we only compute
        # the numerator because the denominator of the cosine fraction is the width,
        # which we have already computed. They are combined in the viscous drag
        # calculation.)
        cos_sweep = np.linalg.norm(quarter_chord[1:, [1,2]] - quarter_chord[:-1, [1,2]], axis=1)

        # Compute the length of each chordwise set of mesh points through the camber line.
        dx = mesh[1:, :, 0] - mesh[:-1, :, 0]
        dy = mesh[1:, :, 1] - mesh[:-1, :, 1]
        dz = mesh[1:, :, 2] - mesh[:-1, :, 2]
        lengths = np.sum(np.sqrt(dx**2 + dy**2 + dz**2), axis=0)

        # Compute the normal of each panel by taking the cross-product of
        # its diagonals. Note that this could be a nonplanar surface
        normals = np.cross(
            mesh[:-1,  1:, :] - mesh[1:, :-1, :],
            mesh[:-1, :-1, :] - mesh[1:,  1:, :],
            axis=2)

        # Normalize the normal vectors
        norms = np.sqrt(np.sum(normals**2, axis=2))
        for j in range(3):
            normals[:, :, j] /= norms

        # Compute the wetted surface area
        if self.surface['S_ref_type'] == 'wetted':
            S_ref = 0.5 * np.sum(norms)

        # Compute the projected surface area
        elif self.surface['S_ref_type'] == 'projected':
            proj_mesh = mesh.copy()
            proj_mesh[: , :, 2] = 0.
            proj_normals = np.cross(
                proj_mesh[:-1,  1:, :] - proj_mesh[1:, :-1, :],
                proj_mesh[:-1, :-1, :] - proj_mesh[1:,  1:, :],
                axis=2)

            proj_norms = np.sqrt(np.sum(proj_normals**2, axis=2))
            for j in range(3):
                proj_normals[:, :, j] /= proj_norms

            S_ref = 0.5 * np.sum(proj_norms)

        # Multiply the surface area by 2 if symmetric to get consistent area measures
        if self.surface['symmetry']:
            S_ref *= 2

        # Compute the chord for each spanwise portion.
        # This is the distance from the leading to trailing edge.
        chords = np.linalg.norm(mesh[0, :, :] - mesh[-1, :, :], axis=1)

        # Store each array in the outputs dict
        outputs['b_pts'] = b_pts
        outputs['c_pts'] = c_pts
        outputs['widths'] = widths
        outputs['cos_sweep'] = cos_sweep
        outputs['lengths'] = lengths
        outputs['normals'] = normals
        outputs['S_ref'] = S_ref
        outputs['chords'] = chords

    def compute_partials(self, inputs, partials):
        """ Jacobian for VLM geometry."""

        nx = self.nx
        ny = self.ny
        mesh = inputs['def_mesh']

        # Compute the widths of each panel at the quarter-chord line
        quarter_chord = 0.25 * mesh[-1] + 0.75 * mesh[0]
        widths = np.linalg.norm(quarter_chord[1:, :] - quarter_chord[:-1, :], axis=1)

        # Compute the cosine of the sweep angle of each panel
        cos_sweep_array = np.linalg.norm(quarter_chord[1:, [1,2]] - quarter_chord[:-1, [1,2]], axis=1)

        delta = np.diff(quarter_chord, axis=0).T
        d1 = delta / widths
        partials['widths', 'def_mesh'] = np.outer([-0.75, 0.75, -0.25, 0.25],
                                                  d1.flatten()).flatten()
        d1 = delta[1:, :] / cos_sweep_array
        partials['cos_sweep', 'def_mesh'] = np.outer([-0.75, 0.75, -0.25, 0.25],
                                                     d1.flatten()).flatten()

        partials['lengths', 'def_mesh'][:] = 0.0
        dmesh = np.diff(mesh, axis=0)
        l = np.sqrt(np.sum(dmesh**2, axis=2))
        dmesh[:, :, 0] /= l
        dmesh[:, :, 1] /= l
        dmesh[:, :, 2] /= l
        derivs = np.transpose(dmesh, axes=[0, 2, 1]).flatten()
        nn = len(derivs)
        partials['lengths', 'def_mesh'][:nn] -= derivs
        partials['lengths', 'def_mesh'][-nn:] += derivs

        dfullmesh = mesh[0, :] - mesh[-1, :]
        l = np.sqrt(np.sum(dfullmesh**2, axis=1))
        derivs = (dfullmesh.T/l).flatten()
        partials['chords', 'def_mesh'] = np.concatenate([derivs, -derivs])

        partials['normals', 'def_mesh'] = np.zeros_like(partials['normals', 'def_mesh'])
        # Partial of f=normals w.r.t. to x=def_mesh
        #   f has shape (nx-1, ny-1, 3)
        #   x has shape (nx, ny, 3)
        for i in range(nx-1):
            for j in range(ny-1):
                # Redo original computation
                ll = mesh[i, j, :]      # leading-left node
                lr = mesh[i, j+1, :]    # leading-right node
                tl = mesh[i+1, j, :]    # trailing-left node
                tr = mesh[i+1, j+1, :]  # trailing-right node

                a = lr - tl
                b = ll - tr
                c = np.cross(a, b)
                n = np.sqrt(np.sum(c**2))
                # f = c / n

                # Now let's work backwards to get derivative
                # dfdc = (dcdc * n - c * dndc) / n**2
                dcdc = np.eye(3)
                dndc = c / n
                dfdc = (dcdc * n - np.einsum('i,j', c, dndc)) / n**2

                # dfdc is now a 3x3 jacobian with f along the rows and c along
                # the columns

                # The next step is to get dcda and dcdb, both of which will be
                # 3x3 jacobians with c along the rows
                dcda = np.array([[0, b[2], -b[1]],
                                [-b[2], 0, b[0]],
                                [b[1], -b[0], 0]])
                dcdb = np.array([[0, -a[2], a[1]],
                                [a[2], 0, -a[0]],
                                [-a[1], a[0], 0]])

                # Now let's do some matrix multiplication to get dfda and dfdb
                dfda = np.einsum('ij,jk->ik', dfdc, dcda)
                dfdb = np.einsum('ij,jk->ik', dfdc, dcdb)

                # Now we need to get dadlr, dadtl, dbdll, and dbdtr and put them
                # in the right indices of the big jacobian dfdx

                # These are the indices of the first and last components of f
                # for the current i and j
                if0 = (i*(ny-1)+j)*3
                if2 = (i*(ny-1)+j)*3+2

                # Partial f w.r.t. lr
                ix0 = (i*ny+j+1)*3      # First index of lr for current i and j
                ix2 = (i*ny+j+1)*3+2    # Last index of lr for current i and j
                partials['normals', 'def_mesh'][if0:if2+1,ix0:ix2+1] = dfda[:,:]

                # Partial f w.r.t. tl
                ix0 = ((i+1)*ny+j)*3    # First index of tl for current i and j
                ix2 = ((i+1)*ny+j)*3+2  # Last index of tl for current i and j
                partials['normals', 'def_mesh'][if0:if2+1,ix0:ix2+1] = -dfda[:,:]

                # Partial f w.r.t. ll
                ix0 = (i*ny+j)*3        # First index of ll for current i and j
                ix2 = (i*ny+j)*3+2      # Last index of ll for current i and j
                partials['normals', 'def_mesh'][if0:if2+1,ix0:ix2+1] = dfdb[:,:]

                # Partial f w.r.t. tr
                ix0 = ((i+1)*ny+j+1)*3   # First index of tr for current i and j
                ix2 = ((i+1)*ny+j+1)*3+2 # Last index of tr for current i and j
                partials['normals', 'def_mesh'][if0:if2+1,ix0:ix2+1] = -dfdb[:,:]
