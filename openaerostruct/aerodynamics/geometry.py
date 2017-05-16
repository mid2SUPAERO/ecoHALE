from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

try:
    import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

class VLMGeometry(Component):
    """ Compute various geometric properties for VLM analysis.

    Parameters
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

    def __init__(self, surface):
        super(VLMGeometry, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']

        self.add_param('def_mesh', val=np.zeros((self.nx, self.ny, 3),
                       dtype=data_type))
        self.add_output('b_pts', val=np.zeros((self.nx-1, self.ny, 3),
                        dtype=data_type))
        self.add_output('c_pts', val=np.zeros((self.nx-1, self.ny-1, 3)))
        self.add_output('widths', val=np.zeros((self.ny-1)))
        self.add_output('cos_sweep', val=np.zeros((self.ny-1)))
        self.add_output('lengths', val=np.zeros((self.ny)))
        self.add_output('chords', val=np.zeros((self.ny)))
        self.add_output('normals', val=np.zeros((self.nx-1, self.ny-1, 3)))
        self.add_output('S_ref', val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        mesh = params['def_mesh']

        # Compute the bound points at quart-chord
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
            for j in xrange(3):
                proj_normals[:, :, j] /= proj_norms

            S_ref = 0.5 * np.sum(proj_norms)

        # Multiply the surface area by 2 if symmetric to get consistent area measures
        if self.surface['symmetry']:
            S_ref *= 2

        # Compute the chord for each spanwise portion.
        # This is the distance from the leading to trailing edge.
        chords = np.linalg.norm(mesh[0, :, :] - mesh[-1, :, :], axis=1)

        # Store each array in the unknowns dict
        unknowns['b_pts'] = b_pts
        unknowns['c_pts'] = c_pts
        unknowns['widths'] = widths
        unknowns['cos_sweep'] = cos_sweep
        unknowns['lengths'] = lengths
        unknowns['normals'] = normals
        unknowns['S_ref'] = S_ref
        unknowns['chords'] = chords

    def linearize(self, params, unknowns, resids):
        """ Jacobian for VLM geometry."""

        jac = self.alloc_jacobian()
        name = self.surface['name']
        mesh = params['def_mesh']

        nx = self.surface['num_x']
        ny = self.surface['num_y']

        if fortran_flag:

            normalsb = np.zeros(unknowns['normals'].shape)
            for i in range(nx-1):
                for j in range(ny-1):
                    for ind in range(3):
                        normalsb[:, :, :] = 0.
                        normalsb[i, j, ind] = 1.
                        meshb, _, _ = OAS_API.oas_api.compute_normals_b(mesh, normalsb, 0.)
                        jac['normals', 'def_mesh'][i*(ny-1)*3 + j*3 + ind, :] = meshb.flatten()

            normalsb[:, :, :] = 0.
            if self.surface['S_ref_type'] == 'wetted':
                seed_mesh = mesh
            elif self.surface['S_ref_type'] == 'projected':
                seed_mesh = mesh.copy()
                seed_mesh[:, :, 2] = 0.
            meshb, _, _ = OAS_API.oas_api.compute_normals_b(seed_mesh, normalsb, 1.)

            jac['S_ref', 'def_mesh'] = np.atleast_2d(meshb.flatten())
            if self.surface['symmetry']:
                jac['S_ref', 'def_mesh'] *= 2

        else:
            cs_jac = self.complex_step_jacobian(params, unknowns, resids,
                                            fd_params=['def_mesh'],
                                            fd_unknowns=['normals', 'S_ref'],
                                            fd_states=[])
            jac.update(cs_jac)

        for iz, v in zip((0, ny*3), (.75, .25)):
            np.fill_diagonal(jac['b_pts', 'def_mesh'][:, iz:], v)

        for iz, v in zip((0, 3, ny*3, (ny+1)*3),
                         (.125, .125, .375, .375)):
            for ix in range(nx-1):
                np.fill_diagonal(jac['c_pts', 'def_mesh']
                    [(ix*(ny-1))*3:((ix+1)*(ny-1))*3, iz+ix*ny*3:], v)

        jac['widths', 'def_mesh'] = np.zeros_like(jac['widths', 'def_mesh'])
        jac['cos_sweep', 'def_mesh'] = np.zeros_like(jac['cos_sweep', 'def_mesh'])
        qc = 0.25 * mesh[-1] + 0.75 * mesh[0]
        gap = [0, (nx-1)*ny*3]
        factor = [0.75, 0.25]
        for i in range(ny-1):
            w = unknowns['widths'][i]
            cos_sweep = unknowns['cos_sweep'][i]
            dx = (qc[i+1, 0] - qc[i, 0])
            dy = (qc[i+1, 1] - qc[i, 1])
            dz = (qc[i+1, 2] - qc[i, 2])
            for j in range(2):
                jac['widths', 'def_mesh'][i, i*3+gap[j]] -= dx * factor[j] / w
                jac['widths', 'def_mesh'][i, (i+1)*3+gap[j]] += dx * factor[j] / w
                jac['widths', 'def_mesh'][i, i*3+1+gap[j]] -= dy * factor[j] / w
                jac['widths', 'def_mesh'][i, (i+1)*3+1+gap[j]] += dy * factor[j] / w
                jac['widths', 'def_mesh'][i, i*3+2+gap[j]] -= dz * factor[j] / w
                jac['widths', 'def_mesh'][i, (i+1)*3+2+gap[j]] += dz * factor[j] / w
                jac['cos_sweep', 'def_mesh'][i, i*3+1+gap[j]] -= dy / cos_sweep * factor[j]
                jac['cos_sweep', 'def_mesh'][i, (i+1)*3+1+gap[j]] += dy / cos_sweep * factor[j]
                jac['cos_sweep', 'def_mesh'][i, i*3+2+gap[j]] -= dz / cos_sweep * factor[j]
                jac['cos_sweep', 'def_mesh'][i, (i+1)*3+2+gap[j]] += dz / cos_sweep * factor[j]

        jac['lengths', 'def_mesh'] = np.zeros_like(jac['lengths', 'def_mesh'])
        for i in range(ny):
            dx = mesh[1:, i, 0] - mesh[:-1, i, 0]
            dy = mesh[1:, i, 1] - mesh[:-1, i, 1]
            dz = mesh[1:, i, 2] - mesh[:-1, i, 2]
            for j in range(nx-1):
                l = np.sqrt(dx[j]**2 + dy[j]**2 + dz[j]**2)
                jac['lengths', 'def_mesh'][i, (j*ny+i)*3] -= dx[j] / l
                jac['lengths', 'def_mesh'][i, ((j+1)*ny+i)*3] += dx[j] / l
                jac['lengths', 'def_mesh'][i, (j*ny+i)*3 + 1] -= dy[j] / l
                jac['lengths', 'def_mesh'][i, ((j+1)*ny+i)*3 + 1] += dy[j] / l
                jac['lengths', 'def_mesh'][i, (j*ny+i)*3 + 2] -= dz[j] / l
                jac['lengths', 'def_mesh'][i, ((j+1)*ny+i)*3 + 2] += dz[j] / l

        jac['chords', 'def_mesh'] = np.zeros_like(jac['chords', 'def_mesh'])
        for i in range(ny):
            dx = mesh[0, i, 0] - mesh[-1, i, 0]
            dy = mesh[0, i, 1] - mesh[-1, i, 1]
            dz = mesh[0, i, 2] - mesh[-1, i, 2]

            l = np.sqrt(dx**2 + dy**2 + dz**2)
            jac['chords', 'def_mesh'][i, i*3] += dx / l
            jac['chords', 'def_mesh'][i, (ny+i)*3] -= dx / l
            jac['chords', 'def_mesh'][i, i*3 + 1] += dy / l
            jac['chords', 'def_mesh'][i, (ny+i)*3 + 1] -= dy / l
            jac['chords', 'def_mesh'][i, i*3 + 2] += dz / l
            jac['chords', 'def_mesh'][i, (ny+i)*3 + 2] -= dz / l

        return jac
