""" Group that manipulates geometry mesh based on high-level design parameters. """

from __future__ import division, print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.geometry.geometry_mesh_transformations import \
     Taper, ScaleX, Sweep, ShearX, Stretch, ShearY, Dihedral, \
     ShearZ, Rotate


class GeometryMesh(Group):
    """
    OpenMDAO group that performs mesh manipulation functions. It reads in
    the initial mesh from the surface dictionary and outputs the altered
    mesh based on the geometric design variables.

    Depending on the design variables selected or the supplied geometry information,
    only some of the follow parameters will actually be given to this component.
    If parameters are not active (they do not deform the mesh), then
    they will not be given to this component.

    Parameters
    ----------
    sweep : float
        Shearing sweep angle in degrees.
    dihedral : float
        Dihedral angle in degrees.
    twist[ny] : numpy array
        1-D array of rotation angles for each wing slice in degrees.
    chord_dist[ny] : numpy array
        Chord length for each panel edge.
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Modified mesh based on the initial mesh in the surface dictionary and
        the geometric design variables.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        ny = surface['num_y']
        mesh = surface['mesh']
        mesh_shape = mesh.shape
        symmetry = surface['symmetry']

        # This flag determines whether or not changes in z (dihedral) add an
        # additional rotation matrix to modify the twist direction
        self.rotate_x = True

        # 1. Taper

        if 'taper' in surface:
            val = surface['taper']
            promotes = ['taper']
        else:
            val = 1.
            promotes = []

        self.add_subsystem('taper', Taper(val=val, mesh=mesh, symmetry=symmetry),
                           promotes_inputs=promotes)

        # 2. Scale X

        val = np.ones(ny)
        if 'chord_cp' in surface:
            promotes = ['chord']
        else:
            promotes = []

        self.add_subsystem('scale_x', ScaleX(val=val, mesh_shape=mesh_shape),
                           promotes_inputs=promotes)

        # 3. Sweep

        if 'sweep' in surface:
            val = surface['sweep']
            promotes = ['sweep']
        else:
            val = 0.
            promotes = []

        self.add_subsystem('sweep', Sweep(val=val, mesh_shape=mesh_shape, symmetry=symmetry),
                           promotes_inputs=promotes)

        # 4. Shear X

        val = np.zeros(ny)
        if 'xshear_cp' in surface:
            promotes = ['xshear']
        else:
            promotes = []

        self.add_subsystem('shear_x', ShearX(val=val, mesh_shape=mesh_shape),
                           promotes_inputs=promotes)

        # 5. Stretch

        if 'span' in surface:
            promotes = ['span']
            val = surface['span']
        else:
            # Compute span. We need .real to make span to avoid OpenMDAO warnings.
            quarter_chord = 0.25 * mesh[-1, :, :] + 0.75 * mesh[0, :, :]
            span = max(quarter_chord[:, 1]).real - min(quarter_chord[:, 1]).real
            if symmetry:
                span *= 2.
            val = span
            promotes = []

        self.add_subsystem('stretch', Stretch(val=val, mesh_shape=mesh_shape, symmetry=symmetry),
                           promotes_inputs=promotes)

        # 6. Shear Y

        val = np.zeros(ny)
        if 'yshear_cp' in surface:
            promotes = ['yshear']
        else:
            promotes = []

        self.add_subsystem('shear_y', ShearY(val=val, mesh_shape=mesh_shape),
                           promotes_inputs=promotes)

        # 7. Dihedral

        if 'dihedral' in surface:
            val = surface['dihedral']
            promotes = ['dihedral']
        else:
            val = 0.
            promotes = []

        self.add_subsystem('dihedral', Dihedral(val=val, mesh_shape=mesh_shape, symmetry=symmetry),
                           promotes_inputs=promotes)

        # 8. Shear Z

        val = np.zeros(ny)
        if 'zshear_cp' in surface:
            promotes = ['zshear']
        else:
            promotes = []

        self.add_subsystem('shear_z', ShearZ(val=val, mesh_shape=mesh_shape),
                           promotes_inputs=promotes)

        # 9. Rotate

        val = np.zeros(ny)
        if 'twist_cp' in surface:
            promotes = ['twist']
        else:
            val = 0.
            promotes = []

        self.add_subsystem('rotate', Rotate(val=val, mesh_shape=mesh_shape, symmetry=symmetry),
                           promotes_inputs=promotes, promotes_outputs=['mesh'])


        names = ['taper', 'scale_x', 'sweep', 'shear_x', 'stretch', 'shear_y', 'dihedral',
                 'shear_z', 'rotate']

        for j in np.arange(len(names) - 1):
            self.connect(names[j] + '.mesh', names[j+1] + '.in_mesh')

        #rows = np.arange(nx*ny*3)
        #base = np.tile(np.zeros(3), ny) + np.repeat(np.arange(ny), 3)
        #cols = np.tile(base, nx)
        #self.declare_partials(of='*', wrt='twist', rows=rows, cols=cols)

        #size = nx * ny
        #rows = 3 * np.arange(size)
        #cols = np.tile(np.arange(ny), nx)
        #vals = np.ones(size)
        #self.declare_partials(of='*', wrt='xshear', rows=rows, cols=cols, val=vals)
        #self.declare_partials(of='*', wrt='yshear', rows=rows + 1, cols=cols, val=vals)
        #self.declare_partials(of='*', wrt='zshear', rows=rows + 2, cols=cols, val=vals)

        #if self.rotate_x:
            #rows = np.tile([1, 2], size) + np.repeat(3*np.arange(size), 2)
            #cols = np.zeros(size*2)
            #self.declare_partials(of='*', wrt='dihedral', rows=rows, cols=cols)
        #else:
            #cols = np.zeros(size)
            #self.declare_partials(of='*', wrt='dihedral', rows=rows + 2, cols=cols)

        #self.set_check_partial_options(wrt='*', method='cs')

    #def compute(self, inputs, outputs):
        #mesh = self.mesh.copy()

        #for key in inputs:
            #if inputs[key].shape[0] > 1:
                #self.geo_params[key] = inputs[key]
            #else:
                #self.geo_params[key] = inputs[key][0]

        ## This line used to work in Clippy
        ## self.geo_params.update(inputs)

        #taper(mesh, self.geo_params['taper'], self.symmetry)
        #scale_x(mesh, self.geo_params['chord'])
        #sweep(mesh, self.geo_params['sweep'], self.symmetry)
        #shear_x(mesh, self.geo_params['xshear'])
        #stretch(mesh, self.geo_params['span'], self.symmetry)
        #shear_y(mesh, self.geo_params['yshear'])
        #dihedral(mesh, self.geo_params['dihedral'], self.symmetry)
        #shear_z(mesh, self.geo_params['zshear'])
        #rotate(mesh, self.geo_params['twist'], self.symmetry, self.rotate_x)

        #outputs['mesh'] = mesh

    #def compute_partials(self, inputs, partials):

        ## We actually use the values in self.geo_params to modify the mesh,
        ## but we update self.geo_params using the OpenMDAO params here.
        ## This makes the geometry manipulation process work for any combination
        ## of design variables without having special logic.
        ## self.geo_params.update(inputs)

        #for key in inputs:
            #if inputs[key].shape[0] > 1:
                #self.geo_params[key] = inputs[key]
            #else:
                #self.geo_params[key] = inputs[key][0]

        #mesh = self.mesh.copy()


        ## Sparase Analytic derivatives.
        #nx, ny, _ = self.mesh.shape

        #d_taper = deriv_taper(mesh, self.geo_params['taper'], self.symmetry)
        #scale_x(mesh, self.geo_params['chord'])
        #sweep(mesh, self.geo_params['sweep'], self.symmetry)
        #shear_x(mesh, self.geo_params['xshear'])
        #stretch(mesh, self.geo_params['span'], self.symmetry)
        #shear_y(mesh, self.geo_params['yshear'])

        #d_dihedral, d_mesh = deriv_dihedral(mesh, self.geo_params['dihedral'], self.symmetry)

        #shear_z(mesh, self.geo_params['zshear'])

        #d_twist, d_ymesh, d_zmesh = deriv_rotate(mesh, self.geo_params['twist'], self.symmetry,
                                                 #self.rotate_x)

        #if self.rotate_x:
            #d_dihedral = np.einsum("ijk, j -> ijk", d_zmesh[:, :, 1:], d_dihedral).flatten()
        #else:
            #d_dihedral = np.tile(d_dihedral, nx)

        #partials['mesh', 'dihedral'] = d_dihedral

        #partials['mesh', 'twist'] = d_twist.flatten()