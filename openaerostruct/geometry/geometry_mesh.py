""" Manipulate geometry mesh based on high-level design parameters. """

from __future__ import division, print_function
import numpy as np
from numpy import cos, sin, tan

from openaerostruct.geometry.utils import \
    rotate, scale_x, shear_x, shear_z, \
    sweep, dihedral, stretch, taper

from openmdao.api import ExplicitComponent
from openaerostruct.structures.utils import radii

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class GeometryMesh(ExplicitComponent):
    """
    OpenMDAO component that performs mesh manipulation functions. It reads in
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
        self.metadata.declare('surface', type_=dict, required=True)
        self.metadata.declare('desvars', default={}, type_=dict)

    def initialize_variables(self):
        surface = self.metadata['surface']
        name = surface['name']

        self.desvar_names = desvar_names = []
        for desvar in self.metadata['desvars']:
            # Check to make sure that the surface's name is in the design
            # variable and only add the desvar to the list if it corresponds
            # to this surface.
            if name[:-1] in desvar:
                desvar_names.append(''.join(desvar.split('.')[1:]))

        ny = surface['num_y']
        self.mesh = surface['mesh']

        # Variables that should be initialized to one
        ones_list = ['taper', 'chord_cp']

        # Variables that should be initialized to zero
        zeros_list = ['sweep', 'dihedral', 'twist_cp', 'xshear_cp', 'zshear_cp']

        # Variables that should be initialized to given value
        set_list = ['span']

        # Make a list of all geometry variables by adding all individual lists
        all_geo_vars = ones_list + zeros_list + set_list
        self.geo_params = geo_params = {}
        for var in all_geo_vars:
            if len(var.split('_')) > 1:
                param = var.split('_')[0]
                if var in ones_list:
                    val = np.ones(ny)
                elif var in zeros_list:
                    val = np.zeros(ny)
                else:
                    val = surface[var]
            else:
                param = var
                if var in ones_list:
                    val = 1.0
                elif var in zeros_list:
                    val = 0.0
                else:
                    val = surface[var]
            geo_params[param] = val

            # If the user supplied a variable or it's a desvar, we add it as a
            # parameter.
            if var in desvar_names or var in surface['initial_geo']:
                self.add_input(param, val=val)

        self.add_output('mesh', val=self.mesh)

        # If the user doesn't provide the radius or it's not a desver, then we must
        # compute it here
        if 'radius_cp' not in desvar_names and 'radius_cp' not in surface['initial_geo']:
            self.compute_radius = True
            self.add_output('radius', val=np.zeros((ny - 1)))
        else:
            self.compute_radius = False

        self.symmetry = surface['symmetry']

        # This flag determines whether or not changes in z (dihedral) add an
        # additional rotation matrix to modify the twist direction
        self.rotate_x = True

    def initialize_partials(self):
        if not fortran_flag:
            self.approx_partials('*', '*')

    def compute(self, inputs, outputs):
        mesh = self.mesh.copy()

        # Dirty hack for now; TODO: fix this
        for key in self.geo_params:
            try:
                if inputs[key].shape[0] > 1:
                    self.geo_params[key] = inputs[key]
                else:
                    self.geo_params[key] = inputs[key][0]
            except:
                pass

        # This line used to work in Clippy
        # self.geo_params.update(inputs)

        if fortran_flag:
            mesh = OAS_API.oas_api.manipulate_mesh(mesh, self.geo_params['taper'],
                self.geo_params['chord'], self.geo_params['sweep'], self.geo_params['xshear'],
                self.geo_params['dihedral'], self.geo_params['zshear'],
                self.geo_params['twist'], self.geo_params['span'], self.symmetry, self.rotate_x)

        else:
            taper(mesh, self.geo_params['taper'], self.symmetry)
            scale_x(mesh, self.geo_params['chord'])
            stretch(mesh, self.geo_params['span'], self.symmetry)
            sweep(mesh, self.geo_params['sweep'], self.symmetry)
            shear_x(mesh, self.geo_params['xshear'])
            dihedral(mesh, self.geo_params['dihedral'], self.symmetry)
            shear_z(mesh, self.geo_params['zshear'])
            rotate(mesh, self.geo_params['twist'], self.symmetry, self.rotate_x)

        # Only compute the radius on the first iteration
        if self.compute_radius and 'radius_cp' not in self.desvar_names:
            # Get spar radii and interpolate to radius control points.
            # Need to refactor this at some point since the derivatives are wrong.
            outputs['radius'] = radii(mesh, self.metadata['surface']['t_over_c'])
            self.compute_radius = False

        outputs['mesh'] = mesh

    if 0:
        def compute_jacvec_product(
                self, inputs, outputs, d_inputs, d_outputs, mode):

            mesh = self.mesh.copy()

            # We actually use the values in self.geo_params to modify the mesh,
            # but we update self.geo_params using the OpenMDAO params here.
            # This makes the geometry manipulation process work for any combination
            # of design variables without having special logic.
            # self.geo_params.update(inputs)

            # Dirty hack for now; TODO: fix this
            for key in self.geo_params:
                try:
                    if inputs[key].shape[0] > 1:
                        self.geo_params[key] = inputs[key]
                    else:
                        self.geo_params[key] = inputs[key][0]
                except:
                    pass

            if mode == 'fwd':

                # We don't know which parameters will be used for a given case
                # so we must check
                if 'sweep' in d_inputs:
                    sweepd = d_inputs['sweep']
                else:
                    sweepd = 0.
                if 'twist' in d_inputs:
                    twistd = d_inputs['twist']
                else:
                    twistd = np.zeros(self.geo_params['twist'].shape)
                if 'chord' in d_inputs:
                    chordd = d_inputs['chord']
                else:
                    chordd = np.zeros(self.geo_params['chord'].shape)
                if 'dihedral' in d_inputs:
                    dihedrald = d_inputs['dihedral']
                else:
                    dihedrald = 0.
                if 'taper' in d_inputs:
                    taperd = d_inputs['taper']
                else:
                    taperd = 0.
                if 'xshear' in d_inputs:
                    xsheard = d_inputs['xshear']
                else:
                    xsheard = np.zeros(self.geo_params['xshear'].shape)
                if 'zshear' in d_inputs:
                    zsheard = d_inputs['zshear']
                else:
                    zsheard = np.zeros(self.geo_params['zshear'].shape)
                if 'span' in d_inputs:
                    spand = d_inputs['span']
                else:
                    spand = 0.

                mesh, d_outputs['mesh'] = OAS_API.oas_api.manipulate_mesh_d(mesh,
                self.geo_params['taper'], taperd, self.geo_params['chord'], chordd,
                self.geo_params['sweep'], sweepd, self.geo_params['xshear'], xsheard,
                self.geo_params['dihedral'], dihedrald, self.geo_params['zshear'],
                zsheard, self.geo_params['twist'], twistd, self.geo_params['span'],
                spand, self.symmetry, self.rotate_x)

            if mode == 'rev':
                taperb, chordb, sweepb, xshearb, dihedralb, zshearb, twistb, spanb, mesh = \
                OAS_API.oas_api.manipulate_mesh_b(mesh, self.geo_params['taper'],
                self.geo_params['chord'], self.geo_params['sweep'],
                self.geo_params['xshear'], self.geo_params['dihedral'],
                self.geo_params['zshear'], self.geo_params['twist'],
                self.geo_params['span'], self.symmetry, self.rotate_x, d_outputs['mesh'])

                if 'sweep' in d_inputs:
                    d_inputs['sweep'] += sweepb
                if 'twist' in d_inputs:
                    d_inputs['twist'] += twistb
                if 'chord' in d_inputs:
                    d_inputs['chord'] += chordb
                if 'dihedral' in d_inputs:
                    d_inputs['dihedral'] += dihedralb
                if 'taper' in d_inputs:
                    d_inputs['taper'] += taperb
                if 'xshear' in d_inputs:
                    d_inputs['xshear'] += xshearb
                if 'zshear' in d_inputs:
                    d_inputs['zshear'] += zshearb
                if 'span' in d_inputs:
                    d_inputs['span'] += spanb

    else:
        def compute_partial_derivs(self, inputs, outputs, partials):

            # We actually use the values in self.geo_params to modify the mesh,
            # but we update self.geo_params using the OpenMDAO params here.
            # This makes the geometry manipulation process work for any combination
            # of design variables without having special logic.
            # self.geo_params.update(inputs)

            # Dirty hack for now; TODO: fix this
            for key in self.geo_params:
                try:
                    if inputs[key].shape[0] > 1:
                        self.geo_params[key] = inputs[key]
                    else:
                        self.geo_params[key] = inputs[key][0]
                except:
                    pass

            for param in inputs:

                d_inputs = {}
                d_inputs[param] = self.geo_params[param].copy()

                if isinstance(d_inputs[param], np.ndarray):
                    for j, val in enumerate(d_inputs[param].flatten()):
                        d_inputs[param][:] = 0.
                        d_inputs[param][j] = 1.

                        # We don't know which parameters will be used for a given case
                        # so we must check
                        if 'sweep' in d_inputs:
                            sweepd = d_inputs['sweep']
                        else:
                            sweepd = 0.
                        if 'twist' in d_inputs:
                            twistd = d_inputs['twist']
                        else:
                            twistd = np.zeros(self.geo_params['twist'].shape)
                        if 'chord' in d_inputs:
                            chordd = d_inputs['chord']
                        else:
                            chordd = np.zeros(self.geo_params['chord'].shape)
                        if 'dihedral' in d_inputs:
                            dihedrald = d_inputs['dihedral']
                        else:
                            dihedrald = 0.
                        if 'taper' in d_inputs:
                            taperd = d_inputs['taper']
                        else:
                            taperd = 0.
                        if 'xshear' in d_inputs:
                            xsheard = d_inputs['xshear']
                        else:
                            xsheard = np.zeros(self.geo_params['xshear'].shape)
                        if 'zshear' in d_inputs:
                            zsheard = d_inputs['zshear']
                        else:
                            zsheard = np.zeros(self.geo_params['zshear'].shape)
                        if 'span' in d_inputs:
                            spand = d_inputs['span']
                        else:
                            spand = 0.

                        _, mesh_d = OAS_API.oas_api.manipulate_mesh_d(self.mesh.copy(),
                        self.geo_params['taper'], taperd, self.geo_params['chord'], chordd,
                        self.geo_params['sweep'], sweepd, self.geo_params['xshear'], xsheard,
                        self.geo_params['dihedral'], dihedrald, self.geo_params['zshear'],
                        zsheard, self.geo_params['twist'], twistd, self.geo_params['span'],
                        spand, self.symmetry, self.rotate_x)

                        partials['mesh', param][:, j] = mesh_d.flatten()

                else:

                    d_inputs[param] = 1.

                    # We don't know which parameters will be used for a given case
                    # so we must check
                    if 'sweep' in d_inputs:
                        sweepd = d_inputs['sweep']
                    else:
                        sweepd = 0.
                    if 'twist' in d_inputs:
                        twistd = d_inputs['twist']
                    else:
                        twistd = np.zeros(self.geo_params['twist'].shape)
                    if 'chord' in d_inputs:
                        chordd = d_inputs['chord']
                    else:
                        chordd = np.zeros(self.geo_params['chord'].shape)
                    if 'dihedral' in d_inputs:
                        dihedrald = d_inputs['dihedral']
                    else:
                        dihedrald = 0.
                    if 'taper' in d_inputs:
                        taperd = d_inputs['taper']
                    else:
                        taperd = 0.
                    if 'xshear' in d_inputs:
                        xsheard = d_inputs['xshear']
                    else:
                        xsheard = np.zeros(self.geo_params['xshear'].shape)
                    if 'zshear' in d_inputs:
                        zsheard = d_inputs['zshear']
                    else:
                        zsheard = np.zeros(self.geo_params['zshear'].shape)
                    if 'span' in d_inputs:
                        spand = d_inputs['span']
                    else:
                        spand = 0.

                    _, mesh_d = OAS_API.oas_api.manipulate_mesh_d(self.mesh.copy(),
                    self.geo_params['taper'], taperd, self.geo_params['chord'], chordd,
                    self.geo_params['sweep'], sweepd, self.geo_params['xshear'], xsheard,
                    self.geo_params['dihedral'], dihedrald, self.geo_params['zshear'],
                    zsheard, self.geo_params['twist'], twistd, self.geo_params['span'],
                    spand, self.symmetry, self.rotate_x)

                    partials['mesh', param] = mesh_d.flatten()
