from __future__ import print_function, division, absolute_import


class LiftingSurface(object):

    def __init__(self, name):
        self.name = name

        self.bsplines = {
            'chord_bspline' : (2, 2),
            'twist_bspline' : (2, 2),
            'sec_x_bspline' : (2, 2),
            'sec_y_bspline' : (2, 2),
            'sec_z_bspline' : (2, 2),
            'thickness_bspline' : (2, 2),
            'radius_bspline' : (2, 2),
            }

    def initialize_mesh(self, num_points_x, num_points_z_half, airfoil_x, airfoil_y, mesh=None):
        """ Set mesh properties OR accept external mesh. """

        if mesh is not None:
            # backcalculate parameters from inputted mesh
            pass

        else:
            self.num_points_x = num_points_x
            self.num_points_z_half = num_points_z_half
            self.airfoil_x = airfoil_x
            self.airfoil_y = airfoil_y

    def set_mesh_parameters(self, distribution, section_origin):
        self.distribution = distribution
        self.section_origin = section_origin

    def set_chord(self, val, n_cp=2, order=2):
        self.chord = val
        self.bsplines['chord_bspline'] = (n_cp, order)

    def set_twist(self, val, n_cp=2, order=2):
        self.twist = val
        self.bsplines['twist_bspline'] = (n_cp, order)

    def set_sweep(self, val, n_cp=2, order=2):
        self.sweep_x = val
        self.bsplines['sec_x_bspline'] = (n_cp, order)

    def set_dihedral(self, val, n_cp=2, order=2):
        self.dihedral_y = val
        self.bsplines['sec_y_bspline'] = (n_cp, order)

    def set_span(self, val, n_cp=2, order=2):
        self.span = val
        self.bsplines['sec_z_bspline'] = (n_cp, order)

    def set_radius(self, val, n_cp=2, order=2):
        self.radius = val
        self.bsplines['radius_bspline'] = (n_cp, order)

    def set_thickness(self, val, n_cp=2, order=2):
        self.thickness = val
        self.bsplines['thickness_bspline'] = (n_cp, order)

    def set_structural_properties(self, E, G, spar_location, sigma_y, rho):
        self.E = E
        self.G = G
        self.spar_location = spar_location
        self.sigma_y = sigma_y
        self.rho = rho

    def set_aero_properties(self, factor2, factor4, cl_factor):
        self.CL0 = 0.
        self.CD0 = 0.

        self.factor2 = factor2
        self.factor4 = factor4
        self.cl_factor = cl_factor
