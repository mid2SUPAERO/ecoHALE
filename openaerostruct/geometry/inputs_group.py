from __future__ import print_function
import numpy as np

from openmdao.api import Group, IndepVarComp

from openaerostruct.common.bspline_comp import BsplineComp
from openaerostruct.common.array_expansion_comp import ArrayExpansionComp

from openaerostruct.utils.misc_utils import expand_array


class InputsGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        default_bspline = (2, 2)

        initial_vals = {}
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            chord = lifting_surface_data.chord
            twist = lifting_surface_data.twist
            sweep_x = lifting_surface_data.sweep_x
            dihedral_y = lifting_surface_data.dihedral_y
            span = lifting_surface_data.span
            airfoil_x = lifting_surface_data.airfoil_x
            airfoil_y = lifting_surface_data.airfoil_y

            chord_ncp, chord_order = lifting_surface_data.bsplines['chord_bspline']
            twist_ncp, twist_order = lifting_surface_data.bsplines['twist_bspline']
            sec_x_ncp, sec_x_order = lifting_surface_data.bsplines['sec_x_bspline']
            sec_y_ncp, sec_y_order = lifting_surface_data.bsplines['sec_y_bspline']
            sec_z_ncp, sec_z_order = lifting_surface_data.bsplines['sec_z_bspline']

            if chord is not None:
                initial_vals[lifting_surface_name, 'chord'] = chord * np.ones(2 * chord_ncp - 1)
            else:
                initial_vals[lifting_surface_name, 'chord'] = None

            if twist is not None:
                initial_vals[lifting_surface_name, 'twist'] = twist * np.ones(2 * twist_ncp - 1)
            else:
                initial_vals[lifting_surface_name, 'twist'] = None

            if sweep_x is not None:
                sec_x = np.zeros(2 * sec_x_ncp - 1)
                sec_x[sec_x_ncp - 1:] = np.linspace(0., sweep_x, sec_x_ncp)
                sec_x[:sec_x_ncp][::-1] = np.linspace(0., sweep_x, sec_x_ncp)
                initial_vals[lifting_surface_name, 'sec_x'] = sec_x
            else:
                initial_vals[lifting_surface_name, 'sec_x'] = None

            if dihedral_y is not None:
                sec_y = np.zeros(2 * sec_y_ncp - 1)
                sec_y[sec_y_ncp - 1:] = np.linspace(0., dihedral_y, sec_y_ncp)
                sec_y[:sec_y_ncp][::-1] = np.linspace(0., dihedral_y, sec_y_ncp)
                initial_vals[lifting_surface_name, 'sec_y'] = sec_y
            else:
                initial_vals[lifting_surface_name, 'sec_y'] = None

            if span is not None:
                sec_z = np.zeros(2 * sec_z_ncp - 1)
                sec_z = span * np.linspace(-1., 1., 2 * sec_z_ncp - 1)
                initial_vals[lifting_surface_name, 'sec_z'] = sec_z
            else:
                initial_vals[lifting_surface_name, 'sec_z'] = None

            if airfoil_x is not None:
                airfoil_x = expand_array(airfoil_x, (num_nodes, num_points_x, num_points_z), [0, 2])
                initial_vals[lifting_surface_name, 'airfoil_x'] = airfoil_x
            else:
                initial_vals[lifting_surface_name, 'airfoil_x'] = None

            if airfoil_y is not None:
                airfoil_y = expand_array(airfoil_y, (num_nodes, num_points_x, num_points_z), [0, 2])
                initial_vals[lifting_surface_name, 'airfoil_y'] = airfoil_y
            else:
                initial_vals[lifting_surface_name, 'airfoil_y'] = None

        comp = IndepVarComp()
        comp.add_output('dummy_var')
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                val = initial_vals[lifting_surface_name, name]
                if val is not None:
                    name = '{}_{}_dv'.format(lifting_surface_name, name)
                    comp.add_output(name, val=val)

            for name in ['airfoil_x', 'airfoil_y']:
                val = initial_vals[lifting_surface_name, name]
                if val is not None:
                    name = '{}_{}'.format(lifting_surface_name, name)
                    comp.add_output(name, val=val)

        self.add_subsystem('indep_var_comp', comp, promotes=['*'])

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                ncp, bspline_order = lifting_surface_data.bsplines.get(
                    '{}_bspline'.format(name), default_bspline)
                num_control_points = 2 * ncp - 1

                in_name = '{}_{}_dv'.format(lifting_surface_name, name)
                out_name = '{}_{}_cp'.format(lifting_surface_name, name)
                comp = ArrayExpansionComp(
                    shape=(num_nodes, num_control_points),
                    expand_indices=[0],
                    in_name=in_name,
                    out_name=out_name,
                )

                val = initial_vals[lifting_surface_name, name]
                if val is not None:
                    self.add_subsystem('{}_{}_expand_cp_comp'.format(lifting_surface_name, name), comp,
                        promotes=['*'])

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            distribution = lifting_surface_data.distribution

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                ncp, bspline_order = lifting_surface_data.bsplines.get(
                    '{}_bspline'.format(name), default_bspline)
                num_control_points = 2 * ncp - 1

                in_name = '{}_{}_cp'.format(lifting_surface_name, name)
                out_name = '{}_{}'.format(lifting_surface_name, name)
                comp = BsplineComp(
                    num_nodes=num_nodes,
                    num_control_points=num_control_points,
                    num_points=num_points_z,
                    bspline_order=bspline_order,
                    in_name=in_name,
                    out_name=out_name,
                    distribution=distribution,
                )
                self.add_subsystem('{}_{}_bspline_comp'.format(lifting_surface_name, name), comp,
                    promotes=['*'])
