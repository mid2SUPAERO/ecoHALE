from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class FEAMeshComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('section_origin', type_=(int, float))
        self.metadata.declare('spar_location', type_=(int, float))

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.add_input(in_name, shape=num_points_z)

            self.add_output(mesh_name, shape=(num_points_z, 3))

            vals_dict = {
                'chord': 1.0,
                'twist': 1.0,
                'sec_x': np.outer(np.ones(num_points_z), np.array([1., 0., 0.])).flatten(),
                'sec_y': np.outer(np.ones(num_points_z), np.array([0., 1., 0.])).flatten(),
                'sec_z': np.outer(np.ones(num_points_z), np.array([0., 0., 1.])).flatten(),
            }
            rows = np.arange(num_points_z * 3)
            cols = np.outer(np.arange(num_points_z), np.ones(3, int)).flatten()

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.declare_partials(mesh_name, in_name, val=vals_dict[name], rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)

            chord = inputs['{}_{}'.format(lifting_surface_name, 'chord')]
            twist = inputs['{}_{}'.format(lifting_surface_name, 'twist')]
            sec_x = inputs['{}_{}'.format(lifting_surface_name, 'sec_x')]
            sec_y = inputs['{}_{}'.format(lifting_surface_name, 'sec_y')]
            sec_z = inputs['{}_{}'.format(lifting_surface_name, 'sec_z')]

            outputs[mesh_name] = 0.

            outputs[mesh_name][:, 0] += sec_x
            outputs[mesh_name][:, 1] += sec_y
            outputs[mesh_name][:, 2] += sec_z

            outputs[mesh_name][:, 0] += np.cos(twist) * (spar_location - section_origin) * chord
            outputs[mesh_name][:, 1] -= np.sin(twist) * (spar_location - section_origin) * chord

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_mesh'.format(lifting_surface_name)

            chord_name = '{}_{}'.format(lifting_surface_name, 'chord')
            twist_name = '{}_{}'.format(lifting_surface_name, 'twist')

            chord = inputs[chord_name]
            twist = inputs[twist_name]

            unit_x = np.array([1., 0., 0.])
            unit_y = np.array([0., 1., 0.])

            partials[mesh_name, chord_name] = (
                np.outer( np.cos(twist) * (spar_location - section_origin), unit_x) +
                np.outer(-np.sin(twist) * (spar_location - section_origin), unit_y)
            ).flatten()

            partials[mesh_name, twist_name] = (
                np.outer(-np.sin(twist) * (spar_location - section_origin) * chord, unit_x) +
                np.outer(-np.cos(twist) * (spar_location - section_origin) * chord, unit_y)
            ).flatten()