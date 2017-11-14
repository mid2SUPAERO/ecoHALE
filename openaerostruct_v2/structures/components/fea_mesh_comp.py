from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import tile_sparse_jac


class FEAMeshComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.add_input(in_name, shape=(num_nodes, num_points_z))

            self.add_output(mesh_name, shape=(num_nodes, num_points_z, 3))

            vals_dict = {
                'chord': 1.0,
                'twist': 1.0,
                'sec_x': np.einsum('ij,k->ijk',
                    np.ones((num_nodes, num_points_z)),
                    np.array([1., 0., 0.])).flatten(),
                'sec_y': np.einsum('ij,k->ijk',
                    np.ones((num_nodes, num_points_z)),
                    np.array([0., 1., 0.])).flatten(),
                'sec_z': np.einsum('ij,k->ijk',
                    np.ones((num_nodes, num_points_z)),
                    np.array([0., 0., 1.])).flatten(),
            }
            rows = np.arange(num_points_z * 3)
            cols = np.outer(np.arange(num_points_z), np.ones(3, int)).flatten()
            _, rows, cols = tile_sparse_jac(1., rows, cols,
                num_points_z * 3, num_points_z, num_nodes)

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.declare_partials(mesh_name, in_name, val=vals_dict[name], rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            section_origin = lifting_surface_data['section_origin']
            spar_location = lifting_surface_data['spar_location']

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)

            chord = inputs['{}_{}'.format(lifting_surface_name, 'chord')]
            twist = inputs['{}_{}'.format(lifting_surface_name, 'twist')]
            sec_x = inputs['{}_{}'.format(lifting_surface_name, 'sec_x')]
            sec_y = inputs['{}_{}'.format(lifting_surface_name, 'sec_y')]
            sec_z = inputs['{}_{}'.format(lifting_surface_name, 'sec_z')]

            outputs[mesh_name] = 0.

            outputs[mesh_name][:, :, 0] += sec_x
            outputs[mesh_name][:, :, 1] += sec_y
            outputs[mesh_name][:, :, 2] += sec_z

            outputs[mesh_name][:, :, 0] += np.cos(twist) * (spar_location - section_origin) * chord
            outputs[mesh_name][:, :, 1] -= np.sin(twist) * (spar_location - section_origin) * chord

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            section_origin = lifting_surface_data['section_origin']
            spar_location = lifting_surface_data['spar_location']

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)

            chord_name = '{}_{}'.format(lifting_surface_name, 'chord')
            twist_name = '{}_{}'.format(lifting_surface_name, 'twist')

            chord = inputs[chord_name]
            twist = inputs[twist_name]

            unit_x = np.array([1., 0., 0.])
            unit_y = np.array([0., 1., 0.])

            partials[mesh_name, chord_name] = (
                np.einsum('ij,k->ijk',  np.cos(twist) * (spar_location - section_origin), unit_x) +
                np.einsum('ij,k->ijk', -np.sin(twist) * (spar_location - section_origin), unit_y)
            ).flatten()

            partials[mesh_name, twist_name] = (
                np.einsum('ij,k->ijk', -np.sin(twist) * (spar_location - section_origin) * chord, unit_x) +
                np.einsum('ij,k->ijk', -np.cos(twist) * (spar_location - section_origin) * chord, unit_y)
            ).flatten()
