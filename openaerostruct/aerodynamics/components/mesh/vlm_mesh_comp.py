from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import tile_sparse_jac, get_array_indices


class VLMMeshComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('vortex_mesh', default=False, types=bool)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.add_input(in_name, shape=(num_nodes, num_points_z))

            if vortex_mesh:
                mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
                airfoil_x_name = '{}_vortex_airfoil_x'.format(lifting_surface_name)
                airfoil_y_name = '{}_vortex_airfoil_y'.format(lifting_surface_name)
            else:
                mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
                airfoil_x_name = '{}_airfoil_x'.format(lifting_surface_name)
                airfoil_y_name = '{}_airfoil_y'.format(lifting_surface_name)

            if not vortex_mesh:
                area_name = '{}_area_m2'.format(lifting_surface_name)
                sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

                self.add_output(area_name, shape=num_nodes)
                self.add_output(sec_areas_name, shape=(num_nodes, num_points_z - 1))

            self.add_input(airfoil_x_name, shape=(num_nodes, num_points_x, num_points_z))
            self.add_input(airfoil_y_name, shape=(num_nodes, num_points_x, num_points_z))
            self.add_output(mesh_name, shape=(num_nodes, num_points_x, num_points_z, 3))

            vals_dict = {
                'chord': 1.0,
                'twist': 1.0,
                'sec_x': np.einsum('ijk,l->ijkl',
                    np.ones((num_nodes, num_points_x, num_points_z)),
                    np.array([1., 0., 0.])).flatten(),
                'sec_y': np.einsum('ijk,l->ijkl',
                    np.ones((num_nodes, num_points_x, num_points_z)),
                    np.array([0., 1., 0.])).flatten(),
                'sec_z': np.einsum('ijk,l->ijkl',
                    np.ones((num_nodes, num_points_x, num_points_z)),
                    np.array([0., 0., 1.])).flatten(),
            }

            rows = np.arange(num_nodes * num_points_x * num_points_z * 3)
            cols = np.einsum('ik,jl->ijkl',
                get_array_indices(num_nodes, num_points_z),
                np.ones((num_points_x, 3), int),
            ).flatten()
            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.declare_partials(mesh_name, in_name, val=vals_dict[name], rows=rows, cols=cols)

            rows = np.arange(num_nodes * num_points_x * num_points_z * 3)
            cols = np.einsum('ijk,l->ijkl',
                get_array_indices(num_nodes, num_points_x, num_points_z),
                np.ones(3, int),
            ).flatten()
            self.declare_partials(mesh_name, airfoil_x_name, rows=rows, cols=cols)
            self.declare_partials(mesh_name, airfoil_y_name, rows=rows, cols=cols)

            if not vortex_mesh:
                rows = np.zeros(num_points_z, int)
                cols = np.arange(num_points_z)
                _, rows, cols = tile_sparse_jac(1., rows, cols, 1, num_points_z, num_nodes)

                in_name = '{}_{}'.format(lifting_surface_name, 'chord')
                self.declare_partials(area_name, in_name, rows=rows, cols=cols)

                in_name = '{}_{}'.format(lifting_surface_name, 'sec_z')
                self.declare_partials(area_name, in_name, rows=rows, cols=cols)

                rows = np.tile(np.arange(num_points_z - 1), 2)
                cols = np.concatenate([
                    np.arange(num_points_z)[:-1],
                    np.arange(num_points_z)[1: ],
                ])
                _, rows, cols = tile_sparse_jac(1., rows, cols,
                    num_points_z - 1, num_points_z, num_nodes)

                in_name = '{}_{}'.format(lifting_surface_name, 'chord')
                self.declare_partials(sec_areas_name, in_name, rows=rows, cols=cols)

                in_name = '{}_{}'.format(lifting_surface_name, 'sec_z')
                self.declare_partials(sec_areas_name, in_name, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            section_origin = lifting_surface_data.section_origin

            if vortex_mesh:
                mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
                airfoil_x_name = '{}_vortex_airfoil_x'.format(lifting_surface_name)
                airfoil_y_name = '{}_vortex_airfoil_y'.format(lifting_surface_name)
            else:
                mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
                airfoil_x_name = '{}_airfoil_x'.format(lifting_surface_name)
                airfoil_y_name = '{}_airfoil_y'.format(lifting_surface_name)

            chord = np.einsum('ik,j->ijk', inputs['{}_{}'.format(lifting_surface_name, 'chord')], np.ones(num_points_x))
            twist = np.einsum('ik,j->ijk', inputs['{}_{}'.format(lifting_surface_name, 'twist')], np.ones(num_points_x))
            sec_x = np.einsum('ik,j->ijk', inputs['{}_{}'.format(lifting_surface_name, 'sec_x')], np.ones(num_points_x))
            sec_y = np.einsum('ik,j->ijk', inputs['{}_{}'.format(lifting_surface_name, 'sec_y')], np.ones(num_points_x))
            sec_z = np.einsum('ik,j->ijk', inputs['{}_{}'.format(lifting_surface_name, 'sec_z')], np.ones(num_points_x))
            airfoil_x = inputs[airfoil_x_name]
            airfoil_y = inputs[airfoil_y_name]

            outputs[mesh_name] = 0.

            outputs[mesh_name][:, :, :, 0] += (airfoil_x - section_origin) *  np.cos(twist) * chord
            outputs[mesh_name][:, :, :, 0] += (airfoil_y                 ) *  np.sin(twist) * chord
            outputs[mesh_name][:, :, :, 1] += (airfoil_x - section_origin) * -np.sin(twist) * chord
            outputs[mesh_name][:, :, :, 1] += (airfoil_y                 ) *  np.cos(twist) * chord

            outputs[mesh_name][:, :, :, 0] += sec_x
            outputs[mesh_name][:, :, :, 1] += sec_y
            outputs[mesh_name][:, :, :, 2] += sec_z

            if not vortex_mesh:
                area_name = '{}_area_m2'.format(lifting_surface_name)
                sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

                outputs[area_name] = np.sum(
                    0.5 * (sec_z[:, 0, 1:] - sec_z[:, 0, :-1]) * (chord[:, 0, 1:] + chord[:, 0, :-1]),
                    axis=1)
                outputs[sec_areas_name] = 0.5 * (sec_z[:, 0, 1:] - sec_z[:, 0, :-1]) \
                    * (chord[:, 0, 1:] + chord[:, 0, :-1])

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            section_origin = lifting_surface_data.section_origin

            if vortex_mesh:
                mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
                airfoil_x_name = '{}_vortex_airfoil_x'.format(lifting_surface_name)
                airfoil_y_name = '{}_vortex_airfoil_y'.format(lifting_surface_name)
            else:
                mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)
                airfoil_x_name = '{}_airfoil_x'.format(lifting_surface_name)
                airfoil_y_name = '{}_airfoil_y'.format(lifting_surface_name)

            chord_name = '{}_{}'.format(lifting_surface_name, 'chord')
            twist_name = '{}_{}'.format(lifting_surface_name, 'twist')
            sec_z_name = '{}_{}'.format(lifting_surface_name, 'sec_z')

            chord = np.einsum('ik,j->ijk', inputs[chord_name], np.ones(num_points_x))
            twist = np.einsum('ik,j->ijk', inputs[twist_name], np.ones(num_points_x))
            sec_z = np.einsum('ik,j->ijk', inputs[sec_z_name], np.ones(num_points_x))
            airfoil_x = inputs[airfoil_x_name]
            airfoil_y = inputs[airfoil_y_name]

            unit_x = np.array([1., 0., 0.])
            unit_y = np.array([0., 1., 0.])

            derivs = partials[mesh_name, chord_name].reshape((num_nodes, num_points_x, num_points_z, 3))
            derivs[:, :, :, :] = 0.
            derivs[:, :, :, 0] += (airfoil_x - section_origin) *  np.cos(twist)
            derivs[:, :, :, 0] += (airfoil_y                 ) *  np.sin(twist)
            derivs[:, :, :, 1] += (airfoil_x - section_origin) * -np.sin(twist)
            derivs[:, :, :, 1] += (airfoil_y                 ) *  np.cos(twist)

            derivs = partials[mesh_name, twist_name].reshape((num_nodes, num_points_x, num_points_z, 3))
            derivs[:, :, :, :] = 0.
            derivs[:, :, :, 0] += (airfoil_x - section_origin) * -np.sin(twist) * chord
            derivs[:, :, :, 0] += (airfoil_y                 ) *  np.cos(twist) * chord
            derivs[:, :, :, 1] += (airfoil_x - section_origin) * -np.cos(twist) * chord
            derivs[:, :, :, 1] += (airfoil_y                 ) * -np.sin(twist) * chord

            derivs = partials[mesh_name, airfoil_x_name].reshape((num_nodes, num_points_x, num_points_z, 3))
            derivs[:, :, :, :] = 0.
            derivs[:, :, :, 0] +=  np.cos(twist) * chord
            derivs[:, :, :, 1] += -np.sin(twist) * chord

            derivs = partials[mesh_name, airfoil_y_name].reshape((num_nodes, num_points_x, num_points_z, 3))
            derivs[:, :, :, :] = 0.
            derivs[:, :, :, 0] += np.sin(twist) * chord
            derivs[:, :, :, 1] += np.cos(twist) * chord

            if not vortex_mesh:
                area_name = '{}_area_m2'.format(lifting_surface_name)
                sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

                # outputs[area_name] = 0.5 * (sec_z[1:] - sec_z[:-1]) * (chord[1:] + chord[:-1])
                derivs = partials[area_name, chord_name].reshape((num_nodes, num_points_z))
                derivs[:, :] = 0.
                derivs[:, 1: ] += 0.5 * (sec_z[:, 0, 1:] - sec_z[:, 0, :-1])
                derivs[:, :-1] += 0.5 * (sec_z[:, 0, 1:] - sec_z[:, 0, :-1])

                derivs = partials[area_name, sec_z_name].reshape((num_nodes, num_points_z))
                derivs[:, :] = 0.
                derivs[:, 1: ] +=  0.5 * (chord[:, 0, 1:] + chord[:, 0, :-1])
                derivs[:, :-1] += -0.5 * (chord[:, 0, 1:] + chord[:, 0, :-1])

                derivs = partials[sec_areas_name, chord_name].reshape((num_nodes, 2, num_points_z - 1))
                derivs[:, 0, :] = 0.5 * (sec_z[:, 0, 1:] - sec_z[:, 0, :-1])
                derivs[:, 1, :] = 0.5 * (sec_z[:, 0, 1:] - sec_z[:, 0, :-1])

                derivs = partials[sec_areas_name, sec_z_name].reshape((num_nodes, 2, num_points_z - 1))
                derivs[:, 0, :] = -0.5 * (chord[:, 0, 1:] + chord[:, 0, :-1])
                derivs[:, 1, :] =  0.5 * (chord[:, 0, 1:] + chord[:, 0, :-1])
