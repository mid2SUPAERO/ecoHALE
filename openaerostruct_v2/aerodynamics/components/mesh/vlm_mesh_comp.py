from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent


class VLMMeshComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('section_origin', type_=(int, float))
        self.metadata.declare('vortex_mesh', default=False, type_=bool)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        section_origin = self.metadata['section_origin']
        vortex_mesh = self.metadata['vortex_mesh']

        self.airfoils = airfoils = {}

        self.declare_partials('*', '*', dependent=False)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.add_input(in_name, shape=num_points_z)

            if vortex_mesh:
                mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            else:
                mesh_name = '{}_mesh'.format(lifting_surface_name)

            if not vortex_mesh:
                area_name = '{}_area_m2'.format(lifting_surface_name)
                sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

                self.add_output(area_name)
                self.add_output(sec_areas_name, shape=(num_points_z - 1))

            self.add_output(mesh_name, shape=(num_points_x, num_points_z, 3))

            vals_dict = {
                'chord': 1.0,
                'twist': 1.0,
                'sec_x': np.einsum('ij,k->ijk',
                    np.ones((num_points_x, num_points_z)), np.array([1., 0., 0.])).flatten(),
                'sec_y': np.einsum('ij,k->ijk',
                    np.ones((num_points_x, num_points_z)), np.array([0., 1., 0.])).flatten(),
                'sec_z': np.einsum('ij,k->ijk',
                    np.ones((num_points_x, num_points_z)), np.array([0., 0., 1.])).flatten(),
            }
            rows = np.arange(num_points_x * num_points_z * 3)
            cols = np.einsum('j,ik->ijk',
                np.arange(num_points_z), np.ones((num_points_x, 3), int)).flatten()

            for name in ['chord', 'twist', 'sec_x', 'sec_y', 'sec_z']:
                in_name = '{}_{}'.format(lifting_surface_name, name)
                self.declare_partials(mesh_name, in_name, val=vals_dict[name], rows=rows, cols=cols)

            if not vortex_mesh:
                in_name = '{}_{}'.format(lifting_surface_name, 'chord')
                self.declare_partials(area_name, in_name)

                in_name = '{}_{}'.format(lifting_surface_name, 'sec_z')
                self.declare_partials(area_name, in_name)

                in_name = '{}_{}'.format(lifting_surface_name, 'chord')
                self.declare_partials(sec_areas_name, in_name,
                    rows=np.tile(np.arange(num_points_z - 1), 2),
                    cols=np.concatenate([
                        np.arange(num_points_z)[:-1],
                        np.arange(num_points_z)[1: ],
                    ])
                )

                in_name = '{}_{}'.format(lifting_surface_name, 'sec_z')
                self.declare_partials(sec_areas_name, in_name,
                    rows=np.tile(np.arange(num_points_z - 1), 2),
                    cols=np.concatenate([
                        np.arange(num_points_z)[:-1],
                        np.arange(num_points_z)[1: ],
                    ])
                )

            airfoil_x = np.linspace(0., 1., num_points_x) - section_origin
            airfoil_y = np.array(lifting_surface_data['airfoil'])

            if vortex_mesh:
                airfoil_x[:-1] = 0.75 * airfoil_x[:-1] + 0.25 * airfoil_x[1:]
                airfoil_y[:-1] = 0.75 * airfoil_y[:-1] + 0.25 * airfoil_y[1:]

            airfoils[lifting_surface_name] = (airfoil_x, airfoil_y)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            airfoil_x, airfoil_y = self.airfoils[lifting_surface_name]

            if vortex_mesh:
                mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            else:
                mesh_name = '{}_mesh'.format(lifting_surface_name)

            chord = inputs['{}_{}'.format(lifting_surface_name, 'chord')]
            twist = inputs['{}_{}'.format(lifting_surface_name, 'twist')]
            sec_x = inputs['{}_{}'.format(lifting_surface_name, 'sec_x')]
            sec_y = inputs['{}_{}'.format(lifting_surface_name, 'sec_y')]
            sec_z = inputs['{}_{}'.format(lifting_surface_name, 'sec_z')]

            outputs[mesh_name] = 0.

            outputs[mesh_name][:, :, 0] += np.outer(airfoil_x,  np.cos(twist) * chord)
            outputs[mesh_name][:, :, 0] += np.outer(airfoil_y,  np.sin(twist) * chord)
            outputs[mesh_name][:, :, 1] += np.outer(airfoil_x, -np.sin(twist) * chord)
            outputs[mesh_name][:, :, 1] += np.outer(airfoil_y,  np.cos(twist) * chord)

            outputs[mesh_name][:, :, 0] += np.outer(np.ones(num_points_x), sec_x)
            outputs[mesh_name][:, :, 1] += np.outer(np.ones(num_points_x), sec_y)
            outputs[mesh_name][:, :, 2] += np.outer(np.ones(num_points_x), sec_z)

            if not vortex_mesh:
                area_name = '{}_area_m2'.format(lifting_surface_name)
                sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

                outputs[area_name] = np.sum(0.5 * (sec_z[1:] - sec_z[:-1]) * (chord[1:] + chord[:-1]))
                outputs[sec_areas_name] = 0.5 * (sec_z[1:] - sec_z[:-1]) * (chord[1:] + chord[:-1])

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            airfoil_x, airfoil_y = self.airfoils[lifting_surface_name]

            if vortex_mesh:
                mesh_name = '{}_vortex_mesh'.format(lifting_surface_name)
            else:
                mesh_name = '{}_mesh'.format(lifting_surface_name)

            chord_name = '{}_{}'.format(lifting_surface_name, 'chord')
            twist_name = '{}_{}'.format(lifting_surface_name, 'twist')
            sec_z_name = '{}_{}'.format(lifting_surface_name, 'sec_z')

            chord = inputs[chord_name]
            twist = inputs[twist_name]
            sec_z = inputs[sec_z_name]

            unit_x = np.array([1., 0., 0.])
            unit_y = np.array([0., 1., 0.])

            partials[mesh_name, chord_name] = (
                np.einsum('i,j,k->ijk', airfoil_x,  np.cos(twist), unit_x) +
                np.einsum('i,j,k->ijk', airfoil_y,  np.sin(twist), unit_x) +
                np.einsum('i,j,k->ijk', airfoil_x, -np.sin(twist), unit_y) +
                np.einsum('i,j,k->ijk', airfoil_y,  np.cos(twist), unit_y)
            ).flatten()

            partials[mesh_name, twist_name] = (
                np.einsum('i,j,k->ijk', airfoil_x, -np.sin(twist) * chord, unit_x) +
                np.einsum('i,j,k->ijk', airfoil_y,  np.cos(twist) * chord, unit_x) +
                np.einsum('i,j,k->ijk', airfoil_x, -np.cos(twist) * chord, unit_y) +
                np.einsum('i,j,k->ijk', airfoil_y, -np.sin(twist) * chord, unit_y)
            ).flatten()

            if not vortex_mesh:
                area_name = '{}_area_m2'.format(lifting_surface_name)
                sec_areas_name = '{}_sec_areas_m2'.format(lifting_surface_name)

                # outputs[area_name] = 0.5 * (sec_z[1:] - sec_z[:-1]) * (chord[1:] + chord[:-1])
                partials[area_name, chord_name] = 0.
                partials[area_name, chord_name][0, 1: ] += 0.5 * (sec_z[1:] - sec_z[:-1])
                partials[area_name, chord_name][0, :-1] += 0.5 * (sec_z[1:] - sec_z[:-1])

                partials[area_name, sec_z_name] = 0.
                partials[area_name, sec_z_name][0, 1: ] +=  0.5 * (chord[1:] + chord[:-1])
                partials[area_name, sec_z_name][0, :-1] += -0.5 * (chord[1:] + chord[:-1])

                partials[sec_areas_name, chord_name] = np.concatenate([
                    0.5 * (sec_z[1:] - sec_z[:-1]),
                    0.5 * (sec_z[1:] - sec_z[:-1]),
                ])

                partials[sec_areas_name, sec_z_name] = np.concatenate([
                    -0.5 * (chord[1:] + chord[:-1]),
                     0.5 * (chord[1:] + chord[:-1]),
                ])
