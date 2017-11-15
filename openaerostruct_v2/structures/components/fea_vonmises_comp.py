from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_airfoils, tile_sparse_jac

def norm(vec):
    return np.sqrt(np.sum(vec**2))

def unit(vec):
    return vec / norm(vec)


class FEAVonmisesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            for in_name_ in ['radius']:
                in_name = '{}_tube_{}'.format(lifting_surface_name, in_name_)
                self.add_input(in_name, shape=(num_nodes, num_points_z - 1))

            disp_name = '{}_disp'.format(lifting_surface_name)
            self.add_input(disp_name, shape=(num_nodes, num_points_z, 6))

            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            self.add_input(mesh_name, shape=(num_nodes, num_points_z, 3))

            vonmises_name = '{}_vonmises'.format(lifting_surface_name)
            self.add_output(vonmises_name, shape=(num_nodes, num_points_z - 1, 2))

            self.T = np.zeros((3, 3))
            self.x_gl = np.array([1, 0, 0])

            self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        num_nodes = self.metadata['num_nodes']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            E = lifting_surface_data['E']
            G = lifting_surface_data['G']

            radius_name = '{}_tube_{}'.format(lifting_surface_name, 'radius')
            disp_name = '{}_disp'.format(lifting_surface_name)
            mesh_name = '{}_fea_mesh'.format(lifting_surface_name)
            vonmises_name = '{}_vonmises'.format(lifting_surface_name)

            radius = inputs[radius_name]
            disp = inputs[disp_name]
            fea_mesh = inputs[mesh_name]
            vonmises = outputs[vonmises_name]
            T = self.T
            x_gl = self.x_gl

            for node in range(num_nodes):

                for ielem in range(num_points_z-1):

                    P0 = fea_mesh[node, ielem, :]
                    P1 = fea_mesh[node, ielem+1, :]
                    L = norm(P1 - P0)

                    x_loc = unit(P1 - P0)
                    y_loc = unit(np.cross(x_loc, x_gl))
                    z_loc = unit(np.cross(x_loc, y_loc))

                    T[0, :] = x_loc
                    T[1, :] = y_loc
                    T[2, :] = z_loc

                    u0x, u0y, u0z = T.dot(disp[node, ielem, :3])
                    r0x, r0y, r0z = T.dot(disp[node, ielem, 3:])
                    u1x, u1y, u1z = T.dot(disp[node, ielem+1, :3])
                    r1x, r1y, r1z = T.dot(disp[node, ielem+1, 3:])

                    tmp = np.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
                    sxx0 = E * (u1x - u0x) / L + E * radius[node, ielem] / L * tmp
                    sxx1 = E * (u0x - u1x) / L + E * radius[node, ielem] / L * tmp
                    sxt = G * radius[node, ielem] * (r1x - r0x) / L

                    vonmises[node, ielem, 0] = np.sqrt(sxx0**2 + 3*sxt**2)
                    vonmises[node, ielem, 1] = np.sqrt(sxx1**2 + 3*sxt**2)
