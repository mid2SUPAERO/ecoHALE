from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct_v2.utils.misc_utils import get_array_indices, get_airfoils
from openaerostruct_v2.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class ASDispTransferComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('vortex_mesh', default=False, type_=bool)

    def setup(self):
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            disp_name = '{}_disp'.format(lifting_surface_name)
            axis_name = '{}_ref_axis'.format(lifting_surface_name)
            transform_name = '{}_transform_mtx'.format(lifting_surface_name)
            if vortex_mesh:
                mesh_disp_name = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
                mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            else:
                mesh_disp_name = '{}_mesh_displacement'.format(lifting_surface_name)
                mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)

            self.add_input(disp_name, shape=(num_points_z, 6))
            self.add_input(axis_name, shape=(num_points_z, 3))
            self.add_input(mesh_name, shape=(num_points_x, num_points_z, 3))
            self.add_input(transform_name, shape=(num_points_z, 3, 3))
            self.add_output(mesh_disp_name, shape=(num_points_x, num_points_z, 3))

            disp_indices = get_array_indices(num_points_z, 6)
            axis_indices = get_array_indices(num_points_z, 3)
            mesh_indices = get_array_indices(num_points_x, num_points_z, 3)
            transform_indices = get_array_indices(num_points_z, 3, 3)
            mesh_disp_indices = get_array_indices(num_points_x, num_points_z, 3)

            rows = mesh_disp_indices.flatten()
            cols = np.einsum('i,jk->ijk', np.ones(num_points_x), disp_indices[:, :3]).flatten()
            self.declare_partials(mesh_disp_name, disp_name, val=1., rows=rows, cols=cols)

            rows = np.einsum('ijk,l->ijkl', mesh_disp_indices, np.ones(3, int)).flatten()
            cols = np.einsum('ik,jl->ijkl', np.ones((num_points_x, 3), int), axis_indices).flatten()
            self.declare_partials(mesh_disp_name, axis_name, rows=rows, cols=cols)

            rows = np.einsum('ijk,l->ijkl', mesh_disp_indices, np.ones(3, int)).flatten()
            cols = np.einsum('ijl,k->ijkl', mesh_indices, np.ones(3, int)).flatten()
            self.declare_partials(mesh_disp_name, mesh_name, rows=rows, cols=cols)

            rows = np.einsum('ijl,k->ijkl', mesh_disp_indices, np.ones(3, int)).flatten()
            cols = np.einsum('jkl,i->ijkl', transform_indices, np.ones(num_points_x, int)).flatten()
            self.declare_partials(mesh_disp_name, transform_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            disp_name = '{}_disp'.format(lifting_surface_name)
            axis_name = '{}_ref_axis'.format(lifting_surface_name)
            transform_name = '{}_transform_mtx'.format(lifting_surface_name)
            if vortex_mesh:
                mesh_disp_name = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
                mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            else:
                mesh_disp_name = '{}_mesh_displacement'.format(lifting_surface_name)
                mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)

            outputs[mesh_disp_name] = np.einsum('i,jk->ijk',
                np.ones(num_points_x), inputs[disp_name][:, :3])
            outputs[mesh_disp_name] += np.einsum('ijk,jkl->ijl',
                inputs[mesh_name] - np.einsum('i,jk->ijk', np.ones(num_points_x), inputs[axis_name]),
                inputs[transform_name])

    def compute_partials(self, inputs, partials):
        lifting_surfaces = self.metadata['lifting_surfaces']
        vortex_mesh = self.metadata['vortex_mesh']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

            disp_name = '{}_disp'.format(lifting_surface_name)
            axis_name = '{}_ref_axis'.format(lifting_surface_name)
            transform_name = '{}_transform_mtx'.format(lifting_surface_name)
            if vortex_mesh:
                mesh_disp_name = '{}_vortex_mesh_displacement'.format(lifting_surface_name)
                mesh_name = '{}_undeformed_vortex_mesh'.format(lifting_surface_name)
            else:
                mesh_disp_name = '{}_mesh_displacement'.format(lifting_surface_name)
                mesh_name = '{}_undeformed_mesh'.format(lifting_surface_name)

            partials[mesh_disp_name, axis_name] = -np.einsum('i,jlk->ijkl',
                np.ones(num_points_x), inputs[transform_name]).flatten()

            partials[mesh_disp_name, mesh_name] = np.einsum('i,jlk->ijkl',
                np.ones(num_points_x), inputs[transform_name]).flatten()

            partials[mesh_disp_name, transform_name] = np.einsum('ijk,l->ijkl',
                inputs[mesh_name] - np.einsum('i,jk->ijk', np.ones(num_points_x), inputs[axis_name]),
                np.ones(3)).flatten()
