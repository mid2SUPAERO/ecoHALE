from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.utils.misc_utils import tile_sparse_jac, get_array_indices

def view_mat(mat):
    """ Helper function used to visually examine matrices. """
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = np.sum(mat, axis=2)
    mat[mat != 0] = 1.
    print(mat.shape)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

class VLMWidthsSweepComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            ref_axis_name = '{}_ref_axis'.format(lifting_surface_name)
            widths_name = '{}_widths'.format(lifting_surface_name)
            cos_sweep_name = '{}_cos_sweep'.format(lifting_surface_name)

            self.add_input(ref_axis_name, shape=(num_nodes, num_points_z, 3),
                val=np.random.rand(num_nodes, num_points_z, 3))
            self.add_output(widths_name, shape=(num_nodes, num_points_z - 1))
            self.add_output(cos_sweep_name, shape=(num_nodes, num_points_z - 1))

            rows = np.tile(np.arange(num_points_z - 1), 2*3)
            cols = np.concatenate([
                np.arange(num_points_z*3)[:-3],
                np.arange(num_points_z*3)[3: ],
            ])

            # _, rows, cols = tile_sparse_jac(1., rows, cols,
            #     num_points_z - 1, num_points_z, num_nodes)

            self.declare_partials(widths_name, ref_axis_name, rows=rows, cols=cols)
            # self.declare_partials(widths_name, ref_axis_name, method='fd')
            self.declare_partials(cos_sweep_name, ref_axis_name)

        self.declare_partials('*', '*', method='cs')

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            section_origin = lifting_surface_data.section_origin

            ref_axis_name = '{}_ref_axis'.format(lifting_surface_name)
            widths_name = '{}_widths'.format(lifting_surface_name)
            cos_sweep_name = '{}_cos_sweep'.format(lifting_surface_name)

            ref_axis = inputs[ref_axis_name]

            left_side = ref_axis[:, :-1, :]
            right_side = ref_axis[:, 1:, :]
            diff = right_side - left_side
            widths = np.sqrt(np.einsum('Bij,Bij->Bi', diff, diff))

            outputs[widths_name] = widths

            # Compute the numerator of the cosine of the sweep angle of each panel
            # (we need this for the viscous drag dependence on sweep, and we only compute
            # the numerator because the denominator of the cosine fraction is the width,
            # which we have already computed. They are combined in the viscous drag
            # calculation.)
            left_side = ref_axis[:, :-1, [1, 2]]
            right_side = ref_axis[:, 1:, [1, 2]]
            diff = right_side - left_side
            cos_sweep = np.sqrt(np.einsum('Bij,Bij->Bi', diff, diff))

            outputs[cos_sweep_name] = cos_sweep

    def compute_partials(self, inputs, partials):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            section_origin = lifting_surface_data.section_origin

            ref_axis_name = '{}_ref_axis'.format(lifting_surface_name)
            widths_name = '{}_widths'.format(lifting_surface_name)
            cos_sweep_name = '{}_cos_sweep'.format(lifting_surface_name)

            ref_axis = inputs[ref_axis_name]

            # derivs = partials[widths_name, ref_axis_name].reshape((num_nodes, 2*num_points_z-2, 3))
            # print(derivs.shape)
            # # view_mat(derivs)
            # derivs[:, :-1, :] = ref_axis[:, :-1, :]
            # derivs[:, 1:, :] = ref_axis[:, 1:, :]


if __name__ == "__main__":
    from openaerostruct.tests.utils import run_test, get_default_lifting_surfaces

    lifting_surfaces = get_default_lifting_surfaces()

    run_test('dummy', VLMWidthsSweepComp(num_nodes=1, lifting_surfaces=lifting_surfaces))
