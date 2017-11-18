import numpy as np


def get_array_expansion_data(shape, expand_indices):
    alphabet = 'abcdefghij'

    in_string = ''
    out_string = ''
    ones_string = ''
    in_shape = []
    out_shape = []
    ones_shape = []
    for index in range(len(shape)):
        if index not in expand_indices:
            in_string += alphabet[index]
            in_shape.append(shape[index])
        else:
            ones_string += alphabet[index]
            ones_shape.append(shape[index])
        out_string += alphabet[index]
        out_shape.append(shape[index])

    einsum_string = '{},{}->{}'.format(in_string, ones_string, out_string)
    in_shape = tuple(in_shape)
    out_shape = tuple(out_shape)
    ones_shape = tuple(ones_shape)

    return einsum_string, in_shape, out_shape, ones_shape

def expand_array(in_array, shape, expand_indices):
    einsum_string, in_shape, out_shape, ones_shape = get_array_expansion_data(shape, expand_indices)

    out_array = np.einsum(einsum_string, in_array, np.ones(ones_shape))

    return out_array

def tile_sparse_jac(data, rows, cols, nrow, ncol, num_nodes):
    nnz = len(rows)

    if np.isscalar(data):
        data = data * np.ones(nnz)

    if not np.isscalar(nrow):
        nrow = np.prod(nrow)

    if not np.isscalar(ncol):
        ncol = np.prod(ncol)

    data = np.tile(data, num_nodes)
    rows = np.tile(rows, num_nodes) + np.repeat(np.arange(num_nodes), nnz) * nrow
    cols = np.tile(cols, num_nodes) + np.repeat(np.arange(num_nodes), nnz) * ncol

    return data, rows, cols

def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)

def get_airfoils(lifting_surfaces, vortex_mesh):
    airfoils = {}

    for lifting_surface_name, lifting_surface_data in lifting_surfaces:
        num_points_x = lifting_surface_data['num_points_x']
        num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

        section_origin = lifting_surface_data['section_origin']

        airfoil_x = np.linspace(0., 1., num_points_x) - section_origin
        airfoil_y = np.array(lifting_surface_data['airfoil_y'])

        if vortex_mesh:
            airfoil_x[:-1] = 0.75 * airfoil_x[:-1] + 0.25 * airfoil_x[1:]
            airfoil_y[:-1] = 0.75 * airfoil_y[:-1] + 0.25 * airfoil_y[1:]

        airfoils[lifting_surface_name] = (airfoil_x, airfoil_y)

    return airfoils
