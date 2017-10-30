import numpy as np


def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)

def get_airfoils(lifting_surfaces, section_origin, vortex_mesh):
    airfoils = {}

    for lifting_surface_name, lifting_surface_data in lifting_surfaces:
        num_points_x = lifting_surface_data['num_points_x']
        num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1

        airfoil_x = np.linspace(0., 1., num_points_x) - section_origin
        airfoil_y = np.array(lifting_surface_data['airfoil'])

        if vortex_mesh:
            airfoil_x[:-1] = 0.75 * airfoil_x[:-1] + 0.25 * airfoil_x[1:]
            airfoil_y[:-1] = 0.75 * airfoil_y[:-1] + 0.25 * airfoil_y[1:]

        airfoils[lifting_surface_name] = (airfoil_x, airfoil_y)

    return airfoils
