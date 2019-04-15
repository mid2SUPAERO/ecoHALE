from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent
from openaerostruct.structures.utils import norm
from openaerostruct.utils.constants import grav_constant


class AddPointMasses(ExplicitComponent):
    """
    Add point mass to the structure and compute the loads due to the point masses.
    The current method adds loads and moments to all of the structural nodes, but
    the nodes closest to the point masses receive proportionally larger loads.

    Parameters
    ----------
    point_mass_locations[n_point_masses, 3] : numpy array
        XYZ location for each point mass, relative to the spar and symmetry plane.
    point_masses[n_point_masses] : numpy array
        Actual magnitude of each point mass, in same order as point_mass_locations.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    load_factor : float
        Load factor for the flight point. Multiplier on the effects of gravity.

    Returns
    -------
    distances[n_point_masses, ny] : numpy array
        Actual Euclidean distance from each point mass to each node.
    loading_weights[n_point_masses, ny] : numpy array
        The normalized weight that each point mass has on each node. The closest
        nodes have the greatest weighting, while farther away nodes have less.
    loads_from_point_masses[ny, 6] : numpy array
        The actual loads array that will be added to the total loads array.
        This is cumulative and includes the forces and moments from all point masses.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']
        self.ny = surface['mesh'].shape[1]
        self.n_point_masses = surface['n_point_masses']

        self.add_input('point_mass_locations', val=np.zeros((self.n_point_masses, 3)), units='m')
        self.add_input('point_masses', val=np.zeros((self.n_point_masses)), units='kg')
        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('load_factor', val=1.0)

        self.add_output('loading_weights', val=np.zeros((self.n_point_masses, self.ny)))
        self.add_output('loads_from_point_masses', val=np.zeros((self.ny, 6)), units='N') ## WARNING!!! UNITS ARE A MIXTURE OF N & N*m

        self.set_check_partial_options(wrt='*', method='fd')
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nodes = inputs['nodes']
        loading_weights = outputs['loading_weights']
        loads_from_point_masses = outputs['loads_from_point_masses']
        loads_from_point_masses[:] = 0.

        for idx, point_mass_location in enumerate(inputs['point_mass_locations']):
            xyz_dist = point_mass_location - nodes
            span_dist = xyz_dist[:, 1]

            # vec_to_nodes = nodes[1:, :] - nodes[:-1, :]

            torsional_moment_arms = xyz_dist.copy()
            torsional_moment_arms[:] = 0.
            torsional_moment_arms[:, 0] = xyz_dist[:, 0]

            dist10 = span_dist**10
            inv_dist10 = 1 / dist10# + 1e-10)
            loading_weights[idx, :] = inv_dist10 / np.sum(inv_dist10)

            directional_weights = np.outer(loading_weights[idx, :], np.array([[0., 0., -1.]]))
            weight_forces = directional_weights * grav_constant * inputs['load_factor'] * inputs['point_masses'][idx]

            loads_from_point_masses[:, :3] += weight_forces

            moments = np.cross(torsional_moment_arms, weight_forces)
            loads_from_point_masses[:, 3:] += moments
