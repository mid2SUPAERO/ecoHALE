# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 18:51:02 2019

@author: e.duriez
"""

from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


def norm(vec):
    return np.sqrt(np.sum(vec**2))

class PVAreas(ExplicitComponent):
    """
    Computes the areas of the wingbox segments.

    parameters
    ----------
    nodes[ny, 3] : numpy array
        Coordinates of FEM nodes.
    A_int[ny-1] : numpy array
        Internal cross-sectional area of each wingbox segment.

    Returns
    -------
    PV_areas[ny-1] : numpy array
        outer area of each wingbox segment.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.nx = surface['mesh'].shape[0]
        self.ny = surface['mesh'].shape[1]

        self.add_input('mesh', val=np.zeros((self.nx, self.ny, 3)), units='m')
        self.add_output('PV_areas', val=np.zeros((self.ny-1)), units='m**2')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        mesh = inputs['mesh']
        mesh[: , :, 2] = 0. #projected mesh
        proj_normals = np.cross(
            mesh[:-1,  1:, :] - mesh[1:, :-1, :],
            mesh[:-1, :-1, :] - mesh[1:,  1:, :],
            axis=2)

        proj_norms = np.sqrt(np.sum(proj_normals**2, axis=2))
        PV_areas = (proj_norms[0,:]+proj_norms[1,:])/2
        
        outputs['PV_areas'] = PV_areas