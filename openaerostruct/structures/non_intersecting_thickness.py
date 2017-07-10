from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

class NonIntersectingThickness(ExplicitComponent):
    """
    Create a constraint so the thickness of the spar does not intersect
    itself in the center of the spar. Basically, the thickness must be less
    than or equal to the radius.

    inputeters
    ----------
    thickness[ny-1] : numpy array
        Thickness of each element of the FEM spar.
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.

    Returns
    -------
    thickness_intersects[ny-1] : numpy array
        If all the values are negative, each element does not intersect itself.
        If a value is positive, then the thickness within the hollow spar
        intersects itself and presents an impossible design.
        Add a constraint as
        `OASProblem.add_constraint('thickness_intersects', upper=0.)`
    """

    def initialize(self):
        self.metadata.declare('surface', type_=dict)

    def setup(self):
        self.surface = surface = self.metadata['surface']

        self.ny = surface['num_y']

        self.add_input('thickness', val=np.random.rand((self.ny-1)))
        self.add_input('radius', val=np.random.rand((self.ny-1)))
        self.add_output('thickness_intersects', val=np.zeros((self.ny-1)))

        mat = np.eye(self.ny-1)
        self.declare_partials('thickness_intersects', 'thickness', val=mat)
        self.declare_partials('thickness_intersects', 'radius', val=-mat)

    def compute(self, inputs, outputs):
        outputs['thickness_intersects'] = inputs['thickness'] - inputs['radius']
