import numpy as np

from openaerostruct.geometry.ffd_component import GeometryMesh
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer

from openmdao.api import IndepVarComp, Group

class Geometry(Group):
    """ Group that contains everything needed for a structural-only problem. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        # Get the surface name and create a group to contain components
        # only for this surface
        nx, ny = surface['mesh'].shape[:2]

        # Add independent variables that do not belong to a specific component
        indep_var_comp = IndepVarComp()

        # Add structural components to the surface-specific group
        self.add_subsystem('indep_vars',
                 indep_var_comp,
                 promotes=['*'])

        mx, my = 3, 3

        indep_var_comp.add_output('twist', val=0.)
        indep_var_comp.add_output('shape', val=np.zeros((mx*my)))

        bsp_inputs = []

        mesh_promotes = ['mesh']

        self.add_subsystem('mesh',
            GeometryMesh(surface=surface, mx=mx, my=my),
            promotes_inputs=['twist', 'shape'],
            promotes_outputs=['mesh'])

        if surface['type'] == 'aero':
            indep_var_comp.add_output('disp', val=np.zeros((ny, 6)))
            self.add_subsystem('def_mesh',
                DisplacementTransfer(surface=surface),
                promotes_inputs=['disp', 'mesh'],
                promotes_outputs=['def_mesh'])
