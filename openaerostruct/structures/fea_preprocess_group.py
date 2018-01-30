from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct.structures.components.tube_properties_comp import TubePropertiesComp
from openaerostruct.structures.components.fea_mesh_comp import FEAMeshComp
from openaerostruct.structures.components.fea_transform_comp import FEATransformComp
from openaerostruct.structures.components.fea_length_comp import FEALengthComp
from openaerostruct.structures.components.fea_local_stiff_comp import FEALocalStiffComp
from openaerostruct.structures.components.fea_local_stiff_permuted_comp import FEALocalStiffPermutedComp
from openaerostruct.structures.components.fea_local_stiff_transformed_comp import FEALocalStiffTransformedComp
from openaerostruct.structures.components.fea_global_stiff_comp import FEAGlobalStiffComp


class FEAPreprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)
        self.metadata.declare('fea_scaler', types=float, default=1e6)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']
        fea_scaler = self.metadata['fea_scaler']

        comp = TubePropertiesComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('tube_properties_comp', comp, promotes=['*'])

        comp = FEAMeshComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_mesh_comp', comp, promotes=['*'])

        comp = FEATransformComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_transform_comp', comp, promotes=['*'])

        comp = FEALengthComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_length_comp', comp, promotes=['*'])

        comp = FEALocalStiffComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_comp', comp, promotes=['*'])

        comp = FEALocalStiffPermutedComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_permuted_comp', comp, promotes=['*'])

        comp = FEALocalStiffTransformedComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_transformed_comp', comp, promotes=['*'])

        comp = FEAGlobalStiffComp(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler)
        self.add_subsystem('fea_global_stiff_comp', comp, promotes=['*'])
