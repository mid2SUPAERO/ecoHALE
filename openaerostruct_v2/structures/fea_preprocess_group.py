from __future__ import print_function
import numpy as np

from openmdao.api import Group

from openaerostruct_v2.structures.components.tube_properties_comp import TubePropertiesComp
from openaerostruct_v2.structures.components.fea_mesh_comp import FEAMeshComp
from openaerostruct_v2.structures.components.fea_transform_comp import FEATransformComp
from openaerostruct_v2.structures.components.fea_length_comp import FEALengthComp
from openaerostruct_v2.structures.components.fea_local_stiff_comp import FEALocalStiffComp
from openaerostruct_v2.structures.components.fea_local_stiff_permuted_comp import FEALocalStiffPermutedComp
from openaerostruct_v2.structures.components.fea_local_stiff_transformed_comp import FEALocalStiffTransformedComp
from openaerostruct_v2.structures.components.fea_global_stiff_comp import FEAGlobalStiffComp


class FEAPreprocessGroup(Group):

    def initialize(self):
        self.metadata.declare('section_origin', type_=(int, float))
        self.metadata.declare('spar_location', type_=(int, float))
        self.metadata.declare('lifting_surfaces', type_=list)
        self.metadata.declare('E')
        self.metadata.declare('G')

    def setup(self):
        section_origin = self.metadata['section_origin']
        spar_location = self.metadata['spar_location']
        lifting_surfaces = self.metadata['lifting_surfaces']
        E = self.metadata['E']
        G = self.metadata['G']

        comp = TubePropertiesComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('tube_properties_comp', comp, promotes=['*'])

        comp = FEAMeshComp(lifting_surfaces=lifting_surfaces, section_origin=section_origin,
            spar_location=spar_location)
        self.add_subsystem('fea_mesh_comp', comp, promotes=['*'])

        comp = FEATransformComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_transform_comp', comp, promotes=['*'])

        comp = FEALengthComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_length_comp', comp, promotes=['*'])

        comp = FEALocalStiffComp(lifting_surfaces=lifting_surfaces, E=E, G=G)
        self.add_subsystem('fea_local_stiff_comp', comp, promotes=['*'])

        comp = FEALocalStiffPermutedComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_permuted_comp', comp, promotes=['*'])

        comp = FEALocalStiffTransformedComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_local_stiff_transformed_comp', comp, promotes=['*'])

        comp = FEAGlobalStiffComp(lifting_surfaces=lifting_surfaces)
        self.add_subsystem('fea_global_stiff_comp', comp, promotes=['*'])
