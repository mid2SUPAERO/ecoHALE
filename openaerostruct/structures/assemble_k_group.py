from openmdao.api import Group
from openaerostruct.structures.transform import Transform
from openaerostruct.structures.length import Length
from openaerostruct.structures.local_stiff import LocalStiff
from openaerostruct.structures.local_stiff_permuted import LocalStiffPermuted
from openaerostruct.structures.local_stiff_transformed import LocalStiffTransformed

class AssembleKGroup(Group):
    """ Assemble that there compact local stiffness matrix. """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        comp = Transform(surface=surface)
        self.add_subsystem('transform', comp, promotes=['*'])

        comp = Length(surface=surface)
        self.add_subsystem('length', comp, promotes=['*'])

        comp = LocalStiff(surface=surface)
#        self.add_subsystem('local_stiff', comp, promotes=['*'])
        self.add_subsystem('local_stiff', comp, promotes_inputs=['A','J','Iy','Iz','element_lengths'], promotes_outputs=['*'])

        comp = LocalStiffPermuted(surface=surface)
        self.add_subsystem('local_stiff_permuted', comp, promotes=['*'])

        comp = LocalStiffTransformed(surface=surface)
        self.add_subsystem('local_stiff_transformed', comp, promotes=['*'])

