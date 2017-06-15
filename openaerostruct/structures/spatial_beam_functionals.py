from openmdao.api import Group
from openaerostruct.structures.energy import Energy
from openaerostruct.structures.weight import Weight
from openaerostruct.structures.vonmises_tube import VonMisesTube
from openaerostruct.structures.non_intersecting_thickness import NonIntersectingThickness
from openaerostruct.structures.spar_within_wing import SparWithinWing
from openaerostruct.structures.failure_exact import FailureExact
from openaerostruct.structures.failure_ks import FailureKS

class SpatialBeamFunctionals(Group):
    """ Group that contains the spatial beam functionals used to evaluate
    performance. """

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        # Commented out energy for now since we haven't ever used its output
        # self.add_subsystem('energy',
        #          Energy(surface=surface),
        #          promotes=['*'])
        self.add_subsystem('thicknessconstraint',
                 NonIntersectingThickness(surface=surface),
                 promotes=['*'])
        self.add_subsystem('structural_weight',
                 Weight(surface=surface),
                 promotes=['*'])
        self.add_subsystem('vonmises',
                 VonMisesTube(surface=surface),
                 promotes=['*'])

        # The following component has not been fully tested so we leave it
        # commented out for now. Use at own risk.
        # self.add_subsystem('sparconstraint',
        #          SparWithinWing(surface=surface),
        #          promotes=['*'])

        if surface['exact_failure_constraint']:
            self.add_subsystem('failure',
                     FailureExact(surface=surface),
                     promotes=['*'])
        else:
            self.add_subsystem('failure',
                    FailureKS(surface=surface),
                    promotes=['*'])
