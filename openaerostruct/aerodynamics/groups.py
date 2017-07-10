from openmdao.api import Group, LinearRunOnce
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.utils import connect_aero
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.functionals.total_aero_performance import TotalAeroPerformance


class AeroPoint(Group):

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def setup(self):
        surfaces = self.metadata['surfaces']
        prob_dict = self.metadata['prob_dict']

        for surface in surfaces:
            self.add_subsystem(surface['name'][:-1], VLMGeometry(surface=surface), promotes=[])

        # Add a single 'aero_states' component that solves for the circulations
        # and forces from all the surfaces.
        # While other components only depends on a single surface,
        # this component requires information from all surfaces because
        # each surface interacts with the others.
        aero_states = VLMStates(surfaces=surfaces)
        aero_states.linear_solver = LinearRunOnce()
        self.add_subsystem('aero_states',
                 aero_states,
                 promotes=['circulations', 'v', 'alpha', 'rho'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        # This is necessary because the VLMStates component requires information
        # from each surface, but this information is stored within each
        # surface's group.
        for surface in surfaces:
            self.add_subsystem(surface['name'] +'perf', VLMFunctionals(surface=surface, prob_dict=prob_dict),
                    promotes=["v", "alpha", "M", "re", "rho"])

        self.add_subsystem('total_perf',
            TotalAeroPerformance(surfaces=surfaces, prob_dict=prob_dict),
            promotes=['CM', 'CL', 'CD', 'v', 'rho', 'cg'])
