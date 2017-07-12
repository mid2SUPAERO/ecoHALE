from openmdao.api import Group, LinearRunOnce
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.functionals.total_aero_performance import TotalAeroPerformance


class AeroPoint(Group):

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)

    def setup(self):
        surfaces = self.metadata['surfaces']

        for surface in surfaces:
            name = surface['name']

            self.connect(name[:-1] + '.b_pts', 'aero_states.' + name + 'b_pts')
            self.connect(name[:-1] + '.c_pts', 'aero_states.' + name + 'c_pts')
            self.connect(name[:-1] + '.normals', 'aero_states.' + name + 'normals')

            # Connect the results from 'aero_states' to the performance groups
            self.connect('aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

            # Connect S_ref for performance calcs
            self.connect(name[:-1] + '.S_ref', name + 'perf' + '.S_ref')
            self.connect(name[:-1] + '.widths', name + 'perf' + '.widths')
            self.connect(name[:-1] + '.chords', name + 'perf' + '.chords')
            self.connect(name[:-1] + '.lengths', name + 'perf' + '.lengths')
            self.connect(name[:-1] + '.cos_sweep', name + 'perf' + '.cos_sweep')

            # Connect S_ref for performance calcs
            self.connect(name[:-1] + '.S_ref', 'total_perf.' + name + 'S_ref')
            self.connect(name[:-1] + '.widths', 'total_perf.' + name + 'widths')
            self.connect(name[:-1] + '.chords', 'total_perf.' + name + 'chords')
            self.connect(name[:-1] + '.b_pts', 'total_perf.' + name + 'b_pts')
            self.connect(name + 'perf' + '.CL', 'total_perf.' + name + 'CL')
            self.connect(name + 'perf' + '.CD', 'total_perf.' + name + 'CD')
            self.connect('aero_states.' + name + 'sec_forces', 'total_perf.' + name + 'sec_forces')

            self.add_subsystem(name[:-1], VLMGeometry(surface=surface))

        # Add a single 'aero_states' component that solves for the circulations
        # and forces from all the surfaces.
        # While other components only depends on a single surface,
        # this component requires information from all surfaces because
        # each surface interacts with the others.
        aero_states = VLMStates(surfaces=surfaces)
        aero_states.linear_solver = LinearRunOnce()
        self.add_subsystem('aero_states',
                 aero_states,
                 promotes_inputs=['v', 'alpha', 'rho'], promotes_outputs=['circulations'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        # This is necessary because the VLMStates component requires information
        # from each surface, but this information is stored within each
        # surface's group.
        for surface in surfaces:
            self.add_subsystem(surface['name'] +'perf', VLMFunctionals(surface=surface),
                    promotes_inputs=["v", "alpha", "M", "re", "rho"])

        self.add_subsystem('total_perf',
            TotalAeroPerformance(surfaces=surfaces),
            promotes_inputs=['v', 'rho', 'cg', 'S_ref_total'],
            promotes_outputs=['CM', 'CL', 'CD'])
