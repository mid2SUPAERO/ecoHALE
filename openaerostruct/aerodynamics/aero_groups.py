from openmdao.api import Group, LinearRunOnce
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.functionals.total_aero_performance import TotalAeroPerformance


class AeroPoint(Group):

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        for surface in surfaces:
            name = surface['name']

            self.connect(name + '.normals', 'aero_states.' + name + '_normals')
            # self.connect(name + '.b_pts', 'aero_states.' + name + '_b_pts')
            # self.connect(name + '.c_pts', 'aero_states.' + name + '_c_pts')
            # self.connect(name + '.cos_sweep', 'aero_states.' + name + '_cos_sweep')
            # self.connect(name + '.widths', 'aero_states.' + name + '_widths')

            # Connect the results from 'aero_states' to the performance groups
            self.connect('aero_states.' + name + '_sec_forces', name + '_perf' + '.sec_forces')

            # Connect S_ref for performance calcs
            self.connect(name + '.S_ref', name + '_perf.S_ref')
            self.connect(name + '.widths', name + '_perf.widths')
            self.connect(name + '.chords', name + '_perf.chords')
            self.connect(name + '.lengths', name + '_perf.lengths')
            self.connect(name + '.cos_sweep', name + '_perf.cos_sweep')

            # Connect S_ref for performance calcs
            self.connect(name + '.S_ref', 'total_perf.' + name + '_S_ref')
            self.connect(name + '.widths', 'total_perf.' + name + '_widths')
            self.connect(name + '.chords', 'total_perf.' + name + '_chords')
            self.connect(name + '.b_pts', 'total_perf.' + name + '_b_pts')
            self.connect(name + '_perf' + '.CL', 'total_perf.' + name + '_CL')
            self.connect(name + '_perf' + '.CD', 'total_perf.' + name + '_CD')
            self.connect('aero_states.' + name + '_sec_forces', 'total_perf.' + name + '_sec_forces')

            self.add_subsystem(name, VLMGeometry(surface=surface))

        # Add a single 'aero_states' component that solves for the circulations
        # and forces from all the surfaces.
        # While other components only depends on a single surface,
        # this component requires information from all surfaces because
        # each surface interacts with the others.
        aero_states = VLMStates(surfaces=surfaces)
        aero_states.linear_solver = LinearRunOnce()

        self.add_subsystem('aero_states',
                 aero_states,
                 promotes_inputs=['v', 'alpha', 'rho'],
                 promotes_outputs=['circulations'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        # This is necessary because the VLMStates component requires information
        # from each surface, but this information is stored within each
        # surface's group.
        for surface in surfaces:
            self.add_subsystem(surface['name'] +'_perf', VLMFunctionals(surface=surface),
                    promotes_inputs=["v", "alpha", "M", "re", "rho"])

        self.add_subsystem('total_perf',
            TotalAeroPerformance(surfaces=surfaces),
            promotes_inputs=['v', 'rho', 'cg', 'S_ref_total'],
            promotes_outputs=['CM', 'CL', 'CD'])
