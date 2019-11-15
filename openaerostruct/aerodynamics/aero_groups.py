from openmdao.api import Group, LinearRunOnce
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.functionals.total_aero_performance import TotalAeroPerformance


class AeroPoint(Group):
    """
    This group contains all the components needed for a single-point aerodynamic
    analysis. You would have one instance of `AeroPoint` for each flight
    condition you want to study.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('user_specified_Sref', types=bool, default=False)

    def setup(self):
        surfaces = self.options['surfaces']

        # Loop through each surface and connect relevant parameters
        for surface in surfaces:
            name = surface['name']

            self.connect(name + '.normals', 'aero_states.' + name + '_normals')

            # Connect the results from 'aero_states' to the performance groups
            self.connect('aero_states.' + name + '_sec_forces',
                name + '_perf' + '.sec_forces')

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
            self.connect('aero_states.' + name + '_sec_forces',
                'total_perf.' + name + '_sec_forces')

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
            self.add_subsystem(surface['name'] +'_perf',
                VLMFunctionals(surface=surface),
                promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho'])

        # Add the total aero performance group to compute the CL, CD, and CM
        # of the total aircraft. This accounts for all lifting surfaces.
        
        self.add_subsystem('total_perf',
            TotalAeroPerformance(surfaces=surfaces,
            user_specified_Sref=self.options['user_specified_Sref']),
            promotes_inputs=['v', 'rho', 'cg', 'S_ref_total'],
            promotes_outputs=['CM', 'CL', 'CD'])
