from openmdao.api import Group, ExplicitComponent, BsplinesComp
from openaerostruct.common.reynolds_comp import ReynoldsComp
from openaerostruct.common.atmos_comp import AtmosComp


class AtmosGroup(Group):

    def setup(self):
        self.add_subsystem('atmos',
            AtmosComp(),
            promotes_inputs=['altitude', 'Mach_number'],
            promotes_outputs=['T', 'P', 'rho', 'speed_of_sound', 'mu', 'v'])

        self.add_subsystem('reynolds',
            ReynoldsComp(),
            promotes_inputs=['rho', 'mu', 'v'],
            promotes_outputs=['re'])
