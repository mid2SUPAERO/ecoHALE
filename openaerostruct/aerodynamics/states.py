from openmdao.api import Group
from openaerostruct.aerodynamics.assemble_aic import AssembleAIC
from openaerostruct.aerodynamics.circulations import Circulations
from openaerostruct.aerodynamics.forces import Forces
from openaerostruct.aerodynamics.assemble_aic_group import AssembleAICGroup


class VLMStates(Group):
    """ Group that contains the aerodynamic states. """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        tot_panels = 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            tot_panels += (nx - 1) * (ny - 1)

        if 0:
            self.add_subsystem('assembly',
                 AssembleAIC(surfaces=surfaces),
                 promotes_inputs=['*'],
                 promotes_outputs=['AIC', 'rhs'])

            self.add_subsystem('circulations',
                Circulations(size=int(tot_panels)),
                promotes_inputs=['AIC', 'rhs'],
                promotes_outputs=['circulations'])

            self.add_subsystem('forces',
                Forces(surfaces=surfaces),
                promotes_inputs=['*'],
                promotes_outputs=['*'])

        else:
            self.add_subsystem('assembly',
                 AssembleAICGroup(surfaces=surfaces),
                 promotes_inputs=['*'],
                 promotes_outputs=['*'])
