from openmdao.api import Group

from openaerostruct.aerodynamics.pg_transform import PGTransform
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.inverse_pg_transform import InversePGTransform


class CompressibleVLMStates(Group):
    """
    Group that contains the states for a compressible aerodynamic analysis.

    This is done in three steps:

        1. Convert all VLM geometries to Prandtl-Glaert domain.

        2. Solve the VLM problem in the Prandtl-Glaert domain using
        VLMStates group.

        3. Convert the resulting forces back to the physical domain through
        the inverse Prandtl-Glauert transform to recover the compressible
        forces.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('rotational', False, types=bool,
                             desc="Set to True to turn on support for computing angular velocities")

    def setup(self):
        surfaces = self.options['surfaces']
        rotational = self.options['rotational']

        # Step 1: Transform geometry to PG domain
        self.add_subsystem('pg_transform', PGTransform(surfaces=surfaces),
            promotes_inputs=['*'])

        #self.connect('pg_transform.alpha_pg', 'pg_states.alpha')
        #self.connect('pg_transform.beta_pg', 'pg_states.beta')
        for surface in surfaces:
            name = surface['name']
            self.connect('pg_transform.' + name + '.def_mesh_pg',
                'pg_states.' + name + '.def_mesh')
            self.connect('pg_transform.' + name + '.b_pts_pg',
                'pg_states.' + name + '.b_pts')
            self.connect('pg_transform.' + name + '.c_pts_pg',
                'pg_states.' + name + '.c_pts')
            self.connect('pg_transform.' + name + '.normals_pg',
                'pg_states.' + name + '.normals')
            self.connect('pg_transform.' + name + '.v_rot_pg',
                'pg_states.' + name + '.v_rot')

        # Step 2: Solve incompressible problem in PG domain
        pg_states = VLMStates(surfaces=surfaces, rotational=rotational)
        self.add_subsystem('pg_states',
                 pg_states, promotes_inputs=['v', 'rho'],
                 promotes_outputs=['circulations'])

        sec_force_prom_list = []
        node_force_prom_list = []
        for surface in surfaces:
            name = surface['name']
            self.connect('pg_states.' + name + '.sec_forces',
                'inverse_pg_transform.' + name + '.sec_forces_pg')
            self.connect('pg_states.' + name + '.node_forces',
                'inverse_pg_transform.' + name + '.node_forces_pg')

        # Step 3: Transform forces from PG domain to physical domain
        self.add_subsystem('inverse_pg_transform', InversePGTransform(surfaces=surfaces),
            promotes_inputs=['alpha', 'beta', 'M'], promotes_outputs=['*forces'])
