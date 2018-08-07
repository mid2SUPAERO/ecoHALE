from openmdao.api import Group
from openaerostruct.aerodynamics.assemble_aic import AssembleAIC
from openaerostruct.aerodynamics.get_vectors import GetVectors
from openaerostruct.aerodynamics.collocation_points import CollocationPoints
from openaerostruct.aerodynamics.eval_mtx import EvalVelMtx
from openaerostruct.aerodynamics.convert_velocity import ConvertVelocity
from openaerostruct.aerodynamics.mtx_rhs import VLMMtxRHSComp


class AssembleAICGroup(Group):
    """ Group that contains the aerodynamic states. """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        num_collocation_points = 0
        for surface in surfaces:
            ny = surface['num_y']
            nx = surface['num_x']
            num_collocation_points += (ny - 1) * (nx - 1)

        # Get collocation points
        self.add_subsystem('collocation_points',
             CollocationPoints(surfaces=surfaces),
             promotes_inputs=['*'],
             promotes_outputs=['coll_pts', 'force_pts', 'bound_vecs'])

        # Get vectors from mesh points to collocation points
        self.add_subsystem('get_vectors',
             GetVectors(surfaces=surfaces, num_eval_points=num_collocation_points,
                eval_name='coll_pts'),
             promotes_inputs=['*'],
             promotes_outputs=['*'])

        # Construct matrix based on rings, not horseshoes
        self.add_subsystem('mtx_assy',
             EvalVelMtx(surfaces=surfaces, num_eval_points=num_collocation_points,
                eval_name='coll_pts'),
             promotes_inputs=['*'],
             promotes_outputs=['*'])

        # Convert freestream velocity to array of velocities
        self.add_subsystem('convert_velocity',
             ConvertVelocity(surfaces=surfaces),
             promotes_inputs=['*'],
             promotes_outputs=['*'])


        # Construct RHS of matrix system
        self.add_subsystem('mtx_rhs',
             VLMMtxRHSComp(surfaces=surfaces),
             promotes_inputs=['*'],
             promotes_outputs=['*'])

        # Solve Mtx RHS to get ring circs

        # Convert ring circs to horseshoe circs

        # Set up force mtx

        # Multiple by horseshoe circs to get forces
