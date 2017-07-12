from openaerostruct.geometry.new_geometry_mesh import GeometryMesh
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.geometry.bsplines import Bsplines
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.materials_tube import MaterialsTube
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.functionals.total_performance import TotalPerformance
from openaerostruct.transfer.load_transfer import LoadTransfer
from openaerostruct.aerodynamics.states import VLMStates

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, ExplicitComponent


class Aerostruct(Group):

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        self.add_subsystem('tube',
            MaterialsTube(surface=surface),
            promotes_inputs=['thickness', 'radius'], promotes_outputs=['A', 'Iy', 'Iz', 'J'])

        self.add_subsystem('struct_setup',
            SpatialBeamSetup(surface=surface),
            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J'], promotes_outputs=['nodes', 'K'])

class PreAS(Group):

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        self.add_subsystem('struct_states',
            SpatialBeamStates(surface=surface),
            promotes_inputs=['K', 'forces', 'loads'], promotes_outputs=['disp'])

        self.add_subsystem('def_mesh',
            DisplacementTransfer(surface=surface),
            promotes_inputs=['mesh', 'disp'], promotes_outputs=['def_mesh'])

        self.add_subsystem('aero_geom',
            VLMGeometry(surface=surface),
            promotes_inputs=['def_mesh'], promotes_outputs=['b_pts', 'c_pts', 'widths', 'cos_sweep', 'lengths', 'chords', 'normals', 'S_ref'])

        self.linear_solver = LinearRunOnce()

class CoupledPerformance(Group):

    def initialize(self):
        self.metadata.declare('surface', type_=dict, required=True)

    def setup(self):
        surface = self.metadata['surface']

        self.add_subsystem('aero_funcs',
            VLMFunctionals(surface=surface),
            promotes_inputs=['v', 'alpha', 'M', 're', 'rho', 'widths', 'cos_sweep', 'lengths', 'chords', 'S_ref', 'sec_forces'], promotes_outputs=['CDv', 'Cl', 'L', 'D', 'CL1', 'CDi', 'CD', 'CL'])

        self.add_subsystem('struct_funcs',
            SpatialBeamFunctionals(surface=surface),
            promotes_inputs=['thickness', 'radius', 'A', 'nodes', 'disp'], promotes_outputs=['thickness_intersects', 'structural_weight', 'cg_location', 'vonmises', 'failure'])

class AerostructPoint(Group):

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)

    def setup(self):
        surfaces = self.metadata['surfaces']

        coupled = Group()

        for surface in surfaces:

            name = surface['name']

            # Connect the output of the loads component with the FEM
            # displacement parameter. This links the coupling within the coupled
            # group that necessitates the subgroup solver.
            coupled.connect(name + 'loads.loads', name[:-1] + '.loads')

            # Perform the connections with the modified names within the
            # 'aero_states' group.
            coupled.connect(name[:-1] + '.def_mesh', 'aero_states.' + name + 'def_mesh')
            coupled.connect(name[:-1] + '.b_pts', 'aero_states.' + name + 'b_pts')
            coupled.connect(name[:-1] + '.c_pts', 'aero_states.' + name + 'c_pts')
            coupled.connect(name[:-1] + '.normals', 'aero_states.' + name + 'normals')

            # Connect the results from 'coupled' to the performance groups
            coupled.connect(name[:-1] + '.def_mesh', name + 'loads.def_mesh')
            coupled.connect('aero_states.' + name + 'sec_forces', name + 'loads.sec_forces')

            # Connect the results from 'aero_states' to the performance groups
            self.connect('coupled.aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

            # Connection performance functional variables
            self.connect(name + 'perf.structural_weight', 'total_perf.' + name + 'structural_weight')
            self.connect(name + 'perf.L', 'total_perf.' + name + 'L')
            self.connect(name + 'perf.CL', 'total_perf.' + name + 'CL')
            self.connect(name + 'perf.CD', 'total_perf.' + name + 'CD')
            self.connect('coupled.aero_states.' + name + 'sec_forces', 'total_perf.' + name + 'sec_forces')

            # Connect parameters from the 'coupled' group to the performance
            # groups for the individual surfaces.
            self.connect('coupled.' + name[:-1] + '.disp', name + 'perf.disp')
            self.connect('coupled.' + name[:-1] + '.S_ref', name + 'perf.S_ref')
            self.connect('coupled.' + name[:-1] + '.widths', name + 'perf.widths')
            self.connect('coupled.' + name[:-1] + '.chords', name + 'perf.chords')
            self.connect('coupled.' + name[:-1] + '.lengths', name + 'perf.lengths')
            self.connect('coupled.' + name[:-1] + '.cos_sweep', name + 'perf.cos_sweep')

            # Connect parameters from the 'coupled' group to the total performance group.
            self.connect('coupled.' + name[:-1] + '.S_ref', 'total_perf.' + name + 'S_ref')
            self.connect('coupled.' + name[:-1] + '.widths', 'total_perf.' + name + 'widths')
            self.connect('coupled.' + name[:-1] + '.chords', 'total_perf.' + name + 'chords')
            self.connect('coupled.' + name[:-1] + '.b_pts', 'total_perf.' + name + 'b_pts')
            self.connect(name + 'perf.cg_location', 'total_perf.' + name + 'cg_location')

            # Add components to the 'coupled' group for each surface.
            # The 'coupled' group must contain all components and parameters
            # needed to converge the aerostructural system.
            coupled_AS_group = PreAS(surface=surface)

            coupled.add_subsystem(name[:-1], coupled_AS_group)

            # TODO: add this info to the metadata
            # prob.model.add_metadata(surface['name'] + 'yield_stress', surface['yield'])
            # prob.model.add_metadata(surface['name'] + 'fem_origin', surface['fem_origin'])

        # Add a single 'aero_states' component for the whole system within the
        # coupled group.
        coupled.add_subsystem('aero_states',
            VLMStates(surfaces=surfaces),
            promotes_inputs=['v', 'alpha', 'rho'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        for surface in surfaces:
            name = surface['name']

            # Add a loads component to the coupled group
            coupled.add_subsystem(name + 'loads', LoadTransfer(surface=surface))

        # Set solver properties for the coupled group
        coupled.linear_solver = ScipyIterativeSolver()
        coupled.linear_solver.precon = LinearRunOnce()

        coupled.nonlinear_solver = NonlinearBlockGS()
        coupled.nonlinear_solver.options['maxiter'] = 50

        # coupled.jacobian = DenseJacobian()
        # coupled.linear_solver = DirectSolver()
        # coupled.nonlinear_solver = NewtonSolver(solve_subsystems=True)

        # # 1. GS without aitken:
        # print('1')
        # coupled.ln_solver = ScipyIterativeSolver()
        # coupled.ln_solver.options['maxiter'] = 200
        # coupled.ln_solver.options['atol'] = 1e-16
        # coupled.ln_solver.preconditioner = LinearBlockGS()
        # coupled.ln_solver.preconditioner.options['maxiter'] = 1
        #
        # coupled.nl_solver = NonlinearBlockGS()
        # coupled.nl_solver.options['maxiter'] = 20
        #
        # coupled.nl_solver.options['atol'] = 1e-12
        # coupled.nl_solver.options['rtol'] = 1e-100
        # # coupled.nl_solver.options['utol'] = 1e-100

        # # 2. GS with aitken:
        #
        # print('2')
        # coupled.ln_solver = ScipyIterativeSolver()
        # coupled.ln_solver.options['maxiter'] = 200
        # coupled.ln_solver.options['atol'] =1e-16
        # coupled.ln_solver.preconditioner = LinearBlockGS()
        # coupled.ln_solver.preconditioner.options['maxiter'] = 1
        #
        # coupled.nl_solver = NonlinearBlockGS()
        # coupled.nl_solver.options['maxiter'] = 2000
        # coupled.nl_solver.options['use_aitken'] = True
        # coupled.nl_solver.options['aitken_alpha_min'] = 0.01
        # coupled.nl_solver.options['aitken_alpha_max'] = 1.5
        #
        # coupled.nl_solver.options['atol'] = 1e-12
        # coupled.nl_solver.options['rtol'] = 1e-100
        # # coupled.nl_solver.options['utol'] = 1e-100

        # # 3. Newton with GMRES:
        #
        # print('3')
        # coupled.ln_solver = ScipyIterativeSolver()
        # coupled.ln_solver.options['maxiter'] = 200
        # coupled.ln_solver.preconditioner = LinearBlockGS()
        # coupled.ln_solver.preconditioner.options['maxiter'] = 1
        # coupled.ln_solver.options['atol'] =1e-16
        #
        # coupled.nl_solver = NewtonSolver()
        # coupled.nl_solver.options['maxiter'] = 30
        # coupled.nl_solver.options['solve_subsystems'] = False
        #
        # coupled.nl_solver.options['atol'] = 1e-12
        # coupled.nl_solver.options['rtol'] = 1e-100
        # # coupled.nl_solver.options['utol'] = 1e-100


        # 4. Newton with DirectSolver:

        # print('4')
        # coupled.ln_solver = DirectSolver()
        # coupled.nl_solver = NewtonSolver()
        # coupled.nl_solver.options['maxiter'] = 10
        #
        # coupled.nl_solver.options['atol'] = 1e-12
        # coupled.nl_solver.options['rtol'] = 1e-100
        # # coupled.nl_solver.options['utol'] = 1e-100

        coupled.linear_solver.options['iprint'] = 2
        coupled.nonlinear_solver.options['iprint'] = 2

        # Add the coupled group to the model problem
        self.add_subsystem('coupled', coupled, promotes_inputs=['v', 'alpha', 'rho'])

        for surface in surfaces:
            name = surface['name']

            # Add a performance group which evaluates the data after solving
            # the coupled system
            perf_group = CoupledPerformance(surface=surface)

            self.add_subsystem(name + 'perf', perf_group, promotes_inputs=["rho", "v", "alpha", "re", "M"])

        # Add functionals to evaluate performance of the system.
        # Note that only the interesting results are promoted here; not all
        # of the parameters.
        self.add_subsystem('total_perf',
                 TotalPerformance(surfaces=surfaces),
                 promotes_inputs=['CL', 'CD', 'v', 'rho', 'empty_cg', 'total_weight', 'CT', 'a', 'R', 'M', 'W0', 'load_factor'],
                 promotes_outputs=['L_equals_W', 'fuelburn', 'CM', 'cg'])
