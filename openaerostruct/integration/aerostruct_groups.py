from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.transfer.displacement_transfer_group import DisplacementTransferGroup
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.functionals.total_performance import TotalPerformance
from openaerostruct.transfer.load_transfer import LoadTransfer
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.structures.tube_group import TubeGroup
from openaerostruct.structures.wingbox_group import WingboxGroup

from openmdao.api import Group, NonlinearBlockGS, DirectSolver, LinearBlockGS, LinearRunOnce


class AerostructGeometry(Group):

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('DVGeo', default=None)

    def setup(self):
        surface = self.options['surface']
        DVGeo = self.options['DVGeo']
		

        geom_promotes = []

        if 'twist_cp' in surface.keys():
            geom_promotes.append('twist_cp')
        if 't_over_c_cp' in surface.keys():
            geom_promotes.append('t_over_c')
        if 'sweep' in surface.keys():
            geom_promotes.append('sweep')
        if 'taper' in surface.keys():
            geom_promotes.append('taper')
        if 'mx' in surface.keys():
            geom_promotes.append('shape')
        if 'span' in surface.keys():
            geom_promotes.append('span')
        if 'chord_cp' in surface.keys():
            geom_promotes.append('chord_cp')

        self.add_subsystem('geometry',
            Geometry(surface=surface, DVGeo=DVGeo),
            promotes_inputs=[],
            promotes_outputs=['mesh'] + geom_promotes)

        if surface['fem_model_type'] == 'tube':
            tube_promotes = []
            tube_inputs = []
            if 'thickness_cp' in surface.keys():
                tube_promotes.append('thickness_cp')
            if 'radius_cp' not in surface.keys():
                tube_inputs = ['mesh', 't_over_c']
            self.add_subsystem('tube_group',
                TubeGroup(surface=surface),
                promotes_inputs=tube_inputs,
                promotes_outputs=['A', 'Iy', 'Iz', 'J', 'radius', 'thickness'] + tube_promotes)
        elif surface['fem_model_type'] == 'wingbox':
            wingbox_promotes = []
            if 'skin_thickness_cp' in surface.keys() and 'spar_thickness_cp' in surface.keys():
                wingbox_promotes.append('skin_thickness_cp')
                wingbox_promotes.append('spar_thickness_cp')
                wingbox_promotes.append('skin_thickness')
                wingbox_promotes.append('spar_thickness')
            elif 'skin_thickness_cp' in surface.keys() or 'spar_thickness_cp' in surface.keys():
                raise NameError('Please have both skin and spar thickness as design variables, not one or the other.')

            self.add_subsystem('wingbox_group',
                WingboxGroup(surface=surface),
                promotes_inputs=['mesh', 't_over_c'],
                promotes_outputs=['A', 'Iy', 'Iz', 'J', 'Qz', 'A_enc', 'A_int', 'htop', 'hbottom', 'hfront', 'hrear'] + wingbox_promotes)
        else:
            raise NameError('Please select a valid `fem_model_type` from either `tube` or `wingbox`.')

        # if surface['fem_model_type'] == 'wingbox':
            # promotes = ['A_int']
        # else:
            # promotes = []
			
        self.add_subsystem('struct_setup',
            SpatialBeamSetup(surface=surface),
#            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J','mrho'] + promotes,   #ED
#            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J'] + promotes,   #ED
            promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J'],   #ED
            promotes_outputs=['nodes', 'local_stiff_transformed', 'structural_mass', 'cg_location', 'element_mass'])


class CoupledAS(Group):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        promotes = []
        if surface['struct_weight_relief']:
            promotes = promotes + list(set(['nodes', 'element_mass', 'load_factor']))
#        if surface['distributed_fuel_weight']:
#            promotes = promotes + list(set(['nodes', 'load_factor']))
        promotes = promotes + list(set(['nodes', 'load_factor'])) #ED
        self.add_subsystem('struct_states',
            SpatialBeamStates(surface=surface),
            promotes_inputs=['local_stiff_transformed', 'forces', 'loads'] + promotes, promotes_outputs=['disp'])

        self.add_subsystem('def_mesh',
            DisplacementTransferGroup(surface=surface),
            promotes_inputs=['nodes', 'mesh', 'disp'], promotes_outputs=['def_mesh'])

        self.add_subsystem('aero_geom',
            VLMGeometry(surface=surface),
            promotes_inputs=['def_mesh'], promotes_outputs=['b_pts', 'widths', 'cos_sweep', 'lengths', 'chords', 'normals', 'S_ref'])

        self.linear_solver = LinearRunOnce()


class CoupledPerformance(Group):

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        surface = self.options['surface']

        self.add_subsystem('aero_funcs',
            VLMFunctionals(surface=surface),
            promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'widths', 'cos_sweep', 'lengths', 'S_ref', 'sec_forces', 't_over_c'], promotes_outputs=['CDv', 'L', 'D', 'CL1', 'CDi', 'CD', 'CL'])

        if surface['fem_model_type'] == 'tube':
            self.add_subsystem('struct_funcs',
                SpatialBeamFunctionals(surface=surface),
                promotes_inputs=['thickness', 'radius', 'nodes', 'disp'], promotes_outputs=['thickness_intersects', 'vonmises', 'failure'])

        elif surface['fem_model_type'] == 'wingbox':
            self.add_subsystem('struct_funcs',
                SpatialBeamFunctionals(surface=surface),
                promotes_inputs=['Qz', 'J', 'A_enc', 'spar_thickness','skin_thickness', 'htop', 'hbottom', 'hfront', 'hrear', 'nodes', 'disp','mrho'], promotes_outputs=['vonmises', 'failure','buckling'])
        else:
            raise NameError('Please select a valid `fem_model_type` from either `tube` or `wingbox`.')


class AerostructPoint(Group):

    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('user_specified_Sref', types=bool, default=False)
#        self.options.declare('internally_connect_fuelburn', types=bool, default=True)

    def setup(self):
        surfaces = self.options['surfaces']

        coupled = Group()

        for surface in surfaces:

            name = surface['name']

            # Connect the output of the loads component with the FEM
            # displacement parameter. This links the coupling within the coupled
            # group that necessitates the subgroup solver.
            coupled.connect(name + '_loads.loads', name + '.loads')

            # Perform the connections with the modified names within the
            # 'aero_states' group.
            coupled.connect(name + '.normals', 'aero_states.' + name + '_normals')
            coupled.connect(name + '.def_mesh', 'aero_states.' + name + '_def_mesh')

            # Connect the results from 'coupled' to the performance groups
            coupled.connect(name + '.def_mesh', name + '_loads.def_mesh')
            coupled.connect('aero_states.' + name + '_sec_forces', name + '_loads.sec_forces')

            # Connect the results from 'aero_states' to the performance groups
            self.connect('coupled.aero_states.' + name + '_sec_forces', name + '_perf' + '.sec_forces')

            # Connection performance functional variables
            self.connect(name + '_perf.CL', 'total_perf.' + name + '_CL')
            self.connect(name + '_perf.CD', 'total_perf.' + name + '_CD')
            self.connect('coupled.aero_states.' + name + '_sec_forces', 'total_perf.' + name + '_sec_forces')
            self.connect('coupled.' + name + '.chords', name + '_perf.aero_funcs.chords')

            # Connect parameters from the 'coupled' group to the performance
            # groups for the individual surfaces.
            self.connect('coupled.' + name + '.disp', name + '_perf.disp')
            self.connect('coupled.' + name + '.S_ref', name + '_perf.S_ref')
            self.connect('coupled.' + name + '.widths', name + '_perf.widths')
            # self.connect('coupled.' + name + '.chords', name + '_perf.chords')
            self.connect('coupled.' + name + '.lengths', name + '_perf.lengths')
            self.connect('coupled.' + name + '.cos_sweep', name + '_perf.cos_sweep')

            # Connect parameters from the 'coupled' group to the total performance group.
            self.connect('coupled.' + name + '.S_ref', 'total_perf.' + name + '_S_ref')
            self.connect('coupled.' + name + '.widths', 'total_perf.' + name + '_widths')
            self.connect('coupled.' + name + '.chords', 'total_perf.' + name + '_chords')
            self.connect('coupled.' + name + '.b_pts', 'total_perf.' + name + '_b_pts')

            # Add components to the 'coupled' group for each surface.
            # The 'coupled' group must contain all components and parameters
            # needed to converge the aerostructural system.
            coupled_AS_group = CoupledAS(surface=surface)
            
            promotes = []
            # if surface['distributed_fuel_weight']:
                # promotes = ['load_factor']
            # else:
                # promotes = []
            promotes = ['load_factor'] #ED
            coupled.add_subsystem(name, coupled_AS_group, promotes_inputs=promotes)

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
            coupled.add_subsystem(name + '_loads', LoadTransfer(surface=surface))

        """
        ### Change the solver settings here ###
        """

        # Set solver properties for the coupled group
        # coupled.linear_solver = ScipyIterativeSolver()
        # coupled.linear_solver.precon = LinearRunOnce()

        coupled.nonlinear_solver = NonlinearBlockGS(use_aitken=True)
        coupled.nonlinear_solver.options['maxiter'] = 100
        coupled.nonlinear_solver.options['atol'] = 1e-7
        coupled.nonlinear_solver.options['rtol'] = 1e-30

        # coupled.linear_solver = DirectSolver()

        coupled.linear_solver = DirectSolver(assemble_jac=True)
        coupled.options['assembled_jac_type'] = 'csc'

        # coupled.nonlinear_solver = NewtonSolver(solve_subsystems=True)
        # coupled.nonlinear_solver.options['maxiter'] = 50
        coupled.nonlinear_solver.options['iprint'] = 2

        """
        ### End change of solver settings ###
        """

        # Add the coupled group to the model problem
        self.add_subsystem('coupled', coupled, promotes_inputs=['v', 'alpha', 'rho'])

        for surface in surfaces:
            name = surface['name']

            # Add a performance group which evaluates the data after solving
            # the coupled system
            perf_group = CoupledPerformance(surface=surface)

            self.add_subsystem(name + '_perf', perf_group, promotes_inputs=['rho', 'v', 'alpha', 're', 'Mach_number','mrho'])

        # Add functionals to evaluate performance of the system.
        # Note that only the interesting results are promoted here; not all
        # of the parameters.
        # self.add_subsystem('total_perf',
                 # TotalPerformance(surfaces=surfaces,
                 # user_specified_Sref=self.options['user_specified_Sref'],
                 # internally_connect_fuelburn=self.options['internally_connect_fuelburn']),
                 # promotes_inputs=['v', 'rho', 'empty_cg', 'total_weight', 'CT', 'speed_of_sound', 'R', 'Mach_number', 'W0', 'load_factor', 'S_ref_total'],
                 # promotes_outputs=['L_equals_W', 'fuelburn', 'CL', 'CD', 'CM', 'cg'])
        self.add_subsystem('total_perf',
                 TotalPerformance(surfaces=surfaces,
                 user_specified_Sref=self.options['user_specified_Sref']),
                 promotes_inputs=['v', 'rho', 'empty_cg', 'total_weight', 'speed_of_sound', 'Mach_number', 'W0', 'load_factor', 'S_ref_total'],
                 promotes_outputs=['L_equals_W', 'CL', 'CD', 'CM', 'cg','enough_power'])