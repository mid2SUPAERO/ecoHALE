from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, view_model, ExecComp, SqliteRecorder

from openaerostruct_v2.geometry.inputs_group import InputsGroup
from openaerostruct_v2.structures.fea_bspline_group import FEABsplineGroup

from openaerostruct_v2.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct_v2.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct_v2.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct_v2.structures.fea_postprocess_group import FEAPostprocessGroup

from openaerostruct_v2.aerostruct.aerostruct_group import AerostructGroup


num_nodes = 1

num_points_x = 2
num_points_z_half = 15
num_points_z = 2 * num_points_z_half - 1
lifting_surfaces = [
    ('wing', {
        'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
        'airfoil': np.zeros(num_points_x),
        'chord': 1., 'twist': 0. * np.pi / 180., 'sweep_x': 0., 'dihedral_y': 0., 'span': 5,
        'twist_bspline': (6, 2),
        'sec_z_bspline': (num_points_z_half, 2),
        'chord_bspline': (2, 2),
        'thickness_bspline': (6, 3),
        'thickness' : 0.05,
        'radius' : 0.1,
        'distribution': 'sine',
        'section_origin': 0.25,
        'spar_location': 0.35,
        'E': 70.e9,
        'G': 29.e9,
        'sigma_y': 200e6, #200e6,
        'rho': 2700,
    })
]

vlm_scaler = 1e2
fea_scaler = 1e6

prob = Problem()
prob.model = Group()

indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

group = FEABsplineGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
prob.model.add_subsystem('tube_bspline_group', group, promotes=['*'])

prob.model.add_subsystem('vlm_preprocess_group',
    VLMPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('fea_preprocess_group',
    FEAPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
    promotes=['*'],
)

prob.model.add_subsystem('aerostruct_group',
    AerostructGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, vlm_scaler=vlm_scaler, fea_scaler=fea_scaler),
    promotes=['*'],
)

prob.model.add_subsystem('vlm_postprocess_group',
    VLMPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('fea_postprocess_group',
    FEAPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)

prob.model.add_subsystem('objective',
    ExecComp('obj=10000 * sum(C_D) + structural_weight', C_D=np.zeros(num_nodes)),
    promotes=['*'],
)

prob.model.add_design_var('alpha_rad', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_design_var('wing_twist_dv', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_design_var('wing_tube_thickness_dv', lower=0.001, upper=0.19, scaler=1e3)
prob.model.add_objective('obj')
prob.model.add_constraint('wing_ks', upper=0.)
prob.model.add_constraint('C_L', equals=np.linspace(0.8, 0.8, num_nodes))

prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.opt_settings['Major optimality tolerance'] = 3e-7
prob.driver.opt_settings['Major feasibility tolerance'] = 3e-7

prob.driver.add_recorder(SqliteRecorder('aerostruct.hst'))
prob.driver.recording_options['includes'] = ['*']

prob.setup()

prob['wing_chord_dv'] = [0.5, 1.0, 0.5]

if 0:
    prob.setup(force_alloc_complex=True)
    prob['wing_chord_dv'] = [0.5, 1.0, 0.5]
    prob.run_model()
    prob.check_partials(compact_print=True)
    exit()

prob.run_driver()

print(prob['wing_twist_dv'])
print(prob['wing_tube_thickness_dv'])

# print(prob['circulations'])
# print(prob['wing_disp'])

# view_model(prob)
