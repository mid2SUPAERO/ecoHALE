from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model, Group

from openaerostruct_v2.geometry.inputs_group import InputsGroup
from openaerostruct_v2.aerodynamics.vlm_preprocess_group import VLMPreprocessGroup
from openaerostruct_v2.aerodynamics.vlm_states_group import VLMStatesGroup
from openaerostruct_v2.aerodynamics.vlm_postprocess_group import VLMPostprocessGroup

from openaerostruct_v2.utils.plot_utils import plot_mesh_2d, scatter_2d, arrow_2d


num_points_x = 2
num_points_z_half = 31

num_points_z = 2 * num_points_z_half - 1

airfoil = np.zeros(num_points_x)
# airfoil[1:-1] = 0.2

section_origin = 0.25
lifting_surfaces = [
    ('wing', {
        'num_points_x': num_points_x, 'num_points_z_half': num_points_z_half,
        'airfoil': airfoil,
        'chord': 1., 'twist': 0. * np.pi / 180., 'sweep_x': 0., 'dihedral_y': 0., 'span': 5,
        'twist_bspline': (11, 3),
        'sec_z_bspline': (num_points_z_half, 2),
        'chord_bspline': (2, 2),
    })
]

prob = Problem()
prob.model = Group()

indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v_m_s', 200.)
indep_var_comp.add_output('alpha_rad', 3. * np.pi / 180.)
indep_var_comp.add_output('rho_kg_m3', 1.225)
prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

inputs_group = InputsGroup(lifting_surfaces=lifting_surfaces)
prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

prob.model.add_subsystem('vlm_preprocess_group',
    VLMPreprocessGroup(section_origin=section_origin, lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('vlm_states_group',
    VLMStatesGroup(lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)
prob.model.add_subsystem('vlm_postprocess_group',
    VLMPostprocessGroup(lifting_surfaces=lifting_surfaces),
    promotes=['*'],
)

prob.model.add_design_var('wing_twist_cp', lower=-3.*np.pi/180., upper=8.*np.pi/180.)
prob.model.add_objective('C_D')
prob.model.add_constraint('C_L', equals=0.5)

prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.opt_settings['Major optimality tolerance'] = 2e-7
prob.driver.opt_settings['Major feasibility tolerance'] = 2e-7
# prob.driver.opt_settings['Verify level'] = 3

prob.setup()

prob['wing_chord_cp'] = [0.5, 1.0, 0.5]

prob.run_model()

if 0:
    prob.check_partials(compact_print=True)
    exit()

prob.run_driver()

if 1:
    C_L = prob['wing_sec_C_L'].reshape((num_points_x - 1, num_points_z - 1))[0, :] \
        * 0.5 * (prob['wing_chord'][1:] + prob['wing_chord'][:-1])
    sec_z = 0.5 * (prob['wing_sec_z'][1:] + prob['wing_sec_z'][:-1])
    elliptical = C_L[num_points_z_half - 1] * np.sqrt(np.abs(1 - (sec_z / sec_z[-1]) ** 2))
    plt.plot(sec_z, C_L, 'bo-')
    plt.plot(sec_z, elliptical, 'ro-')
    plt.show()

if 1:
    mesh = prob['wing_mesh']
    vortex_mesh = prob['wing_vortex_mesh']
    collocation_points = prob['coll_pts']
    force_points = prob['force_pts']
    bound_vecs = prob['bound_vecs']
    wing_normals = prob['wing_normals'].reshape((int(np.prod(prob['wing_normals'].shape[:2])), 3))

    fig = plt.figure()

    # plt.subplot(2, 1, 1)
    ax = fig.gca()
    plot_mesh_2d(ax, vortex_mesh, 2, 0, color='b')
    plot_mesh_2d(ax, mesh, 2, 0, color='k')
    scatter_2d(ax, collocation_points, 2, 0, color='b', markersize=3)
    scatter_2d(ax, force_points, 2, 0, color='r', markersize=3)
    arrow_2d(ax, force_points, 0.5 * bound_vecs, 2, 0, color='grey')
    plt.axis('equal')
    plt.xlabel('z')
    plt.ylabel('x')

    print(prob['horseshoe_circulations'].reshape((
        num_points_x - 1, 2 * num_points_z_half - 2 )))

    plt.show()
