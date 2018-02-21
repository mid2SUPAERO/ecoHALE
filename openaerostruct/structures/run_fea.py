from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, pyOptSparseDriver, view_model, Group, ExecComp, SqliteRecorder

from openaerostruct.geometry.inputs_group import InputsGroup
from openaerostruct.structures.fea_bspline_group import FEABsplineGroup
from openaerostruct.structures.fea_preprocess_group import FEAPreprocessGroup
from openaerostruct.structures.fea_states_group import FEAStatesGroup
from openaerostruct.structures.fea_postprocess_group import FEAPostprocessGroup

from openaerostruct.common.lifting_surface import LiftingSurface


if __name__ == "__main__":
    num_nodes = 1

    num_points_x = 2
    num_points_z_half = 15
    num_points_z = 2 * num_points_z_half - 1

    wing = LiftingSurface('wing')

    wing.initialize_mesh(num_points_x, num_points_z_half, airfoil_x=np.linspace(0., 1., num_points_x), airfoil_y=np.zeros(num_points_x))
    wing.set_mesh_parameters(distribution='sine', section_origin=.25)
    wing.set_structural_properties(E=70.e9, G=29.e9, spar_location=0.35, sigma_y=200e6, rho=2700)
    wing.set_aero_properties(factor2=.119, factor4=-0.064, cl_factor=1.05)

    wing.set_chord(1.)
    wing.set_twist(0.)
    wing.set_sweep(0.)
    wing.set_dihedral(0.)
    wing.set_span(5.)
    wing.set_thickness(0.05, n_cp=5, order=3)
    wing.set_radius(0.12)

    lifting_surfaces = [('wing', wing)]

    wing_loads = np.zeros((num_nodes, num_points_z, 6))
    wing_loads[0, :, 1] = 1e3
    if num_nodes > 1:
        wing_loads[1, :, 0] = 1e3

    fea_scaler = 1e0

    prob = Problem()
    prob.model = Group()

    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v_m_s', shape=num_nodes, val=200.)
    indep_var_comp.add_output('alpha_rad', shape=num_nodes, val=3. * np.pi / 180.)
    indep_var_comp.add_output('rho_kg_m3', shape=num_nodes, val=1.225)
    indep_var_comp.add_output('wing_loads', val=wing_loads)
    prob.model.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])

    inputs_group = InputsGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces)
    prob.model.add_subsystem('inputs_group', inputs_group, promotes=['*'])

    prob.model.add_subsystem('tube_bspline_group',
        FEABsplineGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
        promotes=['*'])

    prob.model.add_subsystem('fea_preprocess_group',
        FEAPreprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
        promotes=['*'],
    )
    prob.model.add_subsystem('fea_states_group',
        FEAStatesGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces, fea_scaler=fea_scaler),
        promotes=['*'],
    )
    prob.model.add_subsystem('fea_postprocess_group',
        FEAPostprocessGroup(num_nodes=num_nodes, lifting_surfaces=lifting_surfaces),
        promotes=['*'],
    )
    prob.model.add_subsystem('objective',
        ExecComp('obj=1000*sum(compliance)', compliance=np.zeros(num_nodes)),
        promotes=['*'],
    )

    prob.model.add_design_var('wing_tube_thickness_dv', lower=0.001, upper=0.1, scaler=1e3)
    prob.model.add_objective('structural_weight', scaler=1e0)
    prob.model.add_constraint('wing_ks', upper=0.)
    # prob.model.add_constraint('wing_vonmises', upper=0.)
    # prob.model.add_objective('obj', scaler=1e-4)
    # prob.model.add_constraint('structural_volume', upper=0.1)

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major optimality tolerance'] = 2e-7
    prob.driver.opt_settings['Major feasibility tolerance'] = 2e-7
    # prob.driver.opt_settings['Verify level'] = -1

    prob.driver.add_recorder(SqliteRecorder('fea.hst'))
    prob.driver.recording_options['includes'] = ['*']

    prob.setup()

    # view_model(prob)

    prob['wing_chord_dv'] = [0.5, 1.0, 0.5]

    if 0:
        prob.setup(force_alloc_complex=True)
        prob['wing_chord_dv'] = [0.5, 1.0, 0.5]
        prob.run_model()
        prob.check_partials(compact_print=True)
        # print(np.linalg.norm(prob['wing_vonmises'] - prob['wing_vonmises_old']))
        exit()

    print(prob['structural_volume'])

    prob.run_driver()

    print(prob['wing_tube_thickness'])

    print(prob['wing_disp'])


    if 1:
        import matplotlib.pyplot as plt
        for i in range(num_nodes):
            x = prob['wing_fea_mesh'][i, :]
            xx = 0.5 * x[:-1] + 0.5 * x[1:]
            plt.subplot(num_nodes, 3, 3*i + 1)
            plt.plot(xx, prob['wing_tube_thickness'][i, :])
            plt.subplot(num_nodes, 3, 3*i + 2)
            plt.plot(x, prob['wing_disp'][i, :, 0])
            plt.subplot(num_nodes, 3, 3*i + 3)
            plt.plot(x, prob['wing_disp'][i, :, 1])
        plt.show()

    # prob.check_partials(compact_print=True)
