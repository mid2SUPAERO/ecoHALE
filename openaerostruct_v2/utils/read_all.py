from __future__ import print_function
from six import iteritems

from openmdao.recorders.case_reader import CaseReader

def read_hist(filename):

    show_wing = False
    show_tube = False
    show_mission = False

    try:
        data_all_iters = read_struct_hist(filename)
        show_tube = True
    except:
        pass

    try:
        data_all_iters = read_aero_hist(filename)
        show_wing = True
    except:
        pass

    try:
        data_all_iters = read_AS_hist(filename)
        show_wing = True
        show_tube = True
    except:
        import traceback
        traceback.print_exc()
        pass

    try:
        data_all_iters = read_AS_mission_hist(filename)
        show_wing = True
        show_tube = True
        show_mission = True
    except:
        import traceback
        traceback.print_exc()
        pass

    return data_all_iters, show_wing, show_tube, show_mission


def read_struct_hist(filename):

    dvs = {
    }

    states = {
        'thickness': 'tube_bspline_group.wing_thickness_bspline_comp.wing_tube_thickness',
        'radius': 'tube_bspline_group.wing_radius_bspline_comp.wing_tube_radius',
        'fea_mesh': 'fea_preprocess_group.fea_mesh_comp.wing_fea_mesh',
    }

    cons = {
        'vonmises': 'fea_postprocess_group.fea_vonmises_comp.wing_vonmises',
    }

    objs = {
        'structural_volume': 'fea_postprocess_group.fea_volume_comp.structural_volume',
    }

    data_all_iters = []

    cr = CaseReader(filename)

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:

        case = cr.driver_cases.get_case(case_key)

        data = {}

        for key, cr_key in iteritems(dvs):
            data[key] = case.desvars[cr_key]

        for key, cr_key in iteritems(states):
            data[key] = case.sysincludes[cr_key]

        for key, cr_key in iteritems(objs):
            data[key] = case.objectives[cr_key]

        for key, cr_key in iteritems(cons):
            data[key] = case.constraints[cr_key]

        data_all_iters.append(data)

    return data_all_iters

def read_aero_hist(filename):

    dvs = {
        'alpha_rad': 'indep_var_comp.alpha_rad'
    }

    states = {
        'mesh': 'vlm_preprocess_group.vlm_mesh_comp.wing_undeformed_mesh',
        'twist': 'inputs_group.wing_twist_bspline_comp.wing_twist',
        'forces': 'vlm_states3_group.vlm_panel_forces_comp.panel_forces',
        'rho_kg_m3': 'indep_var_comp.rho_kg_m3',
        'v_m_s': 'indep_var_comp.v_m_s',
    }

    cons = {
    }

    objs = {
        'CD': 'objective.obj',
    }

    data_all_iters = []

    cr = CaseReader(filename)

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:

        case = cr.driver_cases.get_case(case_key)

        data = {}

        for key, cr_key in iteritems(dvs):
            data[key] = case.desvars[cr_key]

        for key, cr_key in iteritems(states):
            data[key] = case.sysincludes[cr_key]

        for key, cr_key in iteritems(objs):
            data[key] = case.objectives[cr_key]

        for key, cr_key in iteritems(cons):
            data[key] = case.constraints[cr_key]

        data_all_iters.append(data)

    return data_all_iters

def read_AS_hist(filename):

    dvs = {
        'alpha_rad': 'indep_var_comp.alpha_rad'
    }

    states = {
        'rho_kg_m3': 'indep_var_comp.rho_kg_m3',
        'v_m_s': 'indep_var_comp.v_m_s',
        'twist': 'inputs_group.wing_twist_bspline_comp.wing_twist',
        'thickness': 'tube_bspline_group.wing_thickness_bspline_comp.wing_tube_thickness',
        'radius': 'tube_bspline_group.wing_radius_bspline_comp.wing_tube_radius',
        'forces': 'aerostruct_group.vlm_states3_group.vlm_panel_forces_comp.panel_forces',
        'mesh': 'aerostruct_group.vlm_states1_group.vlm_displace_meshes_comp.wing_mesh',
        'disp': 'aerostruct_group.fea_states_group.fea_disp_comp.wing_disp',
        'fea_mesh': 'fea_preprocess_group.fea_mesh_comp.wing_fea_mesh',
        'vonmises': 'fea_postprocess_group.fea_vonmises_comp.wing_vonmises',
    }

    cons = {
    }

    objs = {
        'obj': 'objective.obj',
    }

    data_all_iters = []

    cr = CaseReader(filename)

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:

        case = cr.driver_cases.get_case(case_key)

        data = {}

        for key, cr_key in iteritems(dvs):
            data[key] = case.desvars[cr_key]

        for key, cr_key in iteritems(states):
            data[key] = case.sysincludes[cr_key]

        for key, cr_key in iteritems(objs):
            data[key] = case.objectives[cr_key]

        for key, cr_key in iteritems(cons):
            data[key] = case.constraints[cr_key]

        data_all_iters.append(data)

    return data_all_iters

def read_AS_mission_hist(filename):

    dvs = {

    }

    states = {
        'alpha_rad': 'mission_group.sys_coupled_analysis.oas_solver_comp.alpha_rad',
        'rho_kg_m3': 'mission_group.atmos.density_comp.rho_kg_m3',
        'v_m_s': 'mission_group.atmos.speed_comp.v_m_s',
        'twist': 'mission_group.inputs_group.wing_twist_bspline_comp.wing_twist',
        'thickness': 'mission_group.tube_bspline_group.wing_thickness_bspline_comp.wing_tube_thickness',
        'radius': 'mission_group.tube_bspline_group.wing_radius_bspline_comp.wing_tube_radius',
        'forces': 'mission_group.sys_coupled_analysis.oas_group.aerostruct_group.vlm_states3_group.vlm_panel_forces_comp.panel_forces',
        'mesh':  'mission_group.sys_coupled_analysis.oas_group.aerostruct_group.vlm_states1_group.vlm_displace_meshes_comp.wing_mesh',
        'disp': 'mission_group.sys_coupled_analysis.oas_group.aerostruct_group.fea_states_group.fea_disp_comp.wing_disp',
        'fea_mesh':  'mission_group.sys_coupled_analysis.oas_group.fea_preprocess_group.fea_mesh_comp.wing_fea_mesh',
        'vonmises':  'mission_group.sys_coupled_analysis.oas_group.fea_postprocess_group.fea_vonmises_comp.wing_vonmises',
        'h_km': 'mission_group.bsplines.comp_h.h_km',
        'x_1e3_km': 'mission_group.bsplines.comp_x.x_1e3_km',
        'C_L': 'mission_group.sys_coupled_analysis.oas_group.vlm_postprocess_group.vlm_modify_coeffs_comp.C_L',
        'C_D': 'mission_group.sys_coupled_analysis.oas_group.vlm_postprocess_group.vlm_modify_coeffs_comp.C_D',
    }

    cons = {
    }

    objs = {
        'obj': 'mission_group.functionals.objective_comp.objective'
    }

    data_all_iters = []

    cr = CaseReader(filename)

    case_keys = cr.driver_cases.list_cases()
    for case_key in case_keys:

        case = cr.driver_cases.get_case(case_key)

        data = {}

        for key, cr_key in iteritems(dvs):
            data[key] = case.desvars[cr_key]

        for key, cr_key in iteritems(states):
            data[key] = case.sysincludes[cr_key]

        for key, cr_key in iteritems(objs):
            data[key] = case.objectives[cr_key]

        for key, cr_key in iteritems(cons):
            data[key] = case.constraints[cr_key]

        data_all_iters.append(data)

    return data_all_iters

def check_length(filename):

    cr = CaseReader(filename)

    case_keys = cr.driver_cases.list_cases()

    return len(case_keys)
