# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:12:30 2019

@author: e.duriez
"""

###VERSION WITH FIXED MATERIAL FOR VALIDATION PURPOSE !!!!!

import numpy as np
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openmdao.api import IndepVarComp, Problem, ScipyOptimizeDriver, SqliteRecorder, ExecComp, NewtonSolver
#from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta
from openaerostruct.utils.constants import grav_constant
from fctMultiMatos import material
from emittedCO2byStructure import structureCO2
from acceptableThickness import checkThickness
import os
import time
from math import atan, pi
from random import randint


def fctOptim(mrhoi,skin,spar,span,toverc): 
#def fctOptim(mrhoi,skin,):  #ED2
    
#    puissanceMM=1 #ED2
    starttime=time.time()
    
#    sandw1=material(66.35,4.25e9,1.63e9,58.7e6/1.5,34.7,"sandw1")
#    sandw2=material(174.5,14.15e9,5.44e9,195.6e6/1.5,43.4,"sandw2")
#    sandw3=material(483,42.5e9,16.3e9,586e6/1.5,46.8,"sandw3")
    sandw4=material(504.5,42.5e9,16.3e9,586e6/1.5,44.9,"sandw4")
#    sandw5=material(574.5,42.5e9,16.3e9,586e6/1.5,39.3,"sandw5")
    sandw5=material(560.5,42.5e9,16.3e9,586e6/1.5,40.3,"sandw5")
    sandw6=material(529,42.5e9,16.3e9,237e6/1.5,42.75,"sandw6")




    al7075=material(2.80e3,72.5e9,27e9,444.5e6/1.5,13.15*(1-0.426)+2.61*0.426,"al7075") #from EDUPACK
#    al7075oas=material(2.78e3,73.1e9,73.1e9/2/1.33,444.5e6/1.5,13.15*(1-0.426)+2.61*0.426,"al7075") #from OAS example
    qiCFRP=material(1565,54.9e9,21e9,670e6/1.5,48.1,"qiCFRP")
    steel=material(7750,200e9,78.5e9,562e6/1.5,4.55*(1-0.374)+1.15*0.374,"steel")
    gfrp=material(1860,21.4e9,8.14e9,255e6,6.18,"gfrp")            #epoxy-Eglass,woven,QI
    #nomat=material(1370,0.01,0.01,0.01,60,"noMaterial")
    nomat=material(50,1e8,1e4,1e5,6000,"noMaterial")    
#    nomat=material(50,1e8,1e4,1e5,60,"noMaterial")    
    fakemat=material((2.80e3+7750)/2,(72.5e9+200e9)/2,(27e9+78.5e9)/2,(444.5e6/1.5+562e6/1.5)/2,(13.15*(1-0.426)+2.61*0.426+4.55*(1-0.374)+1.15*0.374)/2,"fakemat")
    nomatEnd=material(10000,5e9,2e9,20e6/1.5,60,"nomatEnd")
    
    materials=[al7075, qiCFRP, steel, gfrp, nomat, fakemat, nomatEnd, sandw4, sandw5, sandw6]
#    materials=[al7075, qiCFRP, steel, gfrp, nomat, fakemat, nomatEnd, sandw3, sandw4, sandw5, sandw6]
#    materials=[al7075, qiCFRP, steel, gfrp, nomat, fakemat, nomatEnd,sandw1,sandw2,sandw3]
#    materials=[al7075, fakemat, steel, nomat]
    
    # Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
    # These should be for an airfoil with the chord scaled to 1.
    # We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
    # We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
    # The first and last x-coordinates of the upper and lower surfaces must be the same
    
    upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
    lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
    upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype = 'complex128')
    lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype = 'complex128')
    
    # Create a dictionary to store options about the surface
    mesh_dict = {'num_y' : 15,
                 'num_x' : 3,
#                 'wing_type' : 'uCRM_based',
                 'wing_type' : 'rect',
                 'symmetry' : True,
                 'chord_cos_spacing' : 0,
                 'span_cos_spacing' : 0,
                 'num_twist_cp' : 4
                 }
    
#    mesh, twist_cp = generate_mesh(mesh_dict)
    mesh = generate_mesh(mesh_dict)
    
    densityPV=0.3 #kg/m^2
#    energeticDensityBattery=400 #Wh/kg
    energeticDensityBattery=400*0.995*0.95*0.875*0.97 #Wh/kg 0.995=battery controller efficiency, 0.95=end of life capacity loss of 5%, 0.97=min battery SOC of 3%, 0.875=packaging efficiency
#    energeticDensityBattery=800 #Wh/kg
#    emissionBat=0.104 #[kgCO2/Wh]
    emissionBat=0.104/0.995/0.95/0.875/0.97 #[kgCO2/Wh]
    night_hours=13 #h
#    night_hours=hours #h #ED2
    productivityPV=54.0*0.97*0.95 #[W/m^2] 54 from FBHALE power figure, 0.97=MPPT efficiency, 0.95=battery round trip efficiency
    emissionPV=0.05/0.97/0.95 #[kgCO2/W] emissions of the needed PV surface to produce 1W
    emissionsPerW=emissionPV+emissionBat*night_hours #[kgCO2/W]
    
    
    surf_dict = {
                # Wing definition
                'name' : 'wing',         # give the surface some name
                'symmetry' : True,       # if True, model only one half of the lifting surface
                'S_ref_type' : 'projected', # how we compute the wing area,
                                         # can be 'wetted' or 'projected'
                'mesh' : mesh,
    
                'fem_model_type' : 'wingbox', # 'wingbox' or 'tube'
                'data_x_upper' : upper_x,
                'data_x_lower' : lower_x,
                'data_y_upper' : upper_y,
                'data_y_lower' : lower_y,
    
                'twist_cp' : np.array([10., 20., 20., 20.]), # [deg]
#                'twist_cp' : np.array([15.        +0.j, 15.        +0.j, 15.        +0.j,  7.32414355+0.j]), # [deg] #TODELETE
    
#                'spar_thickness_cp' : np.array([0.001, 0.001, 0.002, 0.003]), # [m]
                'spar_thickness_cp' : np.array([spar, spar, spar, spar]), # [m]
#                'spar_thickness_cp' : np.array([0.00298327+0.j, 0.00411879+0.j, 0.00702408+0.j, 0.00855659+0.j]), # [m] #TODELETE
#                'skin_thickness_cp' : np.array([0.001, 0.001, 0.002, 0.003]), # [m]
                'skin_thickness_cp' : np.array([skin/2, skin, skin*1.5, 2*skin]), # [m]
#                'skin_thickness_cp' : np.array([0.0001    +0.j, 0.00274638+0.j, 0.00698719+0.j, 0.01217181+0.j]), # [m] #TODELETE
    
#                't_over_c_cp' : np.array([0.08, 0.08, 0.10, 0.08]),
                't_over_c_cp' : np.array([0.75*toverc, toverc, toverc, 1.25*toverc]), #TODELETE
                'original_wingbox_airfoil_t_over_c' : 0.12,
                 # Aerodynamic deltas.
                # These CL0 and CD0 values are added to the CL and CD
                # obtained from aerodynamic analysis of the surface to get
                # the total CL and CD.
                # These CL0 and CD0 values do not vary wrt alpha.
                # They can be used to account for things that are not included, such as contributions from the fuselage, camber, etc.
                'CL0' : 0.0,            # CL delta
                'CD0' : 0.0078,         # CD delta
    
                'with_viscous' : True,  # if true, compute viscous drag
                'with_wave' : True,     # if true, compute wave drag
    
                # Airfoil properties for viscous drag calculation
                'k_lam' : 0.05,         # fraction of chord with laminar
                                        # flow, used for viscous drag
                'c_max_t' : .38,       # chordwise location of maximum thickness
                 # Structural values are based on aluminum 7075
    #            'E' : 73.1e9,              # [Pa] Young's modulus
                'materlist' : materials,
    #            'G' : (73.1e9/2/1.33),     # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
    #            'yield' : (420.e6 / 1.5),  # [Pa] allowable yield stress
    #            'mrho' : 2.80e3,           # [kg/m^3] material density
                'strength_factor_for_upper_skin' : 1.0, # the yield stress is multiplied by this factor for the upper skin
    
                'wing_weight_ratio' : 1.,
#                'wing_weight_ratio' : 1.25,
                'exact_failure_constraint' : False, # if false, use KS function
                'struct_weight_relief' : True,
#                'distributed_fuel_weight' : True,
#                'fuel_density' : 803.,      # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
#                'Wf_reserve' :15000.,       # [kg] reserve fuel mass
                'puissanceMM' : 1,    #power used in muli-material function
    #            'mrho' : 2000.,
#                'span' : 50., #[m]
                'span' : span, #[m]
#                'span' : [100.+0.j], #TODELETE
                'taper' : 0.3,
#                'taper' : [1.+0.j], #TODELETE
#                'taper' : 0.99,
                'chord_cp' : [1.5],
#                'chord_cp' : [2.],
#                'chord_cp' : [32.49012623+0.j], #TODELETE
                'productivityPV' : productivityPV, #[W/m^2]
                'densityPV' : densityPV+productivityPV/energeticDensityBattery*night_hours, #[kg/m^2] the weight of the batteries is counted here
                'payload_power' : 361, #[W] payload=150 + avionics=211
                'motor_propeller_efficiency' : 0.84, #thrusting power/electrical power used by propulsion
                'co2PV' : emissionsPerW*productivityPV/(densityPV+productivityPV/energeticDensityBattery*night_hours), #[kgCO2/kg] #co2 burden of PV cells and battery
                'prop_density' : 0.0058, #[kg/W]
                'mppt_density' : 0.00045, #[kg/W]
                'buckling_coef' : 4, #buckling coeficient
                'inter_stringer' : 0.25, #[m] distance between two stringers
                }
    
    surfaces = [surf_dict]
    
    # Create the problem and assign the model group
    prob = Problem()
    
    # Add problem information as an independent variables component data for altitude=23240 m
    speed=34.5 #m/s
    gust_speed=3.4 #m/s
    rho_air=0.055 #kg/m**3
    speed_sound=297 #m/s
    indep_var_comp = IndepVarComp()
#    indep_var_comp.add_output('Mach_number', val=np.array([0.85, 0.64]))
    indep_var_comp.add_output('Mach_number', val=np.array([speed/speed_sound, (speed**2+gust_speed**2)**0.5/speed_sound]))
#    indep_var_comp.add_output('v', val=np.array([.85 * 295.07, .64 * 340.294]), units='m/s')
    indep_var_comp.add_output('v', val=np.array([speed, (speed**2+gust_speed**2)**0.5]), units='m/s')
    indep_var_comp.add_output('re',val=np.array([rho_air*speed*10./(1.4*1e-5), \
                              rho_air*speed*10./(1.4*1e-5)]),  units='1/m') #L=10m,
    indep_var_comp.add_output('rho', val=np.array([0.055, 0.055]), units='kg/m**3')
#    indep_var_comp.add_output('speed_of_sound', val= np.array([295.07, 340.294]), units='m/s')
    indep_var_comp.add_output('speed_of_sound', val= np.array([speed_sound, speed_sound]), units='m/s')
    
#    indep_var_comp.add_output('CT', val=0.53/3600, units='1/s')
#    indep_var_comp.add_output('R', val=14.307e6, units='m')
#    indep_var_comp.add_output('W0', val=148000 + surf_dict['Wf_reserve'],  units='kg')
    indep_var_comp.add_output('W0', val=20.5,  units='kg')
    
    indep_var_comp.add_output('load_factor', val=np.array([1., 1.]))
    indep_var_comp.add_output('alpha', val=0., units='deg')
    indep_var_comp.add_output('alpha_gust', val=atan(gust_speed/speed)*180/pi, units='deg')
#    indep_var_comp.add_output('alpha_maneuver', val=[15.+0.j], units='deg') #TODELETE
    indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
    
#    indep_var_comp.add_output('fuel_mass', val=10000., units='kg')
    
    indep_var_comp.add_output('mrho', val=mrhoi, units='kg/m**3')
#    indep_var_comp.add_output('mrho', val=[1570.+0.j], units='kg/m**3') #TODELETE
    
    prob.model.add_subsystem('prob_vars',
         indep_var_comp,
         promotes=['*'])
    
    
    
    # Loop over each surface in the surfaces list
    for surface in surfaces:
    
        # Get the surface name and create a group to contain components
        # only for this surface
        name = surface['name']
    
        aerostruct_group = AerostructGeometry(surface=surface)
    
        # Add groups to the problem with the name of the surface.
        prob.model.add_subsystem(name, aerostruct_group)
        
        
    prob.model.connect('mrho',name+'.struct_setup.structural_mass.mrho')  #ED
    prob.model.connect('mrho',name+'.struct_setup.assembly.local_stiff.mrho')  #ED
    
    
    
        
    # Loop through and add a certain number of aerostruct points
    for i in range(2):
#    for i in range(1):
    
        point_name = 'AS_point_{}'.format(i)
        # Connect the parameters within the model for each aero point
    
        # Create the aerostruct point group and add it to the model
#        AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)
        AS_point = AerostructPoint(surfaces=surfaces)
    
        prob.model.add_subsystem(point_name, AS_point)
        # Connect flow properties to the analysis point
        prob.model.connect('v', point_name + '.v', src_indices=[i])
        prob.model.connect('Mach_number', point_name + '.Mach_number', src_indices=[i])
        prob.model.connect('re', point_name + '.re', src_indices=[i])
        prob.model.connect('rho', point_name + '.rho', src_indices=[i])
#        prob.model.connect('CT', point_name + '.CT')
#        prob.model.connect('R', point_name + '.R')
        prob.model.connect('W0', point_name + '.W0')
        prob.model.connect('speed_of_sound', point_name + '.speed_of_sound', src_indices=[i])
        prob.model.connect('empty_cg', point_name + '.empty_cg')
        prob.model.connect('load_factor', point_name + '.load_factor', src_indices=[i])
#        prob.model.connect('fuel_mass', point_name + '.total_perf.L_equals_W.fuelburn')
#        prob.model.connect('fuel_mass', point_name + '.total_perf.CG.fuelburn')
        prob.model.connect('mrho', point_name + '.mrho')
    
    
        for surface in surfaces:
    
            name = surface['name']
    
#            if surf_dict['distributed_fuel_weight']:
#                prob.model.connect('load_factor', point_name + '.coupled.load_factor', src_indices=[i])
            prob.model.connect('load_factor', point_name + '.coupled.load_factor', src_indices=[i]) #for PV distributed weight
    
            com_name = point_name + '.' + name + '_perf.'
            prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
            prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')
            prob.model.connect('mrho',com_name+'struct_funcs.vonmises.mrho')  #ED
            prob.model.connect('mrho',com_name+'struct_funcs.failure.mrho')  #ED
    
            # Connect aerodyamic mesh to coupled group mesh
            prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')
            if surf_dict['struct_weight_relief']:
                prob.model.connect(name + '.element_mass', point_name + '.coupled.' + name + '.element_mass')
    
            # Connect performance calculation variables
            prob.model.connect(name + '.nodes', com_name + 'nodes')
            prob.model.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
            prob.model.connect(name + '.structural_mass', point_name + '.' + 'total_perf.' + name + '_structural_mass')
    
            # Connect wingbox properties to von Mises stress calcs
            prob.model.connect(name + '.Qz', com_name + 'Qz')
            prob.model.connect(name + '.J', com_name + 'J')
            prob.model.connect(name + '.A_enc', com_name + 'A_enc')
            prob.model.connect(name + '.htop', com_name + 'htop')
            prob.model.connect(name + '.hbottom', com_name + 'hbottom')
            prob.model.connect(name + '.hfront', com_name + 'hfront')
            prob.model.connect(name + '.hrear', com_name + 'hrear')
    
            prob.model.connect(name + '.spar_thickness', com_name + 'spar_thickness')
            prob.model.connect(name + '.skin_thickness', com_name + 'skin_thickness')
            prob.model.connect(name + '.t_over_c', com_name + 't_over_c')
    #        prob.model.connect(name + '.mrho', com_name + 'mrho')
            
    prob.model.connect('alpha', 'AS_point_0' + '.alpha')
    prob.model.connect('alpha_gust', 'AS_point_1' + '.alpha')
#    prob.model.connect('alpha_maneuver', 'AS_point_1' + '.alpha')
    
#    # Here we add the fuel volume constraint componenet to the model
#    prob.model.add_subsystem('fuel_vol_delta', WingboxFuelVolDelta(surface=surface))
#    prob.model.connect('wing.struct_setup.fuel_vols', 'fuel_vol_delta.fuel_vols')
#    prob.model.connect('AS_point_0.fuelburn', 'fuel_vol_delta.fuelburn')
    
    # Here we add the co2 objective componenet to the model
    prob.model.add_subsystem('emittedco2', structureCO2(surfaces=surfaces),promotes_inputs=['mrho'],promotes_outputs=['emitted_co2'])
    prob.model.connect('wing.structural_mass', 'emittedco2.mass')
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'emittedco2.PV_mass')
    
#    prob.model.connect('wing.twist_cp', 'emittedco2.twist') #TODELETE    
#    prob.model.connect('wing.spar_thickness_cp', 'emittedco2.spar_thickness') #TODELETE    
#    prob.model.connect('wing.skin_thickness_cp', 'emittedco2.skin_thickness') #TODELETE    
#    prob.model.connect('wing.span', 'emittedco2.span') #TODELETE    
#    prob.model.connect('wing.chord_cp', 'emittedco2.chord') #TODELETE    
#    prob.model.connect('wing.taper', 'emittedco2.taper') #TODELETE    
#    prob.model.connect('wing.geometry.t_over_c_cp', 'emittedco2.t_over_c') #TODELETE    
#    prob.model.connect('alpha_maneuver', 'emittedco2.alpha_maneuver') #TODELETE    
#    prob.model.connect('mrho', 'emittedco2.mrhoVar') #TODELETE    
    
    #Here we add the thickness constraint to the model
    prob.model.add_subsystem('acceptableThickness', checkThickness(surface=surface), promotes_outputs=['acceptableThickness'] )
    prob.model.connect('wing.geometry.t_over_c_cp','acceptableThickness.t_over_c')
    prob.model.connect('wing.chord_cp','acceptableThickness.chordroot')
    prob.model.connect('wing.skin_thickness_cp','acceptableThickness.skinThickness')
    prob.model.connect('wing.taper','acceptableThickness.taper')
    
    
#    if surf_dict['distributed_fuel_weight']:
#        prob.model.connect('wing.struct_setup.fuel_vols', 'AS_point_0.coupled.wing.struct_states.fuel_vols')
#        prob.model.connect('fuel_mass', 'AS_point_0.coupled.wing.struct_states.fuel_mass')
    prob.model.connect('wing.struct_setup.PV_areas', 'AS_point_0.coupled.wing.struct_states.PV_areas')
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'AS_point_0.coupled.wing.struct_states.PV_mass')
    
#        prob.model.connect('wing.struct_setup.fuel_vols', 'AS_point_1.coupled.wing.struct_states.fuel_vols')
#        prob.model.connect('fuel_mass', 'AS_point_1.coupled.wing.struct_states.fuel_mass')
    prob.model.connect('wing.struct_setup.PV_areas', 'AS_point_1.coupled.wing.struct_states.PV_areas')
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'AS_point_1.coupled.wing.struct_states.PV_mass')
    
    prob.model.connect('wing.chord_cp','AS_point_1.wing_perf.struct_funcs.chord')
    prob.model.connect('wing.taper','AS_point_1.wing_perf.struct_funcs.taper')
        
#    comp = ExecComp('fuel_diff = (fuel_mass - fuelburn) / fuelburn')
#    prob.model.add_subsystem('fuel_diff', comp,
#        promotes_inputs=['fuel_mass'],
#        promotes_outputs=['fuel_diff'])
#    prob.model.connect('AS_point_0.fuelburn', 'fuel_diff.fuelburn')
        
    prob.model.add_objective('emitted_co2', scaler=1e-4)
    
    prob.model.add_design_var('wing.twist_cp', lower=-30., upper=30., scaler=0.1)
#    prob.model.add_design_var('wing.spar_thickness_cp', lower=0.0001, upper=0.1, scaler=1e3)
    prob.model.add_design_var('wing.spar_thickness_cp', lower=0.0001, upper=0.1, scaler=1e4)
#    prob.model.add_design_var('wing.skin_thickness_cp', lower=0.0001, upper=0.1, scaler=1e3)
    prob.model.add_design_var('wing.skin_thickness_cp', lower=0.0001, upper=0.1, scaler=1e3)
    prob.model.add_design_var('wing.span', lower=1., upper=1000., scaler=0.1)
    prob.model.add_design_var('wing.chord_cp', lower=1.4, upper=500., scaler=1)
    prob.model.add_design_var('wing.taper', lower=0.3, upper=0.99, scaler=10)
#    prob.model.add_design_var('wing.taper', lower=0.01, upper=0.99, scaler=10)
    prob.model.add_design_var('wing.geometry.t_over_c_cp', lower=0.01, upper=0.4, scaler=10.)
#    prob.model.add_design_var('alpha_maneuver', lower=-15., upper=15)
    prob.model.add_design_var('mrho', lower=mrhoi, upper=mrhoi, scaler=0.001) #ED
#    prob.model.add_design_var('mrho', lower=mrhoi, upper=mrhoi, scaler=0.001) #ED
    
#    prob.model.add_constraint('AS_point_0.CL', equals=0.5)
    
#    prob.model.add_constraint('AS_point_1.L_equals_W', equals=0.)
#    prob.model.add_constraint('AS_point_1.wing_perf.failure', upper=0.)
#    prob.model.add_constraint('AS_point_1.enough_power', upper=0.) #Make sure needed power stays below the solar power producible by the wing
    prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
    prob.model.add_constraint('AS_point_1.wing_perf.failure', upper=0.)
    prob.model.add_constraint('AS_point_0.enough_power', upper=0.) #Make sure needed power stays below the solar power producible by the wing
    prob.model.add_constraint('acceptableThickness', upper=0.) #Make sure skin thickness fits in the wing (to avoid negative spar mass)
    prob.model.add_constraint('AS_point_1.wing_perf.buckling', upper=0.)
    
#    prob.model.add_constraint('fuel_vol_delta.fuel_vol_delta', lower=0.)
    
#    prob.model.add_design_var('fuel_mass', lower=0., upper=2e5, scaler=1e-5)
#    prob.model.add_constraint('fuel_diff', equals=0.)
    
    #prob.model.approx_totals(method='fd', step=5e-7, form='forward', step_calc='rel')
    
#    prob.model.nonlinear_solver = newton = NewtonSolver()
    
    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-4
    prob.driver.options['maxiter']=1000
    #prob.driver.options['tol'] = 1e-3
#    prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','totals']
    
    recorder = SqliteRecorder("aerostructMrhoi"+str(mrhoi)+"sk"+str(skin)+"sr"+str(spar)+"sn"+str(span)+"tc"+str(toverc)+".db")
    prob.driver.add_recorder(recorder)
    
    # We could also just use prob.driver.recording_options['includes']=['*'] here, but for large meshes the database file becomes extremely large. So we just select the variables we need.
    prob.driver.recording_options['includes'] = [
        'alpha', 'rho', 'v', 'cg',
#        'AS_point_1.cg', 'AS_point_0.cg',
        'AS_point_0.cg', #ED
        'AS_point_0.coupled.wing_loads.loads',
#        'AS_point_1.coupled.wing_loads.loads',
        'AS_point_0.coupled.wing.normals',
#        'AS_point_1.coupled.wing.normals',
        'AS_point_0.coupled.wing.widths',
#        'AS_point_1.coupled.wing.widths',
        'AS_point_0.coupled.aero_states.wing_sec_forces',
#        'AS_point_1.coupled.aero_states.wing_sec_forces',
        'AS_point_0.wing_perf.CL1',
#        'AS_point_1.wing_perf.CL1',
        'AS_point_0.coupled.wing.S_ref',
#        'AS_point_1.coupled.wing.S_ref',
        'wing.geometry.twist',
        'wing.geometry.mesh.taper.taper',
        'wing.geometry.mesh.stretch.span',
        'wing.geometry.mesh.scale_x.chord',
        'wing.mesh',
        'wing.skin_thickness',
        'wing.spar_thickness',
        'wing.t_over_c',
        'wing.structural_mass',
        'AS_point_0.wing_perf.vonmises',
#        'AS_point_1.wing_perf.vonmises',
        'AS_point_0.coupled.wing.def_mesh',
#        'AS_point_1.coupled.wing.def_mesh',
        'AS_point_0.total_perf.PV_mass', 
        'AS_point_0.total_perf.total_weight', 
        'AS_point_0.CL',
        'AS_point_0.CD',
        
        ]
    
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['record_inputs'] = True
    
    # Set up the problem
    prob.setup()
#    prob.run_model() #ED2
#
#    data = prob.check_partials(out_stream=None, compact_print=True, method='cs') #ED2
#    print(data)  #ED2   
#    from openmdao.api import view_model
#    view_model(prob)
    
    
    prob.run_driver()
    print('The wingbox mass (including the wing_weight_ratio) is', prob['wing.structural_mass'][0], '[kg]')
    endtime=time.time()
    totaltime=endtime-starttime
    print('computing time is',totaltime)
    print('co2 emissions are',prob['emitted_co2'][0])
    
    
    maxconstraint=max(abs(prob['AS_point_0.L_equals_W']),prob['AS_point_1.wing_perf.failure'],prob['AS_point_0.enough_power'],max(prob['acceptableThickness']),prob['AS_point_1.wing_perf.buckling'])
    return prob['wing.structural_mass'][0], totaltime, prob['mrho'][0],prob['emitted_co2'][0], maxconstraint;