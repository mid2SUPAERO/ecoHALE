# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:12:30 2019

@author: e.duriez and Victor M. Guadano
"""


import numpy as np
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openmdao.api import IndepVarComp, Problem, ScipyOptimizeDriver, SqliteRecorder, ExecComp, NewtonSolver
from openaerostruct.utils.constants import grav_constant
from emittedCO2byStructure import structureCO2
from acceptableThickness import checkThickness
import os
import time
from math import atan, pi
from random import randint
from multiMaterial import material, YoungMM, ShearMM, YieldMM, CO2MM  #VMGM
from pointmasses import PointMassLocations, PointMasses  #VMGM
from AirfoilRadius import RadiusCurvature  #VMGM


# Optimization function
def fctOptim(mrhoi,skin,spar,span,toverc): 
    
    # Starting time
    starttime=time.time()
    
    # Materials
    #sandw1=material(66.35,4.25e9,1.63e9,58.7e6,34.7,"sandw1")
    #sandw2=material(174.5,14.15e9,5.44e9,195.6e6,43.4,"sandw2")
    #sandw3=material(483,42.5e9,16.3e9,586e6,46.8,"sandw3")
    sandw4=material(504.5,42.5e9,16.3e9,586e6,44.9,"sandw4")
    sandw5=material(560.5,42.5e9,16.3e9,586e6,40.3,"sandw5")
    sandw6=material(529,42.5e9,16.3e9,237e6,42.75,"sandw6")
    al7075=material(2.80e3,72.5e9,27e9,444.5e6,13.15*(1-0.426)+2.61*0.426,"al7075") #from EDUPACK
    #al7075oas=material(2.78e3,73.1e9,73.1e9/2/1.33,444.5e6/1.5,13.15*(1-0.426)+2.61*0.426,"al7075") #from OAS example
    qiCFRP=material(1565,54.9e9,21e9,670e6,48.1,"qiCFRP")
    steel=material(7750,200e9,78.5e9,562e6,4.55*(1-0.374)+1.15*0.374,"steel")
    gfrp=material(1860,21.4e9,8.14e9,255e6,6.18,"gfrp")            #epoxy-Eglass,woven,QI
    #nomat=material(1370,0.01,0.01,0.01,60,"noMaterial")
    nomat=material(50,1e8,1e4,1e5,6000,"noMaterial")    
    #nomat=material(50,1e8,1e4,1e5,60,"noMaterial")    
    fakemat=material((2.80e3+7750)/2,(72.5e9+200e9)/2,(27e9+78.5e9)/2,(444.5e6+562e6)/2,(13.15*(1-0.426)+2.61*0.426+4.55*(1-0.374)+1.15*0.374)/2,"fakemat")
    nomatEnd=material(10000,5e9,2e9,20e6,60,"nomatEnd")

    materials=[al7075, qiCFRP, steel, gfrp, nomat, fakemat, nomatEnd, sandw4, sandw5, sandw6]
    
    # Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
    # These should be for an airfoil with the chord scaled to 1.
    # We use the 10% to 60% portion of the NACA 63412 airfoil for this case
    # We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
    # The first and last x-coordinates of the upper and lower surfaces must be the same
    upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
    lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype = 'complex128')
    upper_y = np.array([ 0.0513, 0.0537, 0.0559, 0.0580, 0.0600, 0.0619, 0.0636, 0.0652, 0.0668, 0.0682, 0.0696, 0.0709, 0.0721, 0.0732, 0.0742, 0.0752, 0.0761, 0.0769, 0.0776, 0.0782, 0.0788, 0.0793, 0.0797, 0.0801, 0.0804, 0.0806, 0.0808, 0.0808, 0.0808, 0.0807, 0.0806, 0.0804, 0.0801, 0.0798, 0.0794, 0.0789, 0.0784, 0.0778, 0.0771, 0.0764, 0.0757, 0.0749, 0.0740, 0.0732, 0.0723, 0.0713, 0.0703, 0.0692, 0.0681, 0.0669, 0.0657], dtype = 'complex128')
    lower_y = np.array([-0.0296, -0.0307, -0.0317, -0.0326, -0.0335, -0.0343, -0.0350, -0.0357, -0.0363, -0.0368, -0.0373, -0.0378, -0.0382, -0.0386, -0.0389, -0.0391, -0.0394, -0.0395, -0.0397, -0.0398, -0.0398, -0.0398, -0.0398, -0.0397, -0.0396, -0.0394, -0.0392, -0.0389, -0.0386, -0.0382, -0.0378, -0.0374, -0.0369, -0.0363, -0.0358, -0.0352, -0.0345, -0.0338, -0.0331, -0.0324, -0.0316, -0.0308, -0.0300, -0.0292, -0.0283, -0.0274, -0.0265, -0.0256, -0.0246, -0.0237, -0.0227], dtype = 'complex128')   
    
    Rcurv = RadiusCurvature(upper_x,lower_x,upper_y,lower_y)
    
    # Create a dictionary to store options about the surface
    mesh_dict = {'num_y' : 15,
                 'num_x' : 3,
                 'wing_type' : 'rect',
                 'symmetry' : True,
                 'chord_cos_spacing' : 0,
                 'span_cos_spacing' : 0,
                 'num_twist_cp' : 4
                 }
    
    mesh = generate_mesh(mesh_dict)
    
    # Batteries and solar panels
    densityPV=0.3 #[kg/m^2]
    energeticDensityBattery=400*0.995*0.95*0.875*0.97 #Wh/kg 0.995=battery controller efficiency, 0.95=end of life capacity loss of 5%, 0.97=min battery SOC of 3%, 0.875=packaging efficiency
    emissionBat=0.104/0.995/0.95/0.875/0.97 #[kgCO2/Wh]
    night_hours=13 #h
    productivityPV=54.0*0.97*0.95 #[W/m^2] 54 from FBHALE power figure, 0.97=MPPT efficiency, 0.95=battery round trip efficiency
    emissionPV=0.05/0.97/0.95 #[kgCO2/W] emissions of the needed PV surface to produce 1W
    emissionsPerW=emissionPV+emissionBat*night_hours #[kgCO2/W]
    
    # Dictionary for the lifting surface
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
                
                'airfoil_radius_curvature' : Rcurv,
    
                'twist_cp' : np.array([10., 15., 15., 15.]), # [deg]
                'chord_cp' : [1.5], # [m]
                'span' : span, #[m]
                'taper' : 0.3,
    
                'spar_thickness_cp' : np.array([spar, spar, spar, spar]), # [m]
                'skin_thickness_cp' : np.array([skin/2, skin, skin*1.5, 2*skin]), # [m]
    
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
                'k_lam' : 0.80,         # fraction of chord with laminar
                                        # flow, used for viscous drag
                'c_max_t' : .349,       # chordwise location of maximum thickness

                # Materials
                'materlist' : materials,
                'puissanceMM' : 1,  #power used in muli-material function
                'Nmaterial' : 2,  # number of materials in the model (1 or 2); if 2, different materials for spars and skins
                
                # Structural values
                'strength_factor_for_upper_skin' : 1.0, # the yield stress is multiplied by this factor for the upper skin
                
                'wing_weight_ratio' : 1.,
                'exact_failure_constraint' : False, # if false, use KS function
                
                'struct_weight_relief' : True,
                
                # Motors
                'n_point_masses' : 1,  # number of point masses in the system (half of the wing if symmetry is activated); in this case, the motors (omit option if no point masses)

                # Power
                'productivityPV' : productivityPV, #[W/m^2]
                'densityPV' : densityPV+productivityPV/energeticDensityBattery*night_hours, #[kg/m^2] the weight of the batteries is counted here
                'payload_power' : 361, #[W] payload=150 + avionics=211
                'motor_propeller_efficiency' : 0.84, #thrusting power/electrical power used by propulsion
                'co2PV' : emissionsPerW*productivityPV/(densityPV+productivityPV/energeticDensityBattery*night_hours), #[kgCO2/kg] #co2 burden of PV cells and battery
                'prop_density' : 0.0058, #[kg/W]
                'mppt_density' : 0.00045, #[kg/W]
                }
    
    surfaces = [surf_dict]
    
    # Create the problem and assign the model group
    prob = Problem()
    
    # Add problem information as an independent variables component data for altitude=23240 m and 0 m
    speed=34.5 #m/s
    speed_dive = 1.4 * speed #m/s
    gust_speed=3.4 #m/s
    rho_air=0.055 #kg/m**3
    speed_sound=297 #m/s
    
    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('Mach_number', val=np.array([speed/speed_sound, (speed_dive**2+gust_speed**2)**0.5/speed_sound, 0]))
    indep_var_comp.add_output('v', val=np.array([speed, (speed_dive**2+gust_speed**2)**0.5, 0]), units='m/s')
    indep_var_comp.add_output('re',val=np.array([rho_air*speed*1./(1.4*1e-5), \
                              rho_air*speed_dive*1./(1.4*1e-5), 0]),  units='1/m') #L=10m,
    indep_var_comp.add_output('rho', val=np.array([rho_air, rho_air, 1.225]), units='kg/m**3')
    indep_var_comp.add_output('speed_of_sound', val= np.array([speed_sound, speed_sound, 340]), units='m/s')
    
    indep_var_comp.add_output('W0_without_point_masses', val=20.5,  units='kg') 
    
    indep_var_comp.add_output('load_factor', val=np.array([1., 1.1, 1.]))
    indep_var_comp.add_output('alpha', val=0., units='deg')
    indep_var_comp.add_output('alpha_gust', val=atan(gust_speed/speed_dive)*180/pi, units='deg')
    
    indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
    
    indep_var_comp.add_output('mrho', val=np.array([mrhoi,mrhoi]), units='kg/m**3')
    
    indep_var_comp.add_output('motor_location', val=np.array([-0.3]))  # 1 motor (half of the wing)
    #indep_var_comp.add_output('motor_location', val=np.array([-0.3,-0.6]))  # 2 motors (half of the wing)
    
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
        
    prob.model.add_subsystem('YoungMM', YoungMM(surface=surface), promotes_inputs=['mrho'], promotes_outputs=['young'])  #VMGM
    prob.model.add_subsystem('ShearMM', ShearMM(surface=surface), promotes_inputs=['mrho'], promotes_outputs=['shear'])  #VMGM 
    prob.model.add_subsystem('YieldMM', YieldMM(surface=surface), promotes_inputs=['mrho'], promotes_outputs=['yield'])  #VMGM 
    prob.model.add_subsystem('CO2MM', CO2MM(surface=surface), promotes_inputs=['mrho'], promotes_outputs=['co2'])  #VMGM
    prob.model.add_subsystem('PointMassLocations', PointMassLocations(surface=surface), promotes_inputs=['motor_location', 'span', 'nodes'], promotes_outputs=['point_mass_locations'])  #VMGM        
    prob.model.add_subsystem('PointMasses', PointMasses(surface=surface), promotes_inputs=['PV_surface'], promotes_outputs=['point_masses','prop_mass'])  #VMGM
    
    prob.model.add_subsystem('W0_comp',
        ExecComp('W0 = W0_without_point_masses + prop_mass', units='kg'),
        promotes=['*'])
    
    prob.model.connect('mrho',name+'.struct_setup.structural_mass.mrho')  #ED
    prob.model.connect('young',name+'.struct_setup.assembly.local_stiff.young')  #VMGM   
    prob.model.connect('shear',name+'.struct_setup.assembly.local_stiff.shear')  #VMGM       
    prob.model.connect('wing.span','span')  #VMGM
    prob.model.connect('AS_point_0.total_perf.PV_surface','PV_surface')  #VMGM 
    prob.model.connect(name + '.nodes', 'nodes')  #VMGM

    # Loop through and add a certain number of aerostruct points
    for i in range(3):
    
        point_name = 'AS_point_{}'.format(i)
        # Connect the parameters within the model for each aero point
    
        # Create the aerostruct point group and add it to the model
        AS_point = AerostructPoint(surfaces=surfaces)
    
        prob.model.add_subsystem(point_name, AS_point)
        # Connect flow properties to the analysis point
        prob.model.connect('v', point_name + '.v', src_indices=[i])
        prob.model.connect('Mach_number', point_name + '.Mach_number', src_indices=[i])
        prob.model.connect('re', point_name + '.re', src_indices=[i])
        prob.model.connect('rho', point_name + '.rho', src_indices=[i])
        prob.model.connect('W0', point_name + '.W0')
        prob.model.connect('speed_of_sound', point_name + '.speed_of_sound', src_indices=[i])
        prob.model.connect('empty_cg', point_name + '.empty_cg')
        prob.model.connect('load_factor', point_name + '.load_factor', src_indices=[i])
       
        for surface in surfaces:
    
            name = surface['name']
    
            prob.model.connect('load_factor', point_name + '.coupled.load_factor', src_indices=[i]) #for PV distributed weight
    
            com_name = point_name + '.' + name + '_perf.'
            prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
            prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')
            prob.model.connect('young',com_name+'struct_funcs.vonmises.young')    
            prob.model.connect('shear',com_name+'struct_funcs.vonmises.shear')  
            prob.model.connect('yield',com_name+'struct_funcs.failure.yield')  #VMGM
            prob.model.connect('young',com_name+'struct_funcs.buckling.young')  #VMGM
            prob.model.connect('shear',com_name+'struct_funcs.buckling.shear')  #VMGM
            prob.model.connect(name + '.t_over_c', com_name+'struct_funcs.buckling.t_over_c')  #VMGM
            
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
            prob.model.connect(name + '.Qx', com_name + 'Qx')
    
            prob.model.connect(name + '.spar_thickness', com_name + 'spar_thickness')
            prob.model.connect(name + '.skin_thickness', com_name + 'skin_thickness')
            prob.model.connect(name + '.t_over_c', com_name + 't_over_c')
    
            coupled_name = point_name + '.coupled.' + name
            prob.model.connect('point_masses', coupled_name + '.point_masses')
            prob.model.connect('point_mass_locations', coupled_name + '.point_mass_locations')
            
    prob.model.connect('alpha', 'AS_point_0' + '.alpha')
    prob.model.connect('alpha_gust', 'AS_point_1' + '.alpha')
    prob.model.connect('alpha', 'AS_point_2' + '.alpha')  #VMGM
    
    # Here we add the co2 objective componenet to the model
    prob.model.add_subsystem('emittedco2', structureCO2(surfaces=surfaces), promotes_inputs=['co2'], promotes_outputs=['emitted_co2'])  #VMGM
    prob.model.connect('wing.structural_mass', 'emittedco2.mass')
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'emittedco2.PV_mass')  
    prob.model.connect('wing.spars_mass', 'emittedco2.spars_mass')  #VMGM
    
    #Here we add the thickness constraint to the model
    prob.model.add_subsystem('acceptableThickness', checkThickness(surface=surface), promotes_outputs=['acceptableThickness'] )
    prob.model.connect('wing.geometry.t_over_c_cp','acceptableThickness.t_over_c')
    prob.model.connect('wing.chord_cp','acceptableThickness.chordroot')
    prob.model.connect('wing.skin_thickness_cp','acceptableThickness.skinThickness')
    prob.model.connect('wing.taper','acceptableThickness.taper')
    
    prob.model.connect('wing.struct_setup.PV_areas', 'AS_point_0.coupled.wing.struct_states.PV_areas')
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'AS_point_0.coupled.wing.struct_states.PV_mass')
    prob.model.connect('wing.struct_setup.PV_areas', 'AS_point_1.coupled.wing.struct_states.PV_areas')
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'AS_point_1.coupled.wing.struct_states.PV_mass')
    prob.model.connect('wing.chord_cp','AS_point_1.wing_perf.struct_funcs.chord')
    prob.model.connect('wing.taper','AS_point_1.wing_perf.struct_funcs.taper')
    prob.model.connect('wing.struct_setup.PV_areas', 'AS_point_2.coupled.wing.struct_states.PV_areas')  #VMGM
    prob.model.connect('AS_point_0.total_perf.PV_mass', 'AS_point_2.coupled.wing.struct_states.PV_mass')  #VMGM
    prob.model.connect('wing.chord_cp','AS_point_2.wing_perf.struct_funcs.chord')  #VMGM
    prob.model.connect('wing.taper','AS_point_2.wing_perf.struct_funcs.taper')  #VMGM
    
    # Objective function      
    #prob.model.add_objective('emitted_co2', scaler=1e-4)
    prob.model.add_objective('AS_point_0.total_perf.total_weight', scaler=1e-3)
    
    # Design variables
    prob.model.add_design_var('wing.twist_cp', lower=-15., upper=15., scaler=0.1)  #VMGM
    prob.model.add_design_var('wing.spar_thickness_cp', lower=0.001, upper=0.1, scaler=1e3)
    prob.model.add_design_var('wing.skin_thickness_cp', lower=0.001, upper=0.1, scaler=1e3)
    prob.model.add_design_var('wing.span', lower=1., upper=1000., scaler=0.1)
    prob.model.add_design_var('wing.chord_cp', lower=1.4, upper=500., scaler=1)
    prob.model.add_design_var('wing.taper', lower=0.3, upper=0.99, scaler=10)
    prob.model.add_design_var('wing.geometry.t_over_c_cp', lower=0.01, upper=0.4, scaler=10.)
    prob.model.add_design_var('mrho', lower=504.5, upper=504.5, scaler=0.001) #ED
    #prob.model.add_design_var('mrho', lower=500, upper=8000, scaler=0.001) #ED
    prob.model.add_design_var('motor_location', lower=-1, upper=0, scaler=10.)  #VMGM
    
    # Constraints
    prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
    prob.model.add_constraint('AS_point_0.enough_power', upper=0.) #Make sure needed power stays below the solar power producible by the wing
    prob.model.add_constraint('acceptableThickness', upper=0.) #Make sure skin thickness fits in the wing (to avoid negative spar mass)
    #prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)  #VMGM
    #prob.model.add_constraint('AS_point_0.wing_perf.buckling', upper=0.)  #VMGM
    prob.model.add_constraint('AS_point_1.wing_perf.failure', upper=0.)
    prob.model.add_constraint('AS_point_1.wing_perf.buckling', upper=0.)
    prob.model.add_constraint('AS_point_2.wing_perf.failure', upper=0.)  #VMGM
    prob.model.add_constraint('AS_point_2.wing_perf.buckling', upper=0.)  #VMGM
    prob.model.add_constraint('AS_point_0.coupled.wing.S_ref', upper=200.) # Surface constarint to avoid snowball effect
    
    #prob.model.approx_totals(method='fd', step=5e-7, form='forward', step_calc='rel') 
    #prob.model.nonlinear_solver = newton = NewtonSolver()
    
    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    #prob.driver.options['tol'] = 1e-6
    prob.driver.options['tol'] = 1e-3
    prob.driver.options['maxiter']=250
    #prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','totals']
    
    recorder = SqliteRecorder("aerostructMrhoi"+str(mrhoi)+"sk"+str(skin)+"sr"+str(spar)+"sn"+str(span)+"tc"+str(toverc)+".db")
    prob.driver.add_recorder(recorder)
    
    # We could also just use prob.driver.recording_options['includes']=['*'] here, but for large meshes the database file becomes extremely large. So we just select the variables we need.
    prob.driver.recording_options['includes'] = [
        'alpha', 'rho', 'v', 'cg',
        'alpha_gust', #
        'AS_point_1.cg', 'AS_point_0.cg', #
        'AS_point_0.cg', #ED
        'AS_point_0.coupled.wing_loads.loads',
        'AS_point_1.coupled.wing_loads.loads', #
        'AS_point_2.coupled.wing_loads.loads', #
        'AS_point_0.coupled.wing.normals',
        'AS_point_1.coupled.wing.normals', #
        'AS_point_0.coupled.wing.widths',
        'AS_point_1.coupled.wing.widths', #
        'AS_point_0.coupled.aero_states.wing_sec_forces',
        'AS_point_1.coupled.aero_states.wing_sec_forces', #
        'AS_point_2.coupled.aero_states.wing_sec_forces', #
        'AS_point_0.wing_perf.CL1',
        'AS_point_1.wing_perf.CL1', #
        'AS_point_0.coupled.wing.S_ref',
        'AS_point_1.coupled.wing.S_ref', #
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
        'AS_point_1.wing_perf.vonmises', #
        'AS_point_0.coupled.wing.def_mesh',
        'AS_point_1.coupled.wing.def_mesh', #
        'AS_point_0.total_perf.PV_mass', 
        'AS_point_0.total_perf.total_weight', 
        'AS_point_0.CL',
        'AS_point_0.CD',
        'yield',
        'point_masses',  #VMGM
        'point_mass_locations',  #VMGM
        'motor_location',  #VMGM
        ]
    
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['record_inputs'] = True
    
    # Set up the problem
    prob.setup()
    
    ##prob.run_model() #ED
    ##data = prob.check_partials(out_stream=None, compact_print=True, method='cs') #ED
    ##data = prob.check_partials(out_stream=None, compact_print=True, method='fd') #ED
    ##print(data)  #ED 
    
    
    prob.run_driver()
    ##print (prob.model.list_outputs(values=False, implicit=False))  #VMGM
    endtime=time.time()
    totaltime=endtime-starttime
    print('The wingbox mass (including the wing_weight_ratio) is', prob['wing.structural_mass'][0], '[kg]')
    print('computing time is',totaltime,'[s]')
    print('co2 emissions are',prob['emitted_co2'][0],'[kg]')   
    
    return prob['wing.structural_mass'][0], totaltime, prob['mrho'][0],prob['emitted_co2'][0];