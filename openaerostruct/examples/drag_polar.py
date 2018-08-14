
from __future__ import division, print_function

import numpy as np
import matplotlib.pylab as plt

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
            ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
            DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


def compute_drag_polar(Mach, alphas, surface):

    # Create the OpenMDAO problem
    prob = Problem()
    # Create an independent variable component that will supply the flow
    # conditions to the problem.
    indep_var_comp = IndepVarComp()
    indep_var_comp.add_output('v', val=248.136, units='m/s')
    indep_var_comp.add_output('alpha', val=5., units='deg')
    indep_var_comp.add_output('M', val=Mach)
    indep_var_comp.add_output('re', val=1.e6, units='1/m')
    indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
    indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')
    # Add this IndepVarComp to the problem model
    prob.model.add_subsystem('prob_vars',
        indep_var_comp,
        promotes=['*'])
    # Create and add a group that handles the geometry for the
    # aerodynamic lifting surface
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem('wing', geom_group)
    # Create the aero point group, which contains the actual aerodynamic
    # analyses
    aero_group = AeroPoint(surfaces=[surface])
    point_name = 'aero'
    prob.model.add_subsystem(point_name, aero_group,
        promotes_inputs=['v', 'alpha', 'M', 're', 'rho', 'cg'])

    # Connect the mesh from the geometry component to the analysis point
    prob.model.connect('wing.mesh', 'aero.wing.def_mesh')
    # Perform the connections with the modified names within the
    # 'aero_states' group.
    prob.model.connect('wing.mesh', 'aero.aero_states.wing_def_mesh')

    prob.setup()

    CLs = []
    CDs = []
    for a in alphas:
        prob['alpha'] =  a
        prob.run_model()
        print(a, prob['aero.wing_perf.CL'], prob['aero.wing_perf.CD'])
        CLs.append(prob['aero.wing_perf.CL'][0])
        CDs.append(prob['aero.wing_perf.CD'][0])
        #print(a, prob['aero.wing_perf.CL'], prob['aero.wing_perf.CD'])

    # Plot CL vs alpha and drag polar 
    fig,axes =  plt.subplots(nrows=2)
    axes[0].plot(alphas, CLs)
    axes[1].plot(CLs, CDs)
    fig.savefig('drag_polar.pdf')
    #plt.show()

    return CLs, CDs


if __name__=='__main__':
    # Create a dictionary to store options about the mesh
    mesh_dict = {'num_y' : 7,
                'num_x' : 2,
                'wing_type' : 'CRM',
                'symmetry' : True,
                'num_twist_cp' : 5}

    # Generate the aerodynamic mesh based on the previous dictionary
    mesh, twist_cp = generate_mesh(mesh_dict)

    # Create a dictionary with info and options about the aerodynamic
    # lifting surface
    surface = {
                # Wing definition
                'name' : 'wing',        # name of the surface
                'type' : 'aero',
                'symmetry' : True,     # if true, model one half of wing
                                        # reflected across the plane y = 0
                'S_ref_type' : 'wetted', # how we compute the wing area,
                                        # can be 'wetted' or 'projected'
                'fem_model_type' : 'tube',
                'twist_cp' : twist_cp,
                'mesh' : mesh,
                'num_x' : mesh.shape[0],
                'num_y' : mesh.shape[1],
                # Aerodynamic performance of the lifting surface at
                # an angle of attack of 0 (alpha=0).
                # These CL0 and CD0 values are added to the CL and CD
                # obtained from aerodynamic analysis of the surface to get
                # the total CL and CD.
                # These CL0 and CD0 values do not vary wrt alpha.
                'CL0' : 0.0,            # CL of the surface at alpha=0
                'CD0' : 0.015,            # CD of the surface at alpha=0
                # Airfoil properties for viscous drag calculation
                'k_lam' : 0.05,         # percentage of chord with laminar
                                        # flow, used for viscous drag
                't_over_c' : 0.15,      # thickness over chord ratio (NACA0015)
                'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                        # thickness
                'with_viscous' : True,  # if true, compute viscous drag
                }
    Mach = 0.82
    alphas = np.linspace(-10, 15, 25)
    CL, CD = compute_drag_polar(Mach, alphas, surface)
