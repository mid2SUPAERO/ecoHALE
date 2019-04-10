'''
Compute viscous drag polar by running sweep of anlaysis over different angle of
attacks. Plot drag polar at end. Check output directory for Tecplot solution files.
'''
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

from openmdao.api import IndepVarComp, Problem
import os

# Specify output directory
output_dir = './Outputs/'
# If directory does not exist, then create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Instantiate the problem and the model group
prob = Problem()

# Define flight variables as independent variables of the model
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=248.136, units='m/s') # Freestream Velocity
indep_var_comp.add_output('alpha', val=5., units='deg') # Angle of Attack
indep_var_comp.add_output('beta', val=0., units='deg') # Sideslip angle
indep_var_comp.add_output('Omega', val=np.zeros(3), units='deg/s') # Rotation rate
indep_var_comp.add_output('M', val=0.84) # Freestream Mach number
indep_var_comp.add_output('re', val=1.e6, units='1/m') # Freestream Reynolds number
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3') # Freestream air density
indep_var_comp.add_output('cg', val=np.zeros((3)), units='m') # Aircraft center of gravity
# Add vars to model, promoting is a quick way of automatically connecting inputs
# and outputs of different OpenMDAO components
prob.model.add_subsystem('flight_vars', indep_var_comp, promotes=['*'])

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 35,
             'num_x' : 11,
             'wing_type' : 'rect',
             'symmetry' : True,
             'span' : 10.,
             'chord' : 1,
             'span_cos_spacing' : 1.,
             'chord_cos_spacing' : 1.}

# Generate half-wing mesh of rectangular wing
mesh = generate_mesh(mesh_dict)

# Define input surface dictionary for our wing
surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'type' : 'aero',
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'projected', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'

            'twist_cp' : np.zeros(3), # Define twist using 3 B-spline cp's
                                    # distributed along span
            'mesh' : mesh,

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.0,            # CL of the surface at alpha=0
            'CD0' : 0.0,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c' : 0.12,      # thickness over chord ratio (NACA0015)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,  # if true, compute viscous drag,
            } # end of surface dictionary

# Add geometry to the problem as the name of the surface.
# These groups are responsible for manipulating the geometry of the mesh,
# in this case spanwise twist.
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(surface['name'], geom_group)

# Create the aero point group for this flight condition and add it to the model
aero_group = AeroPoint(surfaces=[surface], output_dir=output_dir)
point_name = 'aero_point_0'
prob.model.add_subsystem(point_name, aero_group, promotes_inputs=['*'])

# Set up the problem
prob.setup()

# Create arrays to hold polar data
alpha = np.linspace(-10,10,20)
CL = np.zeros_like(alpha)
CD = np.zeros_like(alpha)

# Sweep through alphas to create polar
for i in range(len(alpha)):
    # Set alpha
    prob['alpha'] = alpha[i]

    # Run analysis
    prob.run_model()

    # Record CL, CD
    CL[i] = prob['aero_point_0.wing.CL'][0]
    CD[i] = prob['aero_point_0.wing.CD'][0]

# Plot polar
import matplotlib.pyplot as plt
plt.plot(CD*1E4, CL, '-o')
plt.grid(color='lightgray', linestyle='-', linewidth=1)
plt.xlabel('$C_D$ (counts)')
plt.ylabel('$C_L$')
plt.title('Drag polar')
plt.show()
