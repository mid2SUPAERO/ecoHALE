"""
The OASProblem class contains all of the methods necessary to setup and run
aerostructural optimization using OpenAeroStruct.
"""

# =============================================================================
# Standard Python modules
# =============================================================================
from __future__ import division
import sys
from time import time
import numpy

# =============================================================================
# OpenMDAO modules
# =============================================================================
from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from openmdao.devtools.partition_tree_n2 import view_tree

# =============================================================================
# OpenAeroStruct modules
# =============================================================================
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_rect_mesh
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals, VLMGeometry
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium
from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx


class OASProblem():
    """
    Contain surface and problem information for aerostructural optimization.

    Parameters
    ----------
    input_dict : dictionary
        The problem conditions and type of analysis desired. Note that there
        are default values defined by `get_default_prob_dict` that are overwritten
        based on the user-provided input_dict.
    """

    def __init__(self, input_dict={}):

        # Update prob_dict with user-provided values after getting defaults
        self.prob_dict = self.get_default_prob_dict()
        self.prob_dict.update(input_dict)

        # Set the airspeed velocity based on the inputted Mach number
        # and speed of sound
        self.prob_dict['v'] = self.prob_dict['M'] * self.prob_dict['a']
        self.surfaces = []

        # Set the setup function depending on the problem type selected by the user
        if self.prob_dict['type'] == 'aero':
            self.setup = self.setup_aero
        if self.prob_dict['type'] == 'struct':
            self.setup = self.setup_struct
        if self.prob_dict['type'] == 'aerostruct':
            self.setup = self.setup_aerostruct

    def get_default_surf_dict(self):
        """
        Obtain the default settings for the surface descriptions. Note that
        these defaults are overwritten based on user input for each surface.
        Each dictionary describes one surface.
        """

        defaults = {'name' : 'wing',        # name of the surface
                    'num_x' : 3,            # number of chordwise points
                    'num_y' : 5,            # number of spanwise points
                    'span' : 10.,           # full wingspan
                    'chord' : 1.,           # root chord
                    'span_cos_spacing' : 1,   # 0 for uniform spanwise panels
                                            # 1 for cosine-spaced panels
                                            # any value between 0 and 1 for
                                            # a mixed spacing
                    'chord_cos_spacing' : 0,   # 0 for uniform chordwise panels
                                            # 1 for cosine-spaced panels
                                            # any value between 0 and 1 for
                                            # a mixed spacing
                    'dihedral' : 0.,        # wing dihedral angle in degrees
                                            # positive is upward
                    'sweep' : 0.,           # wing sweep angle in degrees
                                            # positive sweeps back
                    'taper' : 1.,           # taper ratio; 1. is uniform chord

                    'CL0' : 0.0,            # CL value at AoA (alpha) = 0
                    'CD0' : 0.0,            # CD value at AoA (alpha) = 0

                    # Structural values are based on aluminum
                    'E' : 70.e9,            # [Pa] Young's modulus of the spar
                    'G' : 30.e9,            # [Pa] shear modulus of the spar
                    'stress' : 20.e6,       # [Pa] yield stress
                    'mrho' : 3.e3,          # [kg/m^3] material density
                    'fem_origin' : 0.35,    # chordwise location of the spar
                    'symmetry' : False,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'W0' : 0.5 * 2.5e6,     # [N] MTOW of B777 is 3e5 kg with fuel
                    'wing_type' : 'rect',   # initial shape of the wing
                                            # either 'CRM' or 'rect'
                    'offset' : numpy.array([0., 0., 0.]) # coordinates to offset
                                    # the surface from its default location
                    }
        return defaults

    def get_default_prob_dict(self):
        """
        Obtain the default settings for the problem description. Note that
        these defaults are overwritten based on user input for the problem.
        """

        defaults = {'optimize' : False,     # flag for analysis or optimization
                    'Re' : 0.,              # Reynolds number
                    'alpha' : 5.,           # angle of attack
                    'CT' : 9.81 * 17.e-6,   # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
                    'R' : 14.3e6,           # [m] maximum range
                    'M' : 0.84,             # Mach number at cruise
                    'rho' : 0.38,           # [kg/m^3] air density at 35,000 ft
                    'a' : 295.4,            # [m/s] speed of sound at 35,000 ft
                    }

        return defaults


    def add_surface(self, input_dict={}):
        """
        Add a surface to the problem. One surface definition is needed for
        each planar lifting surface.

        Parameters
        ----------
        input_dict : dictionary
            Surface definition. Note that there are default values defined by
            `get_default_surf_dict` that are overwritten based on the
            user-provided input_dict.
        """

        # Get defaults and update surf_dict with the user-provided input
        surf_dict = self.get_default_surf_dict()
        surf_dict.update(input_dict)

        # Check to see if the user provides the mesh points. If they do,
        # get the chordwise and spanwise number of points
        if 'mesh' in surf_dict.keys():
            mesh = surf_dict['mesh']
            num_x, num_y = mesh.shape

        # If the user doesn't provide a mesh, obtain the values from surf_dict
        # to create the mesh
        elif 'num_x' in surf_dict.keys():
            num_x = surf_dict['num_x']
            num_y = surf_dict['num_y']
            span = surf_dict['span']
            chord = surf_dict['chord']
            span_cos_spacing = surf_dict['span_cos_spacing']
            chord_cos_spacing = surf_dict['chord_cos_spacing']

            # Generate rectangular mesh
            if surf_dict['wing_type'] == 'rect':
                mesh = gen_rect_mesh(num_x, num_y, span, chord,
                    span_cos_spacing, chord_cos_spacing)

            # Generate CRM mesh
            elif surf_dict['wing_type'] == 'CRM':
                npi = int(((num_y - 1) / 2) * .6)
                npo = int(npi * 5 / 3)
                mesh = gen_crm_mesh(n_points_inboard=npi, n_points_outboard=npo, num_x=num_x)
                num_x, num_y = mesh.shape[:2]

            else:
                print 'Error: wing_type option not understood. Must be either "CRM" or "rectangular".'

            # Chop the mesh in half if using symmetry during analysis.
            # Note that this means that the provided mesh should be the full mesh
            if surf_dict['symmetry']:
                num_y = int((num_y+1)/2)
                mesh = mesh[:, :num_y, :]

        else:
            print "Error: Please either provide a mesh or a valid set of parameters."

        # Apply the user-provided coordinate offset to position the mesh
        mesh = mesh + surf_dict['offset']

        # Get the spar radius
        r = radii(mesh)

        # Set the number of twist and thickness control points.
        # These b-spline control points are what the optimizer sees
        # and controls
        if 'num_twist' not in input_dict.keys():
            surf_dict['num_twist'] = numpy.max([int((num_y - 1) / 5), 5])
        if 'num_thickness' not in input_dict.keys():
            surf_dict['num_thickness'] = numpy.max([int((num_y - 1) / 5), 5])

        # Store updated values
        surf_dict['num_x'] = num_x
        surf_dict['num_y'] = num_y
        surf_dict['mesh'] = mesh
        surf_dict['r'] = r
        surf_dict['t'] = r / 10

        # Set default loads at the tips
        loads = numpy.zeros((r.shape[0] + 1, 6), dtype='complex')
        loads[0, 2] = 1e3
        if not surf_dict['symmetry']:
            loads[-1, 2] = 1e3
        surf_dict['loads'] = loads

        # Throw a warning if the user provides two surfaces with the same name
        name = surf_dict['name']
        for surface in self.surfaces:
            if name == surface['name']:
                print "Warning: Two surfaces have the same name."

        # Append '_' to each repeated surface name
        if not name:
            surf_dict['name'] = name
        else:
            surf_dict['name'] = name + '_'

        # Add the individual surface description to the surface list
        self.surfaces.append(surf_dict)

    def setup_prob(self):
        """
        Short method to select the optimizer. Uses SNOPT if available,
        or SLSQP otherwise.
        """

        try:  # Use SNOPT optimizer if installed
            from openmdao.api import pyOptSparseDriver
            self.prob.driver = pyOptSparseDriver()
            self.prob.driver.options['optimizer'] = "SNOPT"
            self.prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-6,
                                        'Major feasibility tolerance': 1.0e-6}
        except:  # Use SLSQP optimizer if SNOPT not installed
            self.prob.driver = ScipyOptimizer()
            self.prob.driver.options['optimizer'] = 'SLSQP'
            self.prob.driver.options['disp'] = True
            self.prob.driver.options['tol'] = 1.0e-6

    def add_desvar(self, *args, **kwargs):
        """
        Helper function that calls the OpenMDAO method to add design variables.
        """
        self.prob.driver.add_desvar(*args, **kwargs)

    def add_constraint(self, *args, **kwargs):
        """
        Helper function that calls the OpenMDAO method to add constraints.
        """
        self.prob.driver.add_constraint(*args, **kwargs)

    def add_objective(self, *args, **kwargs):
        """
        Helper function that calls the OpenMDAO method to add objectives.
        """
        self.prob.driver.add_objective(*args, **kwargs)

    def run(self):
        """
        Method to actually run analysis or optimization. Also saves history in
        a .db file and creates an N2 diagram to view the problem hierarchy.
        """

        # Uncomment this to use finite differences over the entire model
        # self.prob.root.deriv_options['type'] = 'fd'

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        self.prob.driver.add_recorder(SqliteRecorder(self.prob_dict['prob_name']+".db"))

        # Profile (time) the problem
        # profile.setup(self.prob)
        # profile.start()

        # Set up the problem
        self.prob.setup()

        # Uncomment this line to have more verbose output about convergence
        # self.prob.print_all_convergence()

        # Save an N2 diagram for the problem
        view_tree(self.prob, outfile=self.prob_dict['prob_name']+".html", show_browser=False)

        # Run a single analysis loop to populate uninitialized values
        self.prob.run_once()

        # If `optimize` == True in prob_dict, perform optimization. Otherwise,
        # simply pass the problem since analysis has already been run.
        if not self.prob_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            self.prob.run()

        # Uncomment this to check the partial derivatives of each component
        # self.prob.check_partial_derivatives(compact_print=True)


    def setup_struct(self):
        """
        Specific method to add the necessary components to the problem for a
        structural problem.
        """

        # Set the problem name if the user doesn't
        if 'prob_name' not in self.prob_dict.keys():
            self.prob_dict['prob_name'] = 'struct'

        # Create the base root-level group
        root = Group()

        # Create the problem and assign the root group
        self.prob = Problem()
        self.prob.root = root

        # Loop over each surface in the surfaces list
        for surface in self.surfaces:

            # Get the surface name and create a group to contain components
            # only for this surface.
            # This group's name is whatever the surface's name is.
            # The default is 'wing'.
            name = surface['name']
            tmp_group = Group()

            surface['r'] = surface['r'] / 5
            surface['t'] = surface['r'] / 20

            # Add independent variables that do not belong to a specific component.
            # Note that these are the only ones necessary for structual-only
            # analysis and optimization.
            indep_vars = [
                ('thickness_cp', numpy.ones(surface['num_thickness'])*numpy.max(surface['t'])),
                ('r', surface['r']),
                ('loads', surface['loads'])]

            # Obtain the Jacobians to interpolate the data from the b-spline
            # control points
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])
            jac_thickness = get_bspline_mtx(surface['num_thickness'], surface['num_y']-1)

            # Add structural components to the surface-specific group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('twist_bsp',
                     Bspline('twist_cp', 'twist', jac_twist),
                     promotes=['*'])
            tmp_group.add('thickness_bsp',
                     Bspline('thickness_cp', 'thickness', jac_thickness),
                     promotes=['*'])
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('tube',
                     MaterialsTube(surface),
                     promotes=['*'])
            tmp_group.add('struct_states',
                     SpatialBeamStates(surface),
                     promotes=['*'])
            tmp_group.add('struct_funcs',
                     SpatialBeamFunctionals(surface),
                     promotes=['*'])

            # Add tmp_group to the problem with the name of the surface.
            # The default is 'wing'.
            nm = name
            name = name[:-1]
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=[])')

        # Actually set up the problem
        self.setup_prob()

    def setup_aero(self):
        """
        Specific method to add the necessary components to the problem for an
        aerodynamic problem.
        """

        # Set the problem name if the user doesn't
        if 'prob_name' not in self.prob_dict.keys():
            self.prob_dict['prob_name'] = 'aero'

        # Create the base root-level group
        root = Group()

        # Create the problem and assign the root group
        self.prob = Problem()
        self.prob.root = root

        # Loop over each surface in the surfaces list
        for surface in self.surfaces:

            # Get the surface name and create a group to contain components
            # only for this surface
            name = surface['name']
            tmp_group = Group()

            # Add independent variables that do not belong to a specific component
            indep_vars = [
                ('twist_cp', numpy.zeros(surface['num_twist'])),
                ('dihedral', surface['dihedral']),
                ('sweep', surface['sweep']),
                ('span', surface['span']),
                ('taper', surface['taper']),
                ('disp', numpy.zeros((surface['num_y'], 6)))]

            # Obtain the Jacobian to interpolate the data from the b-spline
            # control points
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])

            # Add aero components to the surface-specific group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('twist_bsp',
                     Bspline('twist_cp', 'twist', jac_twist),
                     promotes=['*'])
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('def_mesh',
                     TransferDisplacements(surface),
                     promotes=['*'])
            tmp_group.add('vlmgeom',
                     VLMGeometry(surface),
                     promotes=['*'])

            # Add tmp_group to the problem as the name of the surface.
            # Note that is a group and performance group for each
            # individual surface.
            name_orig = name.strip('_')
            name = name_orig
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=[])')

            # Add a performance group for each surface
            name = name_orig + '_perf'
            exec('root.add("' + name + '", ' + 'VLMFunctionals(surface)' + ', promotes=["v", "alpha", "M", "Re", "rho"])')

        # Add problem information as an independent variables component
        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('Re', self.prob_dict['Re']),
            ('rho', self.prob_dict['rho'])]
        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])

        # Add a single 'aero_states' component that solves for the circulations
        # and forces from all the surfaces.
        # While other components only depends on a single surface,
        # this component requires information from all surfaces because
        # each surface interacts with the others.
        root.add('aero_states',
                 VLMStates(self.surfaces, self.prob_dict),
                 promotes=['circulations', 'v', 'alpha', 'rho'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        # This is necessary because the VLMStates component requires information
        # from each surface, but this information is stored within each
        # surface's group.
        for surface in self.surfaces:
            name = surface['name']

            # Perform the connections with the modified names within the
            # 'aero_states' group.
            root.connect(name[:-1] + '.def_mesh', 'aero_states.' + name + 'def_mesh')
            root.connect(name[:-1] + '.b_pts', 'aero_states.' + name + 'b_pts')
            root.connect(name[:-1] + '.c_pts', 'aero_states.' + name + 'c_pts')
            root.connect(name[:-1] + '.normals', 'aero_states.' + name + 'normals')

            # Connect the results from 'aero_states' to the performance groups
            root.connect('aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

            # Connect S_ref for performance calcs
            root.connect(name[:-1] + '.S_ref', name + 'perf' + '.S_ref')

        # Actually set up the problem
        self.setup_prob()

    def setup_aerostruct(self):
        """
        Specific method to add the necessary components to the problem for an
        aerostructural problem.
        """

        # Set the problem name if the user doesn't
        if 'prob_name' not in self.prob_dict.keys():
            self.prob_dict['prob_name'] = 'aerostruct'

        # Create the base root-level group
        root = Group()
        coupled = Group()

        # Create the problem and assign the root group
        self.prob = Problem()
        self.prob.root = root

        # Loop over each surface in the surfaces list
        for surface in self.surfaces:

            # Get the surface name and create a group to contain components
            # only for this surface
            name = surface['name']
            tmp_group = Group()

            # Add independent variables that do not belong to a specific component
            indep_vars = [
                ('twist_cp', numpy.zeros(surface['num_twist'])),
                ('thickness_cp', numpy.ones(surface['num_thickness'])*numpy.max(surface['t'])),
                ('r', surface['r']),
                ('dihedral', surface['dihedral']),
                ('sweep', surface['sweep']),
                ('span', surface['span']),
                ('taper', surface['taper'])]

            # Obtain the Jacobians to interpolate the data from the b-spline
            # control points
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])
            jac_thickness = get_bspline_mtx(surface['num_thickness'], surface['num_y']-1)

            # Add components to include in the surface's group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('twist_bsp',
                     Bspline('twist_cp', 'twist', jac_twist),
                     promotes=['*'])
            tmp_group.add('thickness_bsp',
                     Bspline('thickness_cp', 'thickness', jac_thickness),
                     promotes=['*'])
            tmp_group.add('tube',
                     MaterialsTube(surface),
                     promotes=['*'])

            # Add tmp_group to the problem with the name of the surface.
            name_orig = name
            name = name[:-1]
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=[])')

            # Add components to the 'coupled' group for each surface.
            # The 'coupled' group must contain all components and parameters
            # needed to converge the aerostructural system.
            tmp_group = Group()
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('def_mesh',
                     TransferDisplacements(surface),
                     promotes=['*'])
            tmp_group.add('aero_geom',
                     VLMGeometry(surface),
                     promotes=['*'])
            tmp_group.add('struct_states',
                     SpatialBeamStates(surface),
                     promotes=['*'])
            tmp_group.struct_states.ln_solver = LinearGaussSeidel()

            name = name_orig
            exec(name + ' = tmp_group')
            exec('coupled.add("' + name[:-1] + '", ' + name + ', promotes=[])')

            # Add a loads component to the coupled group
            exec('coupled.add("' + name_orig + 'loads' + '", ' + 'TransferLoads(surface)' + ', promotes=[])')

            # Add a performance group which evaluates the data after solving
            # the coupled system
            tmp_group = Group()

            tmp_group.add('struct_funcs',
                     SpatialBeamFunctionals(surface),
                     promotes=['*'])
            tmp_group.add('aero_funcs',
                     VLMFunctionals(surface),
                     promotes=['*'])

            name = name_orig + 'perf'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["rho", "v", "alpha", "Re", "M"])')

        # Add a single 'aero_states' component for the whole system within the
        # coupled group.
        coupled.add('aero_states',
                 VLMStates(self.surfaces, self.prob_dict),
                 promotes=['v', 'alpha', 'rho'])

        # Explicitly connect parameters from each surface's group and the common
        # 'aero_states' group.
        for surface in self.surfaces:
            name = surface['name']

            # Perform the connections with the modified names within the
            # 'aero_states' group.
            root.connect('coupled.' + name[:-1] + '.def_mesh', 'coupled.aero_states.' + name + 'def_mesh')
            root.connect('coupled.' + name[:-1] + '.b_pts', 'coupled.aero_states.' + name + 'b_pts')
            root.connect('coupled.' + name[:-1] + '.c_pts', 'coupled.aero_states.' + name + 'c_pts')
            root.connect('coupled.' + name[:-1] + '.normals', 'coupled.aero_states.' + name + 'normals')

            # Connect the results from 'aero_states' to the performance groups
            root.connect('coupled.aero_states.' + name + 'sec_forces', name + 'perf' + '.sec_forces')

            # Connect the results from 'coupled' to the performance groups
            root.connect('coupled.' + name[:-1] + '.def_mesh', 'coupled.' + name + 'loads.def_mesh')
            root.connect('coupled.aero_states.' + name + 'sec_forces', 'coupled.' + name + 'loads.sec_forces')
            root.connect('coupled.' + name + 'loads.loads', name + 'perf.loads')

            # Connect the output of the loads component with the FEM
            # displacement parameter. This links the coupling within the coupled
            # group that necessitates the subgroup solver.
            root.connect('coupled.' + name + 'loads.loads', 'coupled.' + name[:-1] + '.loads')

            # Connect aerodyamic design variables
            root.connect(name[:-1] + '.dihedral', 'coupled.' + name[:-1] + '.dihedral')
            root.connect(name[:-1] + '.span', 'coupled.' + name[:-1] + '.span')
            root.connect(name[:-1] + '.sweep', 'coupled.' + name[:-1] + '.sweep')
            root.connect(name[:-1] + '.taper', 'coupled.' + name[:-1] + '.taper')
            root.connect(name[:-1] + '.twist', 'coupled.' + name[:-1] + '.twist')

            # Connect structural design variables
            root.connect(name[:-1] + '.A', 'coupled.' + name[:-1] + '.A')
            root.connect(name[:-1] + '.Iy', 'coupled.' + name[:-1] + '.Iy')
            root.connect(name[:-1] + '.Iz', 'coupled.' + name[:-1] + '.Iz')
            root.connect(name[:-1] + '.J', 'coupled.' + name[:-1] + '.J')

            # Connect performance calculation variables
            root.connect(name[:-1] + '.r', name + 'perf.r')
            root.connect(name[:-1] + '.A', name + 'perf.A')

            # Connection performance functional variables
            root.connect(name + 'perf.weight', 'fuelburn.' + name + 'weight')
            root.connect(name + 'perf.weight', 'eq_con.' + name + 'weight')
            root.connect(name + 'perf.L', 'eq_con.' + name + 'L')
            root.connect(name + 'perf.CL', 'fuelburn.' + name + 'CL')
            root.connect(name + 'perf.CD', 'fuelburn.' + name + 'CD')

            # Connect paramters from the 'coupled' group to the performance
            # group.
            root.connect('coupled.' + name[:-1] + '.nodes', name + 'perf.nodes')
            root.connect('coupled.' + name[:-1] + '.disp', name + 'perf.disp')
            root.connect('coupled.' + name[:-1] + '.S_ref', name + 'perf.S_ref')

        # Set solver properties for the coupled group
        coupled.ln_solver = ScipyGMRES()
        coupled.ln_solver.options['iprint'] = 1
        coupled.ln_solver.preconditioner = LinearGaussSeidel()
        coupled.aero_states.ln_solver = LinearGaussSeidel()

        coupled.nl_solver = NLGaussSeidel()
        coupled.nl_solver.options['iprint'] = 1

        # Ensure that the groups are ordered correctly within the coupled group
        # so that a system with multiple surfaces is solved corretly.
        order_list = []
        for surface in self.surfaces:
            order_list.append(surface['name'][:-1])
        order_list.append('aero_states')
        for surface in self.surfaces:
            order_list.append(surface['name']+'loads')
        coupled.set_order(order_list)

        # Add the coupled group to the root problem
        root.add('coupled', coupled, promotes=['v', 'alpha', 'rho'])

        # Add problem information as an independent variables component
        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('Re', self.prob_dict['Re']),
            ('rho', self.prob_dict['rho'])]
        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])

        # Add functionals to evaluate performance of the system.
        # Note that only the interesting results are promoted here; not all
        # of the parameters.
        root.add('fuelburn',
                 FunctionalBreguetRange(self.surfaces, self.prob_dict),
                 promotes=['fuelburn'])
        root.add('eq_con',
                 FunctionalEquilibrium(self.surfaces, self.prob_dict),
                 promotes=['eq_con', 'fuelburn'])

        # Actually set up the system
        self.setup_prob()
