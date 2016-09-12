"""
The OASProblem class contains all of the methods necessary to setup and run aerostructural optimization using OpenAeroStruct.
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
from openmdao.devtools.partition_tree_n2 import view_model

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

        defaults = {'name' : '',            # name of the surface
                    'num_x' : 3,            # number of chordwise points
                    'num_y' : 5,            # number of spanwise points
                    'span' : 10.,           # full wingspan
                    'chord' : 1.,           # root chord
                    'cosine_spacing' : 1,   # 0 for uniform spanwise panels
                                            # 1 for cosine-spaced panels
                                            # any value between 0 and 1 for
                                            # a mixed spacing
                    'dihedral' : 0.,        # wing dihedral angle in degrees
                                            # positive is upward
                    'sweep' : 0.,           # wing sweep angle in degrees
                                            # positive sweeps back
                    'taper' : 1.,           # taper ratio; 1. is uniform chord

                    'CL0' : 0.2,            # CL value at AoA (alpha) = 0
                    'CD0' : 0.015,          # CD value at AoA (alpha) = 0

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
            cosine_spacing = surf_dict['cosine_spacing']

            # Generate rectangular mesh
            if surf_dict['wing_type'] == 'rect':
                mesh = gen_rect_mesh(num_x, num_y, span, chord, cosine_spacing)

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
            # TODO: note this more explicitly in the docs somewhere
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
            self.prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-7,
                                        'Major feasibility tolerance': 1.0e-7}
        except:  # Use SLSQP optimizer if SNOPT not installed
            self.prob.driver = ScipyOptimizer()
            self.prob.driver.options['optimizer'] = 'SLSQP'
            self.prob.driver.options['disp'] = True
            self.prob.driver.options['tol'] = 1.0e-7

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
        # prob.root.deriv_options['type'] = 'fd'

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        self.prob.driver.add_recorder(SqliteRecorder(self.prob_dict['prob_name']+".db"))

        # Set up the problem
        self.prob.setup()

        # Uncomment this line to have more verbose output about convergence
        # self.prob.print_all_convergence()

        # Save an N2 diagram for the problem
        view_model(self.prob, outfile=self.prob_dict['prob_name']+".html", show_browser=False)

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
            # only for this surface
            name = surface['name']
            tmp_group = Group()

            # TODO: check these values
            surface['r'] = surface['r'] / 5
            surface['t'] = surface['r'] / 20

            # Add independent variables that do not belong to a specific component
            indep_vars = [
                (name+'twist_cp', numpy.zeros(surface['num_twist'])),
                (name+'thickness_cp', numpy.ones(surface['num_thickness'])*numpy.max(surface['t'])),
                (name+'dihedral', surface['dihedral']),
                (name+'sweep', surface['sweep']),
                (name+'span', surface['span']),
                (name+'taper', surface['taper']),
                (name+'r', surface['r']),
                (name+'loads', surface['loads'])]

            # Obtain the Jacobians to interpolate the data from the b-spline
            # control points
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])
            jac_thickness = get_bspline_mtx(surface['num_thickness'], surface['num_y']-1)

            # Add structural components to the surface-specific group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('twist_bsp',
                     Bspline(name+'twist_cp', name+'twist', jac_twist),
                     promotes=['*'])
            tmp_group.add('thickness_bsp',
                     Bspline(name+'thickness_cp', name+'thickness', jac_thickness),
                     promotes=['*'])
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('tube',
                     MaterialsTube(surface),
                     promotes=['*'])
            tmp_group.add('spatialbeamstates',
                     SpatialBeamStates(surface),
                     promotes=['*'])
            tmp_group.add('spatialbeamfuncs',
                     SpatialBeamFunctionals(surface),
                     promotes=['*'])

            # Add tmp_group to the problem with the name of the surface
            name = name + 'struct'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

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
                (name+'twist_cp', numpy.zeros(surface['num_twist'])),
                (name+'dihedral', surface['dihedral']),
                (name+'sweep', surface['sweep']),
                (name+'span', surface['span']),
                (name+'taper', surface['taper']),
                (name+'disp', numpy.zeros((surface['num_y'], 6)))]

            # Obtain the Jacobian to interpolate the data from the b-spline
            # control points
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])

            # Add aero components to the surface-specific group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('twist_bsp',
                     Bspline(name+'twist_cp', name+'twist', jac_twist),
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

            # Add tmp_group to the problem with the name of the surface and
            # '_pre_solve' appended.
            # Note that is a '_pre_solve' and '_post_solve' group for each
            # individual surface.
            name_orig = name.strip('_')
            name = name_orig + '_pre_solve'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

            # Add a '_post_solve' group
            name = name_orig + '_post_solve'
            exec('root.add("' + name + '", ' + 'VLMFunctionals(surface)' + ', promotes=["*"])')

        # Add problem information as an independent variables component
        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('Re', self.prob_dict['Re']),
            ('rho', self.prob_dict['rho'])]
        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])

        # Add a single 'VLMStates' component that solves for the circulations
        # and forces from all the surfaces.
        # While other components only depends on a single surface,
        # this component requires information from all surfaces because
        # each surface interacts with the others.
        root.add('vlmstates',
                 VLMStates(self.surfaces, self.prob_dict),
                 promotes=['*'])

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
                (name+'twist_cp', numpy.zeros(surface['num_twist'])),
                (name+'thickness_cp', numpy.ones(surface['num_thickness'])*numpy.max(surface['t'])),
                (name+'r', surface['r']),
                (name+'dihedral', surface['dihedral']),
                (name+'sweep', surface['sweep']),
                (name+'span', surface['span']),
                (name+'taper', surface['taper'])]

            # Obtain the Jacobians to interpolate the data from the b-spline
            # control points
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])
            jac_thickness = get_bspline_mtx(surface['num_thickness'], surface['num_y']-1)

            # Add components to include in the '_pre_solve' group
            tmp_group.add('indep_vars',
                     IndepVarComp(indep_vars),
                     promotes=['*'])
            tmp_group.add('twist_bsp',
                     Bspline(name+'twist_cp', name+'twist', jac_twist),
                     promotes=['*'])
            tmp_group.add('thickness_bsp',
                     Bspline(name+'thickness_cp', name+'thickness', jac_thickness),
                     promotes=['*'])
            tmp_group.add('tube',
                     MaterialsTube(surface),
                     promotes=['*'])

            # Add tmp_group to the problem with the name of the surface and
            # '_pre_solve' appended.
            name_orig = name#.strip('_')
            name = name + 'pre_solve'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

            # Add components to the 'coupled' group for each surface
            tmp_group = Group()
            tmp_group.add('mesh',
                     GeometryMesh(surface),
                     promotes=['*'])
            tmp_group.add('def_mesh',
                     TransferDisplacements(surface),
                     promotes=['*'])
            tmp_group.add('vlmgeom',
                     VLMGeometry(surface),
                     promotes=['*'])
            tmp_group.add('spatialbeamstates',
                     SpatialBeamStates(surface),
                     promotes=['*'])
            tmp_group.spatialbeamstates.ln_solver = LinearGaussSeidel()

            name = name_orig + 'group'
            exec(name + ' = tmp_group')
            exec('coupled.add("' + name + '", ' + name + ', promotes=["*"])')

            # Add a loads component to the coupled group
            exec('coupled.add("' + name_orig + 'loads' + '", ' + 'TransferLoads(surface)' + ', promotes=["*"])')

            # Add a '_post_solve' group which evaluates the data after solving
            # the coupled system
            tmp_group = Group()
            tmp_group.add('spatialbeamfuncs',
                     SpatialBeamFunctionals(surface),
                     promotes=['*'])
            tmp_group.add('vlmfuncs',
                     VLMFunctionals(surface),
                     promotes=['*'])

            name = name_orig + 'post_solve'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

        # Add a single 'VLMStates' component for the whole system
        coupled.add('vlmstates',
                 VLMStates(self.surfaces, self.prob_dict),
                 promotes=['*'])

        # Set solver properties for the coupled group
        coupled.ln_solver = ScipyGMRES()
        coupled.ln_solver.options['iprint'] = 1
        coupled.ln_solver.preconditioner = LinearGaussSeidel()
        coupled.vlmstates.ln_solver = LinearGaussSeidel()

        coupled.nl_solver = HybridGSNewton()
        coupled.nl_solver.nlgs.options['iprint'] = 1
        coupled.nl_solver.nlgs.options['maxiter'] = 10
        coupled.nl_solver.nlgs.options['atol'] = 1e-8
        coupled.nl_solver.nlgs.options['rtol'] = 1e-12
        coupled.nl_solver.newton.options['atol'] = 1e-7
        coupled.nl_solver.newton.options['rtol'] = 1e-7
        coupled.nl_solver.newton.options['maxiter'] = 5
        coupled.nl_solver.newton.options['iprint'] = 1

        # Ensure that the groups are ordered correctly within the coupled group
        order_list = []
        for surface in self.surfaces:
            order_list.append(surface['name']+'group')
        order_list.append('vlmstates')
        for surface in self.surfaces:
            order_list.append(surface['name']+'loads')
        coupled.set_order(order_list)

        # Add the coupled group to the root problem
        root.add('coupled', coupled, promotes=['*'])

        # Add problem information as an independent variables component
        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('Re', self.prob_dict['Re']),
            ('rho', self.prob_dict['rho'])]
        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])

        # Add functionals to evaluate performance of the system
        root.add('fuelburn',
                 FunctionalBreguetRange(self.surfaces, self.prob_dict),
                 promotes=['*'])
        root.add('eq_con',
                 FunctionalEquilibrium(self.surfaces, self.prob_dict),
                 promotes=['*'])

        self.setup_prob()


    # TODO: change all kw checks to lower()
