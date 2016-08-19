""" Class modules to contain aero, strucutral, and
aerostructural problem formulations.
"""


from __future__ import division
import sys
from time import time
import numpy

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from openmdao.devtools.partition_tree_n2 import view_tree
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx

def get_default_dict():
    defaults = {'num_x' : 3,
                'num_y' : 5,
                'span' : 10.,
                'chord' : 1.,
                'cosine_spacing' : 1,
                'dihedral' : 0.,
                'sweep' : 0.,
                'taper' : 1.,
                'Re' : 0.,
                'alpha' : 5.,
                'optimize' : False,
                'W0' : 0.5 * 2.5e6, # [N] (MTOW of B777 is 3e5 kg with fuel)
                'CT' : 9.81 * 17.e-6, # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
                'R' : 14.3e6, # [m] maximum range
                'M' : 0.84, # at cruise
                'rho' : 0.38, # [kg/m^3] at 35,000 ft
                'a' : 295.4, # [m/s] at 35,000 ft
                'CL0' : 0.2,
                'CD0' : 0.015,
                'E' : 70.e9, # [Pa]
                'G' : 30.e9, # [Pa]
                'stress' : 20.e6, # [Pa]
                'mrho' : 3.e3, # [kg/m^3]
                'symmetry' : False,
                'wing_type' : 'rectangular' # set this to choose the initial shape of the wing
                                    # either 'CRM' or 'rectangular'
                }
    return defaults

class BaseClass():

    def __init__(self, input_dict={}):

        self.options_dict = get_default_dict()
        self.options_dict.update(input_dict)

    def create_mesh(self):
        if 'mesh' in self.options_dict.keys():
            mesh = self.options_dict['mesh']
            fem_ind = self.options_dict['fem_ind']
            aero_ind = self.options_dict['aero_ind']

            num_x, num_y = aero_ind[0]
            r = radii(mesh[:num_x*num_y, 3].reshape(num_x, num_y, 3))
            num_twist = numpy.max([int((num_y - 1) / 5), 5])
        elif 'num_x' in self.options_dict.keys():
            num_x = self.options_dict['num_x']
            num_y = self.options_dict['num_y']
            span = self.options_dict['span']
            chord = self.options_dict['chord']
            cosine_spacing = self.options_dict['cosine_spacing']

            if self.options_dict['wing_type'] == 'rectangular':
                mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
            elif self.options_dict['wing_type'] == 'CRM':
                npi = int(((num_y - 1) / 2) * .6)
                npo = int(npi * 5 / 3)
                mesh = gen_crm_mesh(n_points_inboard=npi, n_points_outboard=npo, num_x=num_x)
                num_x, num_y = mesh.shape[:2]

                if self.options_dict['symmetry']:
                    num_y = int((num_y+1)/2)
                    mesh = mesh[:, :num_y, :]
            else:
                print 'Error: wing_type option not understood. Must be either "CRM" or "rectangular".'

            if self.options_dict['symmetry']:
                num_y = int((num_y+1)/2)
                mesh = mesh[:, :num_y, :]

            num_twist = numpy.max([int((num_y - 1) / 5), 5])

            r = radii(mesh)
            mesh = mesh.reshape(-1, mesh.shape[-1])
            aero_ind = numpy.atleast_2d(numpy.array([num_x, num_y]))
            fem_ind = [num_y]
        else:
            print "Error: Please either provide a flattened mesh or a valid set of parameters."

        return mesh, aero_ind, fem_ind, num_twist, r

    def setup_prob(self, root):
        # Set the optimization problem settings
        prob = Problem()
        prob.root = root

        try:  # Use SNOPT optimizer if installed
            from openmdao.api import pyOptSparseDriver
            prob.driver = pyOptSparseDriver()
            prob.driver.options['optimizer'] = "SNOPT"
            prob.driver.opt_settings = {'Major optimality tolerance': 1.0e-8,
                                        'Major feasibility tolerance': 1.0e-8}
        except:  # Use SLSQP optimizer if SNOPT not installed
            prob.driver = ScipyOptimizer()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['disp'] = True
            prob.driver.options['tol'] = 1.0e-8

        return prob


class Aero(BaseClass):

    def run(self):

        mesh, aero_ind, fem_ind, num_twist, _ = self.create_mesh()
        num_x, num_y = aero_ind[0]

        # Compute the aero and fem indices
        aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

        # Define Jacobians for b-spline controls
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        jac = get_bspline_mtx(num_twist, num_y)

        v = self.options_dict['a'] * self.options_dict['M']

        # Define the independent variables
        # TODO: merge two dictionaries here, using the options_dict directly
        des_vars = [
            ('twist_cp', numpy.zeros(num_twist)),
            ('dihedral', self.options_dict['dihedral']),
            ('sweep', self.options_dict['sweep']),
            ('span', self.options_dict['span']),
            ('taper', self.options_dict['taper']),
            ('v', v),
            ('alpha', self.options_dict['alpha']),
            ('rho', self.options_dict['rho']),
            ('disp', numpy.zeros((tot_n_fem, 6))),
            ('aero_ind', aero_ind),
            ('fem_ind', fem_ind),
            ('Re', self.options_dict['Re'])
        ]

        # Create the top-level system
        root = Group()

        # Add VLM components to the top-level system
        root.add('des_vars',
                 IndepVarComp(des_vars),
                 promotes=['*'])
        root.add('twist_bsp',
                 Bspline('twist_cp', 'twist', jac),
                 promotes=['*'])
        root.add('mesh',
                 GeometryMesh(mesh, aero_ind),
                 promotes=['*'])
        root.add('def_mesh',
                 TransferDisplacements(aero_ind, fem_ind),
                 promotes=['*'])
        root.add('vlmstates',
                 VLMStates(aero_ind, self.options_dict['symmetry']),
                 promotes=['*'])
        root.add('vlmfuncs',
                 VLMFunctionals(aero_ind, self.options_dict['CL0'], self.options_dict['CD0']),
                 promotes=['*'])

        prob = self.setup_prob(root)

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('twist_cp', lower=-10., upper=15., scaler=1e0)
        # prob.driver.add_desvar('alpha', lower=-10., upper=10.)
        prob.driver.add_desvar('sweep', lower=-10., upper=30.)
        prob.driver.add_desvar('dihedral', lower=-10., upper=20.)
        prob.driver.add_desvar('taper', lower=.5, upper=2.)

        # Set the objective (minimize CD on the main wing)
        prob.driver.add_objective('CD_wing', scaler=1e4)

        # Set the constraint (CL = 0.5 for the main wing)
        prob.driver.add_constraint('CL_wing', equals=0.5)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('vlm.db'))

        # Can finite difference over the entire model
        # Generally faster than using component derivatives
        prob.root.deriv_options['type'] = 'fd'

        # Setup the problem
        prob.setup()

        prob.run_once()
        if not self.options_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        return prob

class AeroStruct(BaseClass):

    def run(self):

        mesh, aero_ind, fem_ind, num_twist, r = self.create_mesh()
        num_x, num_y = aero_ind[0]

        aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

        # Set the number of thickness control points and the initial thicknesses
        num_thickness = num_twist
        t = r / 10

        if self.options_dict['symmetry']:
            self.options_dict['W0'] /= 2.

        # Create the top-level system
        root = Group()

        # Define Jacobians for b-spline controls
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        num_surf = fem_ind.shape[0]
        jac_twist = get_bspline_mtx(num_twist, num_y)
        jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

        v = self.options_dict['a'] * self.options_dict['M']

        # Define the independent variables
        indep_vars = [
            ('span', self.options_dict['span']),
            ('twist_cp', numpy.zeros(num_twist)),
            ('thickness_cp', numpy.ones(num_thickness)*numpy.max(t)),
            ('v', v),
            ('alpha', self.options_dict['alpha']),
            ('rho', self.options_dict['rho']),
            ('r', r),
            ('Re', self.options_dict['Re']),  # set Re=0 if you don't want skin friction drag added
            ('M', self.options_dict['M']),
            ('aero_ind', aero_ind),
            ('fem_ind', fem_ind)
        ]

        # Add material components to the top-level system
        root.add('indep_vars',
                 IndepVarComp(indep_vars),
                 promotes=['*'])
        root.add('twist_bsp',
                 Bspline('twist_cp', 'twist', jac_twist),
                 promotes=['*'])
        root.add('thickness_bsp',
                 Bspline('thickness_cp', 'thickness', jac_thickness),
                 promotes=['*'])
        root.add('tube',
                 MaterialsTube(fem_ind),
                 promotes=['*'])

        # Create a coupled group to contain the aero, sruct, and transfer components
        coupled = Group()
        coupled.add('mesh',
                    GeometryMesh(mesh, aero_ind),
                    promotes=['*'])
        coupled.add('def_mesh',
                    TransferDisplacements(aero_ind, fem_ind),
                    promotes=['*'])
        coupled.add('vlmstates',
                    VLMStates(aero_ind, self.options_dict['symmetry']),
                    promotes=['*'])
        coupled.add('loads',
                    TransferLoads(aero_ind, fem_ind),
                    promotes=['*'])
        coupled.add('spatialbeamstates',
                    SpatialBeamStates(aero_ind, fem_ind, self.options_dict['E'], self.options_dict['G']),
                    promotes=['*'])

        # Set solver properties
        coupled.ln_solver = ScipyGMRES()
        coupled.ln_solver.preconditioner = LinearGaussSeidel()
        coupled.vlmstates.ln_solver = LinearGaussSeidel()
        coupled.spatialbeamstates.ln_solver = LinearGaussSeidel()

        coupled.nl_solver = HybridGSNewton()   ### Uncomment this out to use Hybrid GS Newton
        coupled.nl_solver.nlgs.options['maxiter'] = 10
        coupled.nl_solver.nlgs.options['atol'] = 1e-8
        coupled.nl_solver.nlgs.options['rtol'] = 1e-12
        coupled.nl_solver.newton.options['atol'] = 1e-7
        coupled.nl_solver.newton.options['rtol'] = 1e-7
        coupled.nl_solver.newton.options['maxiter'] = 5

        # Add the coupled group and functional groups to compute performance
        root.add('coupled',
                 coupled,
                 promotes=['*'])
        root.add('vlmfuncs',
                 VLMFunctionals(aero_ind, self.options_dict['CL0'], self.options_dict['CD0']),
                 promotes=['*'])
        root.add('spatialbeamfuncs',
                 SpatialBeamFunctionals(aero_ind, fem_ind, self.options_dict['E'],
                                        self.options_dict['G'],
                                        self.options_dict['stress'],
                                        self.options_dict['mrho']),
                                        promotes=['*'])
        root.add('fuelburn',
                 FunctionalBreguetRange(self.options_dict['W0'], self.options_dict['CT'], self.options_dict['a'], self.options_dict['R'], self.options_dict['M'], aero_ind),
                 promotes=['*'])
        root.add('eq_con',
                 FunctionalEquilibrium(self.options_dict['W0'], aero_ind),
                 promotes=['*'])

        prob = self.setup_prob(root)

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('twist_cp',lower= -15.,
                               upper=15., scaler=1e0)
        prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1e0)
        prob.driver.add_desvar('thickness_cp',
                               lower= 0.01,
                               upper= 0.25, scaler=1e2)

        # Set the objective (minimize fuelburn)
        prob.driver.add_objective('fuelburn', scaler=1e-5)

        # Set the constraints (no structural failure and lift = weight)
        prob.driver.add_constraint('failure', upper=0.0)
        prob.driver.add_constraint('eq_con', equals=0.0)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

        # Set up the problem
        prob.setup()

        prob.run_once()

        if not self.options_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        return prob

class Struct(BaseClass):

    def run(self):

        mesh, aero_ind, fem_ind, num_twist, r = self.create_mesh()
        num_x, num_y = aero_ind[0]

        aero_ind, fem_ind = get_inds(aero_ind, fem_ind)

        # Set the number of thickness control points and the initial thicknesses
        num_thickness = num_twist
        t = r / 10

        if self.options_dict['symmetry']:
            self.options_dict['W0'] /= 2.

        # Create the top-level system
        root = Group()

        # Define Jacobians for b-spline controls
        tot_n_fem = numpy.sum(fem_ind[:, 0])
        num_surf = fem_ind.shape[0]
        loads = numpy.zeros((tot_n_fem, 6))
        loads[0, 2] = loads[-1, 2] = 1e3  # tip load of 1 kN
        jac_twist = get_bspline_mtx(num_twist, num_y)
        jac_thickness = get_bspline_mtx(num_thickness, tot_n_fem-num_surf)

        # Define the independent variables
        indep_vars = [
            ('twist_cp', numpy.zeros(num_twist)),
            ('thickness_cp', numpy.ones(num_thickness)*numpy.max(t)),
            ('dihedral', self.options_dict['dihedral']),
            ('sweep', self.options_dict['sweep']),
            ('span', self.options_dict['span']),
            ('taper', self.options_dict['taper']),
            ('r', r),
            ('loads', loads),
            ('fem_ind', fem_ind),
            ('aero_ind', aero_ind),
        ]

        # Add material components to the top-level system
        root.add('indep_vars',
                 IndepVarComp(indep_vars),
                 promotes=['*'])
        root.add('twist_bsp',
                 Bspline('twist_cp', 'twist', jac_twist),
                 promotes=['*'])
        root.add('thickness_bsp',
                 Bspline('thickness_cp', 'thickness', jac_thickness),
                 promotes=['*'])
        root.add('mesh',
                 GeometryMesh(mesh, aero_ind),
                 promotes=['*'])
        root.add('tube',
                 MaterialsTube(fem_ind),
                 promotes=['*'])
        root.add('spatialbeamstates',
                 SpatialBeamStates(aero_ind, fem_ind, self.options_dict['E'], self.options_dict['G']),
                 promotes=['*'])
        root.add('spatialbeamfuncs',
                 SpatialBeamFunctionals(aero_ind, fem_ind, self.options_dict['E'], self.options_dict['G'], self.options_dict['stress'], self.options_dict['mrho']),
                 promotes=['*'])

        prob = self.setup_prob(root)

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('thickness_cp',
                               lower=numpy.ones((num_thickness)) * 0.0003,
                               upper=numpy.ones((num_thickness)) * 0.25,
                               scaler=1e5)

        # Set objective (minimize weight)
        prob.driver.add_objective('weight')

        # Set constraint (no structural failure)
        prob.driver.add_constraint('failure', upper=0.0)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('spatialbeam.db'))

        # Can finite difference over the entire model
        # Generally faster than using component derivatives
        # Note that for this case, you may need to loosen the optimizer tolerances
        # prob.root.deriv_options['type'] = 'fd'

        # Setup the problem and produce an N^2 diagram
        prob.setup()
        view_tree(prob, outfile="spatialbeam.html", show_browser=False)

        prob.run_once()

        if not self.options_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        return prob


if __name__ == "__main__":
    print
    print '===================================================='
    print '|                                                  |'
    print '|             Testing run functions                |'
    print '|                                                  |'
    print '===================================================='
    print

    aero = Aero()
    prob = aero.run()  # Match the CL
    print prob['CL'][0], .65655138

    options_dict = {'num_y' : 13,
              'num_x' : 2,
              'wing_type' : 'CRM'}
    AS = AeroStruct(options_dict)
    prob = AS.run()
    print prob['CL'], .58245256
    print prob['failure'], -.431801158

    struct = Struct()
    prob = struct.run()
    print prob['weight'], 4160.56823078
