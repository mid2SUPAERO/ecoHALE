from __future__ import division
import sys
from time import time
import numpy

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from openmdao.devtools.partition_tree_n2 import view_tree
from geometry import GeometryMesh, Bspline, gen_crm_mesh, gen_mesh, get_inds
from transfer import TransferDisplacements, TransferLoads
from vlm import VLMStates, VLMFunctionals, VLMGeometry
from spatialbeam import SpatialBeamStates, SpatialBeamFunctionals, radii
from materials import MaterialsTube
from functionals import FunctionalBreguetRange, FunctionalEquilibrium

from gs_newton import HybridGSNewton
from b_spline import get_bspline_mtx


def get_default_surf_dict():
    defaults = {'num_x' : 3,
                'num_y' : 5,
                'span' : 10.,
                'chord' : 1.,
                'cosine_spacing' : 1,
                'dihedral' : 0.,
                'sweep' : 0.,
                'taper' : 1.,
                'CL0' : 0.2,
                'CD0' : 0.015,
                'E' : 70.e9, # [Pa]
                'G' : 30.e9, # [Pa]
                'stress' : 20.e6, # [Pa]
                'mrho' : 3.e3, # [kg/m^3]
                'fem_origin' : 0.35,
                'symmetry' : False,
                'W0' : 0.5 * 2.5e6, # [N] (MTOW of B777 is 3e5 kg with fuel)
                'wing_type' : 'rectangular', # set this to choose the initial shape of the wing
                                             # either 'CRM' or 'rectangular'
                'name' : '',
                'offset' : numpy.array([0., 0., 0.])
                }
    return defaults

def get_default_prob_dict():
    defaults = {'Re' : 0.,
                'alpha' : 5.,
                'optimize' : True,
                'CT' : 9.81 * 17.e-6, # [1/s] (9.81 N/kg * 17e-6 kg/N/s)
                'R' : 14.3e6, # [m] maximum range
                'M' : 0.84, # at cruise
                'rho' : 0.38, # [kg/m^3] at 35,000 ft
                'a' : 295.4, # [m/s] at 35,000 ft
                }
    return defaults

class OASProblem():

    def __init__(self, input_dict={}):
        self.prob_dict = get_default_prob_dict()
        self.prob_dict.update(input_dict)
        self.prob_dict['v'] = self.prob_dict['M'] * self.prob_dict['a']
        self.surfaces = []

    def add_surface(self, input_dict={}):
        surf_dict = get_default_surf_dict()
        surf_dict.update(input_dict)

        if 'mesh' in surf_dict.keys():
            mesh = surf_dict['mesh']
            num_x, num_y = mesh.shape

        elif 'num_x' in surf_dict.keys():
            num_x = surf_dict['num_x']
            num_y = surf_dict['num_y']
            span = surf_dict['span']
            chord = surf_dict['chord']
            cosine_spacing = surf_dict['cosine_spacing']

            if surf_dict['wing_type'] == 'rectangular':
                mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)

            elif surf_dict['wing_type'] == 'CRM':
                npi = int(((num_y - 1) / 2) * .6)
                npo = int(npi * 5 / 3)
                mesh = gen_crm_mesh(n_points_inboard=npi, n_points_outboard=npo, num_x=num_x)
                num_x, num_y = mesh.shape[:2]
                surf_dict['num_x'] = num_x
                surf_dict['num_y'] = num_y


            else:
                print 'Error: wing_type option not understood. Must be either "CRM" or "rectangular".'

            if surf_dict['symmetry']:
                num_y = int((num_y+1)/2)
                mesh = mesh[:, :num_y, :]

        else:
            print "Error: Please either provide a mesh or a valid set of parameters."

        mesh = mesh + surf_dict['offset']

        r = radii(mesh)
        num_twist = numpy.max([int((num_y - 1) / 5), 5])

        surf_dict['mesh'] = mesh
        surf_dict['num_twist'] = num_twist
        surf_dict['r'] = r
        surf_dict['t'] = r / 10

        loads = numpy.zeros((r.shape[0]+1, 6), dtype='complex')
        loads[0, 2] = 1e3
        loads[-1, 2] = 1e3
        surf_dict['loads'] = loads


        name = surf_dict['name']
        for surface in self.surfaces:
            if name in surface['name']:
                print "Warning: Two surfaces have the same name. Appending '_' to the repeat names."
                name = name + '_'

        surf_dict['name'] = name + '_'
        self.surfaces.append(surf_dict)

    def setup_prob(self, prob):

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

    def run_as(self):

        root = Group()
        coupled = Group()

        # Set the optimization problem settings
        prob = Problem()
        prob.root = root

        for surface in self.surfaces:

            name = surface['name']
            tmp_group = Group()

            indep_vars = [
                (name+'twist_cp', numpy.zeros(surface['num_twist'])),
                (name+'thickness_cp', numpy.ones(surface['num_twist'])*numpy.max(surface['t'])),
                (name+'r', surface['r']),
                (name+'dihedral', surface['dihedral']),
                (name+'sweep', surface['sweep']),
                (name+'span', surface['span']),
                (name+'taper', surface['taper'])]


            # right now use the same number; TODO fix this
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])
            jac_thickness = get_bspline_mtx(surface['num_twist'], surface['num_y']-1)

            # Add material components to the top-level system
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

            name_orig = name.strip('_')
            name = name_orig + '_pre_solve'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

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


            name = name_orig + '_coupled'
            exec(name + ' = tmp_group')
            exec('coupled.add("' + name + '", ' + name + ', promotes=["*"])')

            name = name_orig
            exec('coupled.add("' + name + '_loads' + '", ' + 'TransferLoads(surface)' + ', promotes=["*"])')

            tmp_group = Group()
            tmp_group.add('spatialbeamfuncs',
                     SpatialBeamFunctionals(surface),
                     promotes=['*'])
            tmp_group.add('vlmfuncs',
                     VLMFunctionals(surface),
                     promotes=['*'])

            name = name_orig + '_post_solve'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')


        coupled.add('vlmstates',
                 VLMStates(self.surfaces, self.prob_dict),
                 promotes=['*'])

        # Set solver properties
        coupled.nl_solver = Newton()
        coupled.nl_solver.options['iprint'] = 1
        coupled.ln_solver = ScipyGMRES()
        coupled.ln_solver.options['iprint'] = 1
        coupled.ln_solver.preconditioner = LinearGaussSeidel()
        coupled.vlmstates.ln_solver = LinearGaussSeidel()

        # coupled.nl_solver = NLGaussSeidel()   ### Uncomment this out to use NLGS
        # coupled.nl_solver.options['iprint'] = 1
        coupled.nl_solver.options['atol'] = 1e-5
        coupled.nl_solver.options['rtol'] = 1e-12

        coupled.nl_solver = HybridGSNewton()   ### Uncomment this out to use Hybrid GS Newton
        coupled.nl_solver.nlgs.options['iprint'] = 1
        coupled.nl_solver.nlgs.options['maxiter'] = 10
        coupled.nl_solver.nlgs.options['atol'] = 1e-8
        coupled.nl_solver.nlgs.options['rtol'] = 1e-12
        coupled.nl_solver.newton.options['atol'] = 1e-7
        coupled.nl_solver.newton.options['rtol'] = 1e-7
        coupled.nl_solver.newton.options['maxiter'] = 5
        coupled.nl_solver.newton.options['iprint'] = 1

        order_list = []
        for surface in self.surfaces:
            order_list.append(surface['name']+'coupled')
        order_list.append('vlmstates')
        for surface in self.surfaces:
            order_list.append(surface['name']+'loads')
        coupled.set_order(order_list)

        root.add('coupled', coupled, promotes=['*'])

        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('Re', self.prob_dict['Re']),
            ('rho', self.prob_dict['rho'])]

        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])

        root.add('fuelburn',
                 FunctionalBreguetRange(self.surfaces, self.prob_dict),
                 promotes=['*'])
        root.add('eq_con',
                 FunctionalEquilibrium(self.surfaces, self.prob_dict),
                 promotes=['*'])

        prob = self.setup_prob(prob)


        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('alpha', lower=-10., upper=10., scaler=1)

        prob.driver.add_desvar('wing_twist_cp',lower= -15.,
                               upper=15., scaler=1e0)
        prob.driver.add_desvar('wing_thickness_cp',
                               lower= 0.01,
                               upper= 0.25, scaler=1e3)
        prob.driver.add_constraint('wing_failure', upper=0.0)

        prob.driver.add_desvar('tail_twist_cp',lower= -15.,
                               upper=15., scaler=1e0)
        prob.driver.add_desvar('tail_thickness_cp',
                               lower= 0.01,
                               upper= 0.25, scaler=1e3)
        prob.driver.add_constraint('tail_failure', upper=0.0)

        # Set the objective (minimize fuelburn)
        prob.driver.add_objective('fuelburn', scaler=1e-3)

        # Set the constraints (no structural failure and lift = weight)
        prob.driver.add_constraint('eq_con', equals=0.0)

        # TODO: Need to figure out derivatives
        # prob.root.deriv_options['type'] = 'fd'

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('aerostruct.db'))

        # Set up the problem
        prob.setup()

        # prob.print_all_convergence()
        view_tree(prob, outfile="aerostruct.html", show_browser=False)

        prob.run_once()

        if not self.prob_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        self.prob = prob

    def run_struct(self):

        root = Group()

        # Set the optimization problem settings
        prob = Problem()
        prob.root = root

        for surface in self.surfaces:

            name = surface['name']
            tmp_group = Group()

            surface['r'] = surface['r'] / 5
            surface['t'] = surface['r'] / 20

            indep_vars = [
                (name+'twist_cp', numpy.zeros(surface['num_twist'])),
                (name+'thickness_cp', numpy.ones(surface['num_twist'])*numpy.max(surface['t'])),
                (name+'dihedral', surface['dihedral']),
                (name+'sweep', surface['sweep']),
                (name+'span', surface['span']),
                (name+'taper', surface['taper']),
                (name+'r', surface['r']),
                (name+'loads', surface['loads'])]

            # right now use the same number; TODO fix this
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])
            jac_thickness = get_bspline_mtx(surface['num_twist'], surface['num_y']-1)

            # Add material components to the top-level system
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

            name = name.strip('_')
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

        prob = self.setup_prob(prob)


        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('wing_thickness_cp',
                               lower=numpy.ones((self.surfaces[0]['num_twist'])) * 0.0003,
                               upper=numpy.ones((self.surfaces[0]['num_twist'])) * 0.25,
                               scaler=1e5)

        # Set objective (minimize weight)
        prob.driver.add_objective('wing_weight')

        # Set constraint (no structural failure)
        prob.driver.add_constraint('wing_failure', upper=0.0)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('spatialbeam.db'))

        # Can finite difference over the entire model
        # Generally faster than using component derivatives
        # Note that for this case, you may need to loosen the optimizer tolerances
        # prob.root.deriv_options['type'] = 'fd'

        # Setup the problem and produce an N^2 diagram
        prob.setup()

        # prob.print_all_convergence()
        view_tree(prob, outfile="spatialbeam.html", show_browser=False)

        prob.run_once()

        if not self.prob_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        self.prob = prob

    def run_aero(self):

        root = Group()

        # Set the optimization problem settings
        prob = Problem()
        prob.root = root

        for surface in self.surfaces:

            name = surface['name']
            tmp_group = Group()

            indep_vars = [
                (name+'twist_cp', numpy.zeros(surface['num_twist'])),
                (name+'dihedral', surface['dihedral']),
                (name+'sweep', surface['sweep']),
                (name+'span', surface['span']),
                (name+'taper', surface['taper']),
                (name+'disp', numpy.zeros((surface['num_y'], 6)))]


            # right now use the same number; TODO fix this
            jac_twist = get_bspline_mtx(surface['num_twist'], surface['num_y'])

            # Add material components to the top-level system
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

            name_orig = name.strip('_')
            name = name_orig + '_pre_solve'
            exec(name + ' = tmp_group')
            exec('root.add("' + name + '", ' + name + ', promotes=["*"])')

            name = name_orig + '_post_solve'
            exec('root.add("' + name + '", ' + 'VLMFunctionals(surface)' + ', promotes=["*"])')

        prob_vars = [('v', self.prob_dict['v']),
            ('alpha', self.prob_dict['alpha']),
            ('M', self.prob_dict['M']),
            ('Re', self.prob_dict['Re']),
            ('rho', self.prob_dict['rho'])]

        root.add('prob_vars',
                 IndepVarComp(prob_vars),
                 promotes=['*'])
        root.add('vlmstates',
                 VLMStates(self.surfaces, self.prob_dict),
                 promotes=['*'])


        prob = self.setup_prob(prob)

        # Add design variables for the optimizer to control
        # Note that the scaling is very important to get correct convergence
        prob.driver.add_desvar('wing_twist_cp', lower=-10., upper=15., scaler=1e0)
        # prob.driver.add_desvar('alpha', lower=-10., upper=10.)
        # prob.driver.add_desvar('wing_sweep', lower=-10., upper=30.)
        # prob.driver.add_desvar('wing_dihedral', lower=-10., upper=20.)
        # prob.driver.add_desvar('wing_taper', lower=.5, upper=2.)

        # Set the objective (minimize CD on the main wing)
        prob.driver.add_objective('wing_CD', scaler=1e4)

        # Set the constraint (CL = 0.5 for the main wing)
        prob.driver.add_constraint('wing_CL', equals=0.5)

        # Record optimization history to a database
        # Data saved here can be examined using `plot_all.py`
        prob.driver.add_recorder(SqliteRecorder('vlm.db'))

        # Can finite difference over the entire model
        # Generally faster than using component derivatives
        prob.root.deriv_options['type'] = 'fd'

        # Setup the problem
        prob.setup()

        # prob.print_all_convergence()
        view_tree(prob, outfile="vlm.html", show_browser=False)

        prob.run_once()
        if not self.prob_dict['optimize']:  # run analysis once
            pass
        else:  # perform optimization
            prob.run()
        self.prob = prob


if __name__ == '__main__':
    OAS_prob = OASProblem({'optimize' : True})
    OAS_prob.add_surface({'name' : 'wing',
                          'wing_type' : 'CRM',
                          'num_x' : 2,
                          'num_y' : 9,
                          })
    OAS_prob.add_surface({'name' : 'tail',
                          'wing_type' : 'CRM',
                          'num_x' : 2,
                          'num_y' : 9,
                          'offset' : numpy.array([0., 0., 1000000.])})

    # OAS_prob.run_aero()
    OAS_prob.run_as()
    prob = OAS_prob.prob

    # print prob['wing_weight']
    # OAS_prob.prob.check_partial_derivatives(compact_print=True)
    print
    print
    print prob['wing_CL'], prob['wing_CD']
    # print prob['tail_CL'], prob['tail_CD']

    # print
    # print prob['wing_mesh']


    # TODO: change all kw checks to lower()
