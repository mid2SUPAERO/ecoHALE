from __future__ import division, print_function
from openmdao.utils.assert_utils import assert_rel_error
import unittest
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.struct_groups import SpatialBeamAlone

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, LinearBlockGS, PetscKSP, ScipyOptimizeDriver


class Test(unittest.TestCase):

    def test(self):

        # Create a dictionary to store options about the surface
        mesh_dict = {'num_y' : 7,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surface = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'fem_model_type' : 'tube',

                    'mesh' : mesh,

                    # Structural values are based on aluminum 7075
                    'E' : 70.e9,            # [Pa] Young's modulus of the spar
                    'G' : 30.e9,            # [Pa] shear modulus of the spar
                    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
                    'mrho' : 3.e3,          # [kg/m^3] material density
                    'fem_origin' : 0.35,    # normalized chordwise location of the spar
                    't_over_c_cp' : np.array([0.15]),      # maximum airfoil thickness
                    'thickness_cp' : np.ones((3)) * .1,
                    'wing_weight_ratio' : 2.,
                    'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
                    'distributed_fuel_weight' : False,
                    'exact_failure_constraint' : False,
                    }

        # Create the problem and assign the model group
        prob = Problem()

        ny = surface['mesh'].shape[1]
        surface['n_point_masses'] = 1

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('loads', val=np.zeros((ny, 6)), units='N')
        indep_var_comp.add_output('load_factor', val=1.)

        point_masses = np.array([[8000.]])

        point_mass_locations = np.array([[25, -10., 0.]])

        indep_var_comp.add_output('point_masses', val=point_masses, units='kg')
        indep_var_comp.add_output('point_mass_locations', val=point_mass_locations, units='m')

        struct_group = SpatialBeamAlone(surface=surface)

        # Add indep_vars to the structural group
        struct_group.add_subsystem('indep_vars',
             indep_var_comp,
             promotes=['*'])

        prob.model.add_subsystem(surface['name'], struct_group, promotes=['*'])

        # Set up the problem
        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['structural_mass'][0], 124229.646011, 1e-4)
        assert_rel_error(self, prob['disp'][1, 2:5], np.array([-2.91474315e-03, 4.78977910e-04, -5.48429013e-04]), 1e-4)


if __name__ == '__main__':
    unittest.main()
