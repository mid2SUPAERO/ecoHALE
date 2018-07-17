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

        surf_dict = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'type' : 'structural',
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'fem_model_type' : 'tube',

                    'mesh' : mesh,
                    'num_x' : mesh.shape[0],
                    'num_y' : mesh.shape[1],

                    # Structural values are based on aluminum 7075
                    'E' : 70.e9,            # [Pa] Young's modulus of the spar
                    'G' : 30.e9,            # [Pa] shear modulus of the spar
                    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
                    'mrho' : 3.e3,          # [kg/m^3] material density
                    'fem_origin' : 0.35,    # normalized chordwise location of the spar
                    't_over_c' : 0.15,      # maximum airfoil thickness
                    'thickness_cp' : np.ones((3)) * .1,
                    'wing_weight_ratio' : 2.,

                    'exact_failure_constraint' : False,
                    }

        # Create the problem and assign the model group
        prob = Problem()

        ny = surf_dict['num_y']

        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('loads', val=np.ones((ny, 6)) * 2e5, units='N')
        indep_var_comp.add_output('load_factor', val=2.)

        struct_group = SpatialBeamAlone(surface=surf_dict)

        # Add indep_vars to the structural group
        struct_group.add_subsystem('indep_vars',
             indep_var_comp,
             promotes=['*'])

        prob.model.add_subsystem(surf_dict['name'], struct_group)

        # Set up the problem
        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['wing.structural_weight'][0], 2437385.6547349701, 1e-4)


if __name__ == '__main__':
    unittest.main()
