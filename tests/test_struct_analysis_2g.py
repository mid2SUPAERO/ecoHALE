from __future__ import division, print_function
import sys
from time import time
import unittest
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.structures.struct_groups import SpatialBeamAlone

from openaerostruct.aerodynamics.aero_groups import AeroPoint

from openaerostruct.integration.aerostruct_groups import Aerostruct, AerostructPoint
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry

from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, DirectSolver, DenseJacobian, LinearRunOnce, PetscKSP, ScipyOptimizer# TODO, SqliteRecorder, CaseReader, profile
from openmdao.api import view_model
from six import iteritems


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

        surf_dict.update({'mesh' : mesh})

        surf_dict['num_x'], surf_dict['num_y'] = surf_dict['mesh'].shape[:2]

        surfaces = [surf_dict]

        # Create the problem and assign the model group
        prob = Problem()

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            ny = surface['num_y']

            indep_var_comp = IndepVarComp()
            indep_var_comp.add_output('loads', val=np.ones((ny, 6)) * 2e5, units='N')
            indep_var_comp.add_output('load_factor', val=2.)

            struct_group = SpatialBeamAlone(surface=surface)

            # Add indep_vars to the structural group
            struct_group.add_subsystem('indep_vars',
                 indep_var_comp,
                 promotes=['*'])

            prob.model.add_subsystem(surface['name'], struct_group)

        # Set up the problem
        prob.setup()

        prob.run_model()

        self.assertAlmostEqual(prob['wing.structural_weight'][0], 2437385.6547349701, places=2)


if __name__ == '__main__':
    unittest.main()
