from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.structures.utils import norm, unit

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex

class VonMisesWingbox(ExplicitComponent):
    """ Compute the von Mises stress in each element.

    Parameters
    ----------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    radius[ny-1] : numpy array
        Radii for each FEM element.
    disp[ny, 6] : numpy array
        Displacements of each FEM node.

    Returns
    -------
    vonmises[ny-1, 2] : numpy array
        von Mises stress magnitudes for each FEM element.

    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        self.surface = surface = self.options['surface']

        self.ny = surface['num_y']

        self.add_input('nodes', val=np.zeros((self.ny, 3),
                       dtype=complex))

        self.add_input('disp', val=np.zeros((self.ny, 6),
                       dtype=complex))

        self.add_input('Qz', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('Iz', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('J', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('A_enc', val=np.zeros((self.ny - 1), dtype=complex))

        self.add_input('spar_thickness', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('skin_thickness', val=np.zeros((self.ny - 1), dtype=complex))

        self.add_input('htop', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('hbottom', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('hfront', val=np.zeros((self.ny - 1), dtype=complex))
        self.add_input('hrear', val=np.zeros((self.ny - 1), dtype=complex))

        self.add_output('vonmises', val=np.zeros((self.ny-1, 4),
                        dtype=complex))

        self.E = surface['E']
        self.G = surface['G']

        self.T = np.zeros((3, 3), dtype=complex)
        self.x_gl = np.array([1, 0, 0], dtype=complex)

        self.tssf = top_skin_strength_factor = surface['strength_factor_for_upper_skin']

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        disp = inputs['disp']
        nodes = inputs['nodes']
        A_enc = inputs['A_enc']
        Qy = inputs['Qz']
        Iz = inputs['Iz']
        J = inputs['J']
        htop = inputs['htop']
        hbottom = inputs['hbottom']
        hfront = inputs['hfront']
        hrear = inputs['hrear']
        spar_thickness = inputs['spar_thickness']
        skin_thickness = inputs['skin_thickness']
        vonmises = outputs['vonmises']

        T = self.T
        E = self.E
        G = self.G
        x_gl = self.x_gl

        num_elems = self.ny - 1
        for ielem in range(self.ny-1):

            P0 = nodes[ielem, :]
            P1 = nodes[ielem+1, :]
            L = norm(P1 - P0)

            x_loc = unit(P1 - P0)
            y_loc = unit(np.cross(x_loc, x_gl))
            z_loc = unit(np.cross(x_loc, y_loc))

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            u0x, u0y, u0z = T.dot(disp[ielem, :3])
            r0x, r0y, r0z = T.dot(disp[ielem, 3:])
            u1x, u1y, u1z = T.dot(disp[ielem+1, :3])
            r1x, r1y, r1z = T.dot(disp[ielem+1, 3:])


            axial_stress = E * (u1x - u0x) / L      # this is stress = modulus * strain; positive is tensile
            torsion_stress = G * J[ielem] / L * (r1x - r0x) / 2 / spar_thickness[ielem] / A_enc[ielem]   # this is Torque / (2 * thickness_min * Area_enclosed)
            top_bending_stress = E / (L**2) * (6 * u0y + 2 * r0z * L - 6 * u1y + 4 * r1z * L ) * htop[ielem] # this is moment * htop / I
            bottom_bending_stress = - E / (L**2) * (6 * u0y + 2 * r0z * L - 6 * u1y + 4 * r1z * L ) * hbottom[ielem] # this is moment * htop / I
            front_bending_stress = - E / (L**2) * (-6 * u0z + 2 * r0y * L + 6 * u1z + 4 * r1y * L ) * hfront[ielem] # this is moment * htop / I
            rear_bending_stress = E / (L**2) * (-6 * u0z + 2 * r0y * L + 6 * u1z + 4 * r1y * L ) * hrear[ielem] # this is moment * htop / I

            vertical_shear =  E / (L**3) *(-12 * u0y - 6 * r0z * L + 12 * u1y - 6 * r1z * L ) * Qy[ielem] / (2 * spar_thickness[ielem]) # shear due to bending (VQ/It) note: the I used to get V cancels the other I

            # print("==========",ielem,"================")
            # print("vertical_shear", vertical_shear)
            # print("top",top_bending_stress)
            # print("bottom",bottom_bending_stress)
            # print("front",front_bending_stress)
            # print("rear",rear_bending_stress)
            # print("axial", axial_stress)
            # print("torsion", torsion_stress)

            vonmises[ielem, 0] = np.sqrt((top_bending_stress + rear_bending_stress + axial_stress)**2 + 3*torsion_stress**2) / self.tssf
            vonmises[ielem, 1] = np.sqrt((bottom_bending_stress + front_bending_stress + axial_stress)**2 + 3*torsion_stress**2)
            vonmises[ielem, 2] = np.sqrt((front_bending_stress + axial_stress)**2 + 3*(torsion_stress-vertical_shear)**2)
            vonmises[ielem, 3] = np.sqrt((rear_bending_stress + axial_stress)**2 + 3*(torsion_stress+vertical_shear)**2) / self.tssf
