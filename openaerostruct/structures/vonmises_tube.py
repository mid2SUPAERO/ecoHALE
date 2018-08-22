from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

from openaerostruct.structures.utils import norm, unit, norm_d, unit_d, cross_d

class VonMisesTube(ExplicitComponent):
    """ Compute the von Mises stress in each element.

    parameters
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

        self.add_input('nodes', val=np.zeros((self.ny, 3)), units='m')
        self.add_input('radius', val=np.zeros((self.ny - 1)), units='m')
        self.add_input('disp', val=np.zeros((self.ny, 6)), units='m')

        self.add_output('vonmises', val=np.zeros((self.ny-1, 2)), units='N/m**2')

        self.E = surface['E']
        self.G = surface['G']
        # data_type = float
        # if self._outputs._alloc_complex:
        #     data_type = complex
        # self.T = np.zeros((3, 3),dtype=data_type)
        # self.x_gl = np.array([1, 0, 0],dtype=data_type)

        # TODO fix the sparsity declarations
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        data_type = float
        if self._outputs._vector_info._under_complex_step:
            data_type = complex
        self.T = np.zeros((3, 3),dtype=data_type)
        self.x_gl = np.array([1, 0, 0],dtype=data_type)
        radius = inputs['radius']
        disp = inputs['disp']
        nodes = inputs['nodes']
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

            tmp = np.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
            sxx0 = E * (u1x - u0x) / L + E * radius[ielem] / L * tmp
            sxx1 = E * (u0x - u1x) / L + E * radius[ielem] / L * tmp
            sxt = G * radius[ielem] * (r1x - r0x) / L

            outputs['vonmises'][ielem, 0] = np.sqrt(sxx0**2 + 3 * sxt**2)
            outputs['vonmises'][ielem, 1] = np.sqrt(sxx1**2 + 3 * sxt**2)

    def compute_partials(self, inputs, partials):

        radius = inputs['radius']
        disp = inputs['disp']
        nodes = inputs['nodes']
        T = self.T
        E = self.E
        G = self.G
        x_gl = self.x_gl

        num_elems = self.ny - 1
        for ielem in range(num_elems):

            # Compute the coordinate delta between the two element end points
            P0 = nodes[ielem, :]
            P1 = nodes[ielem+1, :]
            dP = P1 - P0

            # and its derivatives
            ddPdP0 = -1.0*np.eye(3)
            ddPdP1 = 1.0*np.eye(3)

            # Compute the element length and its derivative
            L = norm(dP)
            dLddP = norm_d(dP)

            # unit function converts a vector to a unit vector
            # calculate the transormation to the local element frame.
            # We use x_gl to provide a reference axis to reference
            x_loc = unit(dP)
            dxdP = unit_d(dP)

            y_loc = unit(np.cross(x_loc, x_gl))
            dtmpdx,dummy = cross_d(x_loc,x_gl)
            dydtmp = unit_d(np.cross(x_loc, x_gl))
            dydP = dydtmp.dot(dtmpdx).dot(dxdP)

            z_loc = unit(np.cross(x_loc, y_loc))
            dtmpdx,dtmpdy = cross_d(x_loc,y_loc)
            dzdtmp = unit_d(np.cross(x_loc, y_loc))
            dzdP = dzdtmp.dot(dtmpdx).dot(dxdP)+dzdtmp.dot(dtmpdy).dot(dydP)

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            #$$$$$$$$$$$$$$$$$$$$$$$$$$
            # Original code
            # $$$$$$$$$$$$
            u0x, u0y, u0z = T.dot(disp[ielem, :3])
            r0x, r0y, r0z = T.dot(disp[ielem, 3:])
            u1x, u1y, u1z = T.dot(disp[ielem+1, :3])
            r1x, r1y, r1z = T.dot(disp[ielem+1, 3:])

            # #$$$$$$$$$$$$$$$$$$$$$$$$$$

            # The derivatives of the above code wrt displacement all boil down to sections of the T matrix
            dxddisp = T[0,:]
            dyddisp = T[1,:]
            dzddisp = T[2,:]

            #The derivatives of the above code wrt T all boil down to sections of the #displacement vector
            # du0dT = disp[ielem, :3]
            # dr0dT = disp[ielem, 3:]
            # du1dT = disp[ielem+1, :3]
            # dr1dT = disp[ielem+1, 3:]

            du0dloc = disp[ielem, :3]
            dr0dloc = disp[ielem, 3:]
            du1dloc = disp[ielem+1, :3]
            dr1dloc = disp[ielem+1, 3:]

            #$$$$$$$$$$$$$$$$$$$$$$$$$$
            # Original code
            # $$$$$$$$$$$$
            tmp = np.sqrt((r1y - r0y)**2 + (r1z - r0z)**2)
            sxx0 = E * (u1x - u0x) / L + E * radius[ielem] / L * tmp
            sxx1 = E * (u0x - u1x) / L + E * radius[ielem] / L * tmp
            sxt = G * radius[ielem] * (r1x - r0x) / L
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



            dtmpdr0y = 1/tmp * (r1y - r0y)*-1
            dtmpdr1y = 1/tmp * (r1y - r0y)
            dtmpdr0z = 1/tmp * (r1z - r0z)*-1
            dtmpdr1z = 1/tmp * (r1z - r0z)

            # Combine all of the derivtives for tmp
            dtmpdDisp = np.zeros(2 * 6)
            dr0xdDisp = np.zeros(2*6)
            dr0ydDisp = np.zeros(2*6)
            dr0zdDisp = np.zeros(2*6)
            dr1xdDisp = np.zeros(2*6)
            dr1ydDisp = np.zeros(2*6)
            dr1zdDisp = np.zeros(2*6)

            idx1 = 3
            idx2 = 6 + 3
            #r0 term
            dr0xdDisp[idx1:idx1+3] = dxddisp
            dr0ydDisp[idx1:idx1+3] = dyddisp
            dr0zdDisp[idx1:idx1+3] = dzddisp
            dr1xdDisp[idx2:idx2+3] = dxddisp
            dr1ydDisp[idx2:idx2+3] = dyddisp
            dr1zdDisp[idx2:idx2+3] = dzddisp
            dtmpdDisp = (dtmpdr0y*dr0ydDisp+dtmpdr1y*dr1ydDisp+\
                        dtmpdr0z*dr0zdDisp+dtmpdr1z*dr1zdDisp)

            # x_loc, y_loc and z_loc terms
            dtmpdx_loc = np.array([0,0,0])
            dtmpdy_loc = dtmpdr0y*dr0dloc + dtmpdr1y*dr1dloc
            dtmpdz_loc = dtmpdr0z*dr0dloc + dtmpdr1z*dr1dloc

            dtmpdP = dtmpdx_loc.dot(dxdP) + dtmpdy_loc.dot(dydP) + dtmpdz_loc.dot(dzdP)

            dsxx0dtmp = E * radius[ielem] / L
            dsxx0du0x = -E / L
            dsxx0du1x = E / L
            dsxx0dL =  -E * (u1x - u0x) / (L*L) - E * radius[ielem] / (L*L) * tmp

            dsxx1dtmp = E * radius[ielem] / L
            dsxx1du0x = E / L
            dsxx1du1x = -E / L
            dsxx1dL = -E * (u0x - u1x) / (L*L) - E * radius[ielem] / (L*L) * tmp

            dsxx0dP = dsxx0dtmp * dtmpdP + \
                      dsxx0du0x*du0dloc.dot(dxdP) + dsxx0du1x*du1dloc.dot(dxdP)+\
                      dsxx0dL*dLddP

            dsxx1dP = dsxx1dtmp * dtmpdP + \
                      dsxx1du0x*du0dloc.dot(dxdP)+dsxx1du1x*du1dloc.dot(dxdP)+\
                      dsxx1dL*dLddP

            # Combine sxx0 and sxx1 terms
            idx1 = 0
            idx2 = 6
            # Start with the tmp term
            dsxx0dDisp = dsxx0dtmp * dtmpdDisp
            dsxx1dDisp = dsxx1dtmp * dtmpdDisp

            # Now add the direct u dep
            dsxx0dDisp[idx1:idx1+3] = dsxx0du0x * dxddisp
            dsxx0dDisp[idx2:idx2+3] = dsxx0du1x * dxddisp

            dsxx1dDisp[idx1:idx1+3] = dsxx1du0x * dxddisp
            dsxx1dDisp[idx2:idx2+3] = dsxx1du1x * dxddisp

            # Combine sxt term
            dsxtdDisp = np.zeros(2 * 6)
            idx1 = 3
            idx2 = 6 + 3

            dsxtdr0x = -G * radius[ielem] / L
            dsxtdr1x = G * radius[ielem] / L
            dsxtdL =  - G * radius[ielem] * (r1x - r0x) / (L*L)

            dsxtdP = dsxtdr0x*(dr0dloc.dot(dxdP))+ dsxtdr1x*(dr1dloc.dot(dxdP))+\
                     dsxtdL*dLddP
            #disp
            dsxtdDisp= dsxtdr0x * dr0xdDisp + dsxtdr1x * dr1xdDisp

            #radius derivatives
            dsxx0drad = E / L * tmp
            dsxx1drad = E / L * tmp
            dsxtdrad = G * (r1x - r0x)/L

            dVm0dsxx0 = (sxx0)/(np.sqrt(sxx0**2 + 3 * sxt**2))
            dVm0dsxt = (3*sxt)/(np.sqrt(sxx0**2 + 3 * sxt**2))
            dVm1dsxx1 = (sxx1)/(np.sqrt(sxx1**2 + 3 * sxt**2))
            dVm1dsxt = (3*sxt)/(np.sqrt(sxx1**2 + 3 * sxt**2))

            idx = ielem*2
            partials['vonmises','radius'][idx,ielem] = dVm0dsxx0*dsxx0drad+dVm0dsxt*dsxtdrad
            partials['vonmises','radius'][idx+1,ielem] = dVm1dsxx1*dsxx1drad+dVm1dsxt*dsxtdrad

            idx2 = ielem*6
            partials['vonmises','disp'][idx,idx2:idx2+12] = (dVm0dsxx0*dsxx0dDisp+dVm0dsxt*dsxtdDisp )
            partials['vonmises','disp'][idx+1,idx2:idx2+12] = (dVm1dsxx1*dsxx1dDisp+dVm1dsxt*dsxtdDisp )

            # Compute terms for the nodes
            idx3 = ielem*3

            partials['vonmises','nodes'][idx,idx3:idx3+3] = dVm0dsxx0*dsxx0dP.dot(ddPdP0)+dVm0dsxt*dsxtdP.dot(ddPdP0)
            partials['vonmises','nodes'][idx,idx3+3:idx3+6] = dVm0dsxx0*dsxx0dP.dot(ddPdP1)+dVm0dsxt*dsxtdP.dot(ddPdP1)

            partials['vonmises','nodes'][idx+1,idx3:idx3+3] = dVm1dsxx1*dsxx1dP.dot(ddPdP0)+dVm1dsxt*dsxtdP.dot(ddPdP0)
            partials['vonmises','nodes'][idx+1,idx3+3:idx3+6] = dVm1dsxx1*dsxx1dP.dot(ddPdP1)+dVm1dsxt*dsxtdP.dot(ddPdP1)
