from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

class ViscousDrag(ExplicitComponent):
    """
    Compute the skin friction drag if the with_viscous option is True.
    If not, the CDv is 0.
    This component exists for each lifting surface.

    Parameters
    ----------
    re : float
        Dimensionalized (1/length) Reynolds number. This is used to compute the
        local Reynolds number based on the local chord length.
    M : float
        Mach number.
    S_ref : float
        The reference area of the lifting surface.
    sweep : float
        The angle (in degrees) of the wing sweep. This is used in the form
        factor calculation.
    widths[ny-1] : numpy array
        The spanwise width of each panel.
    lengths[ny] : numpy array
        The sum of the lengths of each line segment along a chord section.

    Returns
    -------
    CDv : float
        Viscous drag coefficient for the lifting surface computed using flat
        plate skin friction coefficient and a form factor to account for wing
        shape.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('with_viscous', types=bool)

    def setup(self):
        self.surface = surface = self.options['surface']
        self.with_viscous = surface['with_viscous']

        # Percentage of chord with laminar flow
        self.k_lam = surface['k_lam']

        # Thickness over chord for the airfoil
        self.t_over_c = surface['t_over_c']
        self.c_max_t = surface['c_max_t']

        ny = surface['num_y']

        self.add_input('re', val=5.e6, units='1/m')
        self.add_input('M', val=1.6)
        self.add_input('S_ref', val=1., units='m**2')
        self.add_input('cos_sweep', val=np.ones((ny-1)), units='m')
        self.add_input('widths', val=np.ones((ny-1)), units='m')
        self.add_input('lengths', val=np.ones((ny)), units='m')
        self.add_output('CDv', val=0.)

        self.declare_partials('CDv', '*')
        # self.declare_partials('CDv', 'M', method='fd')
        # self.declare_partials('CDv', 're')

        self.set_check_partial_options(wrt='*', method='cs', step=1e-50)

    def compute(self, inputs, outputs):
        if self.with_viscous:
            re = inputs['re']
            M = inputs['M']
            S_ref = inputs['S_ref']
            widths = inputs['widths']
            lengths = inputs['lengths']
            cos_sweep = inputs['cos_sweep'] / widths

            # Take panel chord length to be average of its edge lengths
            chords = (lengths[1:] + lengths[:-1]) / 2.
            Re_c = re * chords

            cdturb_total = 0.455 / (np.log10(Re_c))**2.58 / \
                (1.0 + 0.144*M**2)**0.65
            cdlam_tr = 1.328 / np.sqrt(Re_c * self.k_lam)

            # Use eq. 12.27 of Raymer for turbulent Cf
            if self.k_lam == 0:
                cdlam_tr = 0.
                cdturb_tr = 0.

            elif self.k_lam < 1.0:
                cdturb_tr = 0.455 / (np.log10(Re_c*self.k_lam))**2.58 / \
                    (1.0 + 0.144*M**2)**0.65

            else:
                cdturb_total = 0.
                cdturb_tr = 0.

            cd = (cdlam_tr - cdturb_tr)*self.k_lam + cdturb_total

            # Multiply by section width to get total normalized drag for section
            d_over_q = 2 * cd * chords

            # Calculate form factor (Raymer Eq. 12.30)
            k_FF = 1.34 * M**0.18 * \
                (1.0 + 0.6*self.t_over_c/self.c_max_t + 100*self.t_over_c**4)
            FF = k_FF * cos_sweep**0.28

            # Sum individual panel drags to get total drag
            D_over_q = np.sum(d_over_q * widths * FF)

            outputs['CDv'] = D_over_q / S_ref

            if self.surface['symmetry']:
                outputs['CDv'] *= 2
        else:
            outputs['CDv'] = 0.0

    def compute_partials(self, inputs, partials):
        """ Jacobian for viscous drag."""

        partials['CDv', 'lengths'] = np.zeros_like(partials['CDv', 'lengths'])
        re = inputs['re']

        if self.with_viscous:
            p180 = np.pi / 180.
            M = inputs['M'][0]
            S_ref = inputs['S_ref'][0]

            widths = inputs['widths']
            lengths = inputs['lengths']
            cos_sweep = inputs['cos_sweep'] / widths

            # Take panel chord length to be average of its edge lengths
            chords = (lengths[1:] + lengths[:-1]) / 2.
            Re_c = re * chords

            cdturb_total = 0.455 / (np.log10(Re_c))**2.58 / \
                (1.0 + 0.144*M**2)**0.65
            cdlam_tr = 1.328 / np.sqrt(Re_c * self.k_lam)

            # Use eq. 12.27 of Raymer for turbulent Cf
            if self.k_lam == 0:
                cdlam_tr = 0.
                cdturb_tr = 0.

            elif self.k_lam < 1.0:
                cdturb_tr = 0.455 / (np.log10(Re_c*self.k_lam))**2.58 / \
                    (1.0 + 0.144*M**2)**0.65

            else:
                cdturb_total = 0.
                cdturb_tr = 0.

            cd = (cdlam_tr - cdturb_tr)*self.k_lam + cdturb_total

            # Multiply by section width to get total normalized drag for section
            d_over_q = 2 * cd * chords

            # Calculate form factor (Raymer Eq. 12.30)
            k_FF = 1.34 * M**0.18 * \
                (1.0 + 0.6*self.t_over_c/self.c_max_t + 100*self.t_over_c**4)
            FF = k_FF * cos_sweep**0.28

            # Sum individual panel drags to get total drag
            D_over_q = np.sum(d_over_q * widths * FF)

            chords = (lengths[1:] + lengths[:-1]) / 2.
            Re_c = re * chords

            cdl_Re = 0.0
            cdt_Re = 0.0
            cdT_Re = 0.0

            B = (1. + 0.144*M**2)**0.65

            if self.k_lam == 0:
                cdT_Re = 0.455/(np.log10(Re_c))**3.58/B * \
                            -2.58 / np.log(10) / Re_c
            elif self.k_lam < 1.0:

                cdl_Re = 1.328 / (Re_c*self.k_lam)**1.5 * -0.5 * self.k_lam
                cdt_Re = 0.455/(np.log10(Re_c*self.k_lam))**3.58/B * \
                            -2.58 / np.log(10) / Re_c
                cdT_Re = 0.455/(np.log10(Re_c))**3.58/B * \
                            -2.58 / np.log(10) / Re_c

            else:
                cdl_Re = 1.328 / (Re_c*self.k_lam)**1.5 * -0.5 * self.k_lam

            cd_Re = (cdl_Re - cdt_Re)*self.k_lam + cdT_Re

            CDv_lengths = 2 * widths * FF / S_ref * \
                (d_over_q / 4 / chords + chords * cd_Re * re / 2.)

            partials['CDv', 'lengths'][0, 1:] += CDv_lengths
            partials['CDv', 'lengths'][0, :-1] += CDv_lengths
            partials['CDv', 'widths'][0, :] = d_over_q * FF / S_ref * 0.72
            partials['CDv', 'S_ref'] = - D_over_q / S_ref**2
            partials['CDv', 'cos_sweep'][0, :] = 0.28 * k_FF * d_over_q / S_ref / cos_sweep**0.72


            term =  (-0.65/(1+0.144*M**2)**1.65) * 2*0.144*M
            dcdturb_total__dM = 0.455 / (np.log10(Re_c))**2.58 * term
            dcdturb_tr__dM = 0.455 / (np.log10(Re_c*self.k_lam))**2.58 * term

            if self.k_lam == 0:
                dCd__dM = dcdturb_total__dM
            elif self.k_lam < 1:
                dCd__dM = -self.k_lam*dcdturb_tr__dM + dcdturb_total__dM
            else:
                dCd__dM = 0.
            dd_over_q__dM = 2*chords*dCd__dM

            dk_ff__dM = 1.34*0.18*M**-0.82 * (1.0 + 0.6*self.t_over_c/self.c_max_t + 100*self.t_over_c**4)
            dFF__dM = dk_ff__dM*cos_sweep**0.28

            dD_over_q__dM = np.sum(widths* (dd_over_q__dM*FF + dFF__dM*d_over_q))

            partials['CDv','M'] = dD_over_q__dM / S_ref

            term = 0.455/(1+0.144*M**2)**0.65
            dRe_c__dRe = chords
            dcdturb_tr__dRe = term * -2.58/np.log10(Re_c*self.k_lam)**3.58 / (Re_c*self.k_lam*np.log(10)) * self.k_lam * dRe_c__dRe
            dcdlam_tr__dRe = -1.328/2./(Re_c*self.k_lam)**1.5 * self.k_lam * dRe_c__dRe
            dcdturb_total__dRe = term * -2.58/np.log10(Re_c)**3.58 / (Re_c * np.log(10)) * dRe_c__dRe

            if self.k_lam == 0:
                dcd__dRe = dcdturb_total__dRe
            elif self.k_lam < 1:
                dcd__dRe = self.k_lam*(dcdlam_tr__dRe - dcdturb_tr__dRe) + dcdturb_total__dRe
            else:
                dcd__dRe = 0.
            ddoq__dRe = 2*chords*dcd__dRe

            dDoq__dRe = np.sum(widths*ddoq__dRe*FF)

            partials['CDv', 're'] = dDoq__dRe/S_ref

            if self.surface['symmetry']:
                partials['CDv', 'lengths'][0, :] *=  2
                partials['CDv', 'widths'][0, :] *= 2
                partials['CDv', 'S_ref'] *=  2
                partials['CDv', 'cos_sweep'][0, :] *=  2
                partials['CDv', 'M'][0, :] *=  2
                partials['CDv', 're'][0, :] *=  2
