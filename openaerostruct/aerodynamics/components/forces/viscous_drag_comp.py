from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent

class ViscousDragComp(ExplicitComponent):
    """
    Compute the skin friction drag if the with_viscous option is True.
    If not, the CDv is 0.
    This component exists for each lifting surface.

    Parameters
    ----------
    re : float
        Dimensionalized (1/length) Reynolds number. This is used to compute the
        local Reynolds number based on the local chord length.
    Mach : float
        Mach number.
    reference_area : float
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
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('lifting_surfaces', types=list)

    def setup(self):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        self.with_viscous = True
        # Percentage of chord with laminar flow
        self.k_lam = .05
        # Thickness over chord for the airfoil
        self.t_over_c = .12
        self.c_max_t = .25

        self.add_input('mu_Pa_s', shape=num_nodes, val=1e-1)
        self.add_input('rho_kg_m3', shape=num_nodes, val=.384)
        self.add_input('Mach', shape=num_nodes)
        self.add_input('v_m_s', shape=num_nodes, val=200.)
        self.add_input('reference_area', shape=num_nodes)

        arange = np.arange(num_nodes)

        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data.num_points_x
            num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

            sweep_name = '{}_cos_sweep'.format(lifting_surface_name)
            widths_name = '{}_widths'.format(lifting_surface_name)
            lengths_name = '{}_lengths'.format(lifting_surface_name)

            self.add_input(sweep_name, shape=(num_nodes, num_points_z - 1))
            self.add_input(widths_name, shape=(num_nodes, num_points_z - 1))
            self.add_input(lengths_name, shape=(num_nodes, num_points_z))

            self.declare_partials('CDv', sweep_name, method='fd')
            self.declare_partials('CDv', widths_name, method='fd')
            self.declare_partials('CDv', lengths_name, method='fd')

        self.add_output('CDv', val=0.)

        self.declare_partials('CDv', 'v_m_s', method='cs')
        self.declare_partials('CDv', 'rho_kg_m3', method='cs')
        self.declare_partials('CDv', 'Mach', method='cs')
        # TODO: these are wrong; cs doesn't approximate them correctly
        self.declare_partials('CDv', 'mu_Pa_s', method='cs')  # some part of these appears to not be complex safe
        self.declare_partials('CDv', 'reference_area', cols=arange, rows=arange)

    def compute(self, inputs, outputs):
        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        if self.with_viscous:
            Mach = inputs['Mach']
            reference_area = inputs['reference_area']
            re = inputs['rho_kg_m3'] * inputs['v_m_s'] / inputs['mu_Pa_s']

            D_over_q = 0.

            for lifting_surface_name, lifting_surface_data in lifting_surfaces:
                num_points_x = lifting_surface_data.num_points_x
                num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

                sweep_name = '{}_cos_sweep'.format(lifting_surface_name)
                widths_name = '{}_widths'.format(lifting_surface_name)
                lengths_name = '{}_lengths'.format(lifting_surface_name)

                widths = inputs[widths_name]
                lengths = inputs[lengths_name]
                cos_sweep = inputs[sweep_name] / widths

                # Take panel chord length to be average of its edge lengths
                chords = (lengths[:, 1:] + lengths[:, :-1]) / 2.
                Re_c = re * chords

                cdturb_total = 0.455 / (np.log10(Re_c))**2.58 / \
                    (1.0 + 0.144*Mach**2)**0.65
                cdlam_tr = 1.328 / np.sqrt(Re_c * self.k_lam)

                # Use eq. 12.27 of Raymer for turbulent Cf
                if self.k_lam == 0:
                    cdlam_tr = 0.
                    cd = cdturb_total

                elif self.k_lam < 1.0:
                    cdturb_tr = 0.455 / (np.log10(Re_c*self.k_lam))**2.58 / \
                        (1.0 + 0.144*Mach**2)**0.65

                else:
                    cdturb_total = 0.

                cd = (cdlam_tr - cdturb_tr)*self.k_lam + cdturb_total

                d_over_q = 2 * cd * chords

                # Calculate form factor (Raymer Eq. 12.30)
                k_FF = 1.34 * Mach**0.18 * \
                    (1.0 + 0.6*self.t_over_c/self.c_max_t + 100*self.t_over_c**4)
                FF = k_FF * cos_sweep**0.28

                # Sum individual panel drags to get total drag
                D_over_q += np.sum(d_over_q * widths * FF)

            outputs['CDv'] = D_over_q / reference_area

        else:
            outputs['CDv'] = 0.0

    def compute_partials(self, inputs, partials):
        """ Jacobian for viscous drag."""

        num_nodes = self.metadata['num_nodes']
        lifting_surfaces = self.metadata['lifting_surfaces']

        if self.with_viscous:
            re = inputs['rho_kg_m3'] * inputs['v_m_s'] / inputs['mu_Pa_s']

            for lifting_surface_name, lifting_surface_data in lifting_surfaces:
                num_points_x = lifting_surface_data.num_points_x
                num_points_z = 2 * lifting_surface_data.num_points_z_half - 1

                sweep_name = '{}_cos_sweep'.format(lifting_surface_name)
                widths_name = '{}_widths'.format(lifting_surface_name)
                lengths_name = '{}_lengths'.format(lifting_surface_name)

                partials['CDv', lengths_name] = np.zeros_like(partials['CDv', lengths_name])

                p180 = np.pi / 180.
                Mach = inputs['Mach'][0]
                reference_area = inputs['reference_area'][0]

                widths = inputs[widths_name]
                lengths = inputs[lengths_name]
                cos_sweep = inputs[sweep_name] / widths

                # Take panel chord length to be average of its edge lengths
                chords = (lengths[:, 1:] + lengths[:, :-1]) / 2.
                Re_c = re * chords

                cdturb_total = 0.455 / (np.log10(Re_c))**2.58 / \
                    (1.0 + 0.144*Mach**2)**0.65
                cdlam_tr = 1.328 / np.sqrt(Re_c * self.k_lam)

                # Use eq. 12.27 of Raymer for turbulent Cf
                if self.k_lam == 0:
                    cdlam_tr = 0.
                    cd = cdturb_total

                elif self.k_lam < 1.0:
                    cdturb_tr = 0.455 / (np.log10(Re_c*self.k_lam))**2.58 / \
                        (1.0 + 0.144*Mach**2)**0.65

                else:
                    cdturb_total = 0.

                cd = (cdlam_tr - cdturb_tr)*self.k_lam + cdturb_total

                # Multiply by section width to get total normalized drag for section
                d_over_q = 2 * cd * chords

                # Calculate form factor (Raymer Eq. 12.30)
                k_FF = 1.34 * Mach**0.18 * \
                    (1.0 + 0.6*self.t_over_c/self.c_max_t + 100*self.t_over_c**4)
                FF = k_FF * cos_sweep**0.28

                # Sum individual panel drags to get total drag
                D_over_q = np.sum(d_over_q * widths * FF)

                cdl_Re = 0.0
                cdt_Re = 0.0
                cdT_Re = 0.0

                B = (1. + 0.144*Mach**2)**0.65

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

                CDv_lengths = 2 * widths * FF / reference_area * \
                    (d_over_q / 4 / chords + chords * cd_Re * re / 2.)
                partials['CDv', lengths_name][:, 1:] += CDv_lengths
                partials['CDv', lengths_name][:, :-1] += CDv_lengths
                partials['CDv', widths_name] = d_over_q * FF / reference_area * 0.72
                partials['CDv', 'reference_area'] = -D_over_q / reference_area**2
                partials['CDv', sweep_name] = 0.28 * k_FF * d_over_q / reference_area / cos_sweep**0.72

if __name__ == "__main__":
    from openaerostruct.tests.utils import run_test, get_default_lifting_surfaces

    lifting_surfaces = get_default_lifting_surfaces()

    run_test('dummy', ViscousDragComp(num_nodes=1, lifting_surfaces=lifting_surfaces))
