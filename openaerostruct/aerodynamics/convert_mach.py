from __future__ import print_function, division
import numpy as np

from openmdao.api import ExplicitComponent


class ConvertMach(ExplicitComponent):
    """
    Compute the wave drag if the with_wave option is True. If not, the CDw is 0.
    This component exists for each lifting surface.

    Parameters
    ----------
    Mach_number : float
        Mach number.
    cos_sweep[ny-1] : numpy array
        The width in the spanwise direction of each VLM panel. This is the numerator of cos(sweep).
    widths[ny-1] : numpy array
        The actual width of each VLM panel, rotated by the sweep angle. This is the denominator
        of cos(sweep)
    chords[ny] : numpy array
        The chord length of each mesh slice. This is dimension ny rather than ny-1 which would be
        expected for chord length of each VLM panel.

    Returns
    -------
    normal_Mach : float
        Mach number normal to the averaged quarter-chord sweep of the wing
    """

    def initialize(self):
        self.options.declare('surface', types=dict)

    def setup(self):
        ny = self.options['surface']['mesh'].shape[1]

        self.add_input('Mach_number', val=1.6)
        self.add_input('cos_sweep', val=np.ones((ny-1))*.2, units='m')
        self.add_input('widths', val=np.arange((ny-1))+1., units='m')
        self.add_input('chords', val=np.ones((ny)), units='m')

        self.add_output('normal_Mach', val=1.2)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        actual_cos_sweep = inputs['cos_sweep'] / inputs['widths']
        M = inputs['Mach_number']
        chords = inputs['chords']

        mean_chords = (chords[:-1] + chords[1:]) / 2.
        panel_areas = mean_chords * inputs['cos_sweep']
        avg_cos_sweep = np.sum(actual_cos_sweep * panel_areas) / np.sum(panel_areas) # weighted average of 1/4 chord sweep

        outputs['normal_Mach'] = M * avg_cos_sweep
