from __future__ import division, print_function
import numpy as np

from openmdao.api import ExplicitComponent

try:
    from openaerostruct.fortran import OAS_API
    fortran_flag = True
    data_type = float
except:
    fortran_flag = False
    data_type = complex


class TotalLiftDrag(ExplicitComponent):
    """
    Compute the coefficients of lift (CL) and drag (CD) for the entire aircraft.

    Parameters
    ----------
    CL : float
        Coefficient of lift (CL) for one lifting surface.
    CD : float
        Coefficient of drag (CD) for one lifting surface.
    S_ref : float
        Surface area for one lifting surface.

    Returns
    -------
    CL : float
        Total coefficient of lift (CL) for the entire aircraft.
    CD : float
        Total coefficient of drag (CD) for the entire aircraft.

    """

    def initialize(self):
        self.metadata.declare('surfaces', type_=list, required=True)

    def setup(self):
        for surface in self.metadata['surfaces']:
            name = surface['name']
            self.add_input(name + '_CL', val=1.)
            self.add_input(name + '_CD', val=1.)
            self.add_input(name + '_S_ref', val=1.)

        self.add_input('S_ref_total', val=0.)

        self.add_output('CL', val=1.)
        self.add_output('CD', val=1.)

    def compute(self, inputs, outputs):

        # Compute the weighted CL and CD contributions from each surface,
        # weighted by the individual surface areas
        CL = 0.
        CD = 0.
        computed_total_S_ref = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            S_ref = inputs[name + '_S_ref']
            CL += inputs[name + '_CL'] * S_ref
            CD += inputs[name + '_CD'] * S_ref
            computed_total_S_ref += S_ref

        # Use the user-provided area; otherwise, use the computed area
        if inputs['S_ref_total'] == 0.:
            S_ref_total = computed_total_S_ref
        else:
            S_ref_total = inputs['S_ref_total']

        outputs['CL'] = CL / S_ref_total
        outputs['CD'] = CD / S_ref_total

    def compute_partials(self, inputs, outputs, partials):

        # Compute the weighted CL and CD contributions from each surface,
        # weighted by the individual surface areas
        CL = 0.
        CD = 0.
        computed_total_S_ref = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            S_ref = inputs[name + '_S_ref']
            CL += inputs[name + '_CL'] * S_ref
            CD += inputs[name + '_CD'] * S_ref
            computed_total_S_ref += S_ref

        # Use the user-provided area; otherwise, use the computed area
        if inputs['S_ref_total'] == 0.:
            S_ref_total = computed_total_S_ref
        else:
            S_ref_total = inputs['S_ref_total']

        for surface in self.metadata['surfaces']:
            name = surface['name']
            S_ref = inputs[name + '_S_ref']
            partials['CL', name + '_CL'] = S_ref / S_ref_total
            partials['CD', name + '_CD'] = S_ref / S_ref_total

            dCL_dS_ref = 0.
            surf_CL = inputs[name + '_CL']
            dCD_dS_ref = 0.
            surf_CD = inputs[name + '_CD']
            for surface_ in self.metadata['surfaces']:
                name_ = surface_['name']
                if not name == name_:
                    S_ref_ = inputs[name_ + '_S_ref']
                    dCL_dS_ref += surf_CL * S_ref_
                    dCL_dS_ref -= inputs[name_ + '_CL'] * S_ref_
                    dCD_dS_ref += surf_CD * S_ref_
                    dCD_dS_ref -= inputs[name_ + '_CD'] * S_ref_

            partials['CL', name + '_S_ref'] = dCL_dS_ref / S_ref_total**2
            partials['CD', name + '_S_ref'] = dCD_dS_ref / S_ref_total**2
