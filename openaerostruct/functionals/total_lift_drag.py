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
        self.metadata.declare('prob_dict', type_=dict, required=True)

    def setup(self):
        for surface in self.metadata['surfaces']:
            name = surface['name']
            self.add_input(name + 'CL', val=1.)
            self.add_input(name + 'CD', val=1.)
            self.add_input(name + 'S_ref', val=1.)

        self.add_output('CL', val=1.)
        self.add_output('CD', val=1.)

    def compute(self, inputs, outputs):
        prob_dict = self.metadata['prob_dict']

        # Compute the weighted CL and CD contributions from each surface,
        # weighted by the individual surface areas
        CL = 0.
        CD = 0.
        computed_total_S_ref = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            S_ref = inputs[name + 'S_ref']
            CL += inputs[name + 'CL'] * S_ref
            CD += inputs[name + 'CD'] * S_ref
            computed_total_S_ref += S_ref

        # Use the user-provided area; otherwise, use the computed area
        if self.metadata['prob_dict']['S_ref_total'] is not None:
            S_ref_total = self.metadata['prob_dict']['S_ref_total']
        else:
            S_ref_total = computed_total_S_ref

        outputs['CL'] = CL / S_ref_total
        outputs['CD'] = CD / S_ref_total

    def compute_partials(self, inputs, outputs, partials):
        prob_dict = self.metadata['prob_dict']

        # Compute the weighted CL and CD contributions from each surface,
        # weighted by the individual surface areas
        CL = 0.
        CD = 0.
        computed_total_S_ref = 0.
        for surface in self.metadata['surfaces']:
            name = surface['name']
            S_ref = inputs[name + 'S_ref']
            CL += inputs[name + 'CL'] * S_ref
            CD += inputs[name + 'CD'] * S_ref
            computed_total_S_ref += S_ref

        if self.metadata['prob_dict']['S_ref_total'] is not None:
            S_ref_total = self.metadata['prob_dict']['S_ref_total']
        else:
            S_ref_total = computed_total_S_ref

        for surface in self.metadata['surfaces']:
            name = surface['name']
            S_ref = inputs[name + 'S_ref']
            partials['CL', name + 'CL'] = S_ref / S_ref_total
            partials['CD', name + 'CD'] = S_ref / S_ref_total

            dCL_dS_ref = 0.
            surf_CL = inputs[name + 'CL']
            dCD_dS_ref = 0.
            surf_CD = inputs[name + 'CD']
            for surface_ in self.metadata['surfaces']:
                name_ = surface_['name']
                if not name == name_:
                    S_ref_ = inputs[name_ + 'S_ref']
                    dCL_dS_ref += surf_CL * S_ref_
                    dCL_dS_ref -= inputs[name_ + 'CL'] * S_ref_
                    dCD_dS_ref += surf_CD * S_ref_
                    dCD_dS_ref -= inputs[name_ + 'CD'] * S_ref_

            partials['CL', name + 'S_ref'] = dCL_dS_ref / S_ref_total**2
            partials['CD', name + 'S_ref'] = dCD_dS_ref / S_ref_total**2
