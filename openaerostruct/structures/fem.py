"""Define the LinearSystemComp class."""
from __future__ import division, print_function

from six.moves import range

import numpy as np
from scipy import linalg

from openmdao.core.implicitcomponent import ImplicitComponent


class FEM(ImplicitComponent):
    """
    Component that solves a linear system, Ax=b.

    Designed to handle small and dense linear systems that can be
    efficiently solved with lu-decomposition

    Attributes
    ----------
    _lup : object
        matrix factorization returned from scipy.linag.lu_factor
    """

    def initialize(self):
        """
        Declare metadata.
        """
        self.metadata.declare('size', default=1, type_=int, desc='the size of the linear system')
        self.metadata.declare('partial_type', default='dense',
                              values=['dense', 'sparse', 'matrix_free'],
                              desc='the way the derivatives are defined')

    def initialize_variables(self):
        """
        Matrix and RHS are inputs, solution vector is the output.
        """
        size = self.metadata['size']

        self._lup = None

        if self.metadata['partial_type'] == "matrix_free":
            self.apply_linear = self._mat_vec_prod

        self.add_input('K', val=np.eye(size))
        self.add_input('forces', val=np.ones(size))
        self.add_output('disp_aug', shape=size, val=.1)

    def initialize_partials(self):
        """
        Set up the derivatives according to the user specified mode.
        """
        partial_type = self.metadata['partial_type']

        size = self.metadata['size']
        row_col = np.arange(size, dtype="int")

        if partial_type == 'sparse':
            self.declare_partials('disp_aug', 'forces', val=-np.ones(size), rows=row_col, cols=row_col)
            # self.declare_partials('disp_aug', 'forces', val=-1, rows=row_col, cols=row_col)

            rows = []
            cols = []
            for i in range(size):
                for j in range(size):
                    rows.append(i)
                    cols.append(i * size + j)

            self.dx_da_rows = rows
            self.dx_da_cols = cols

            self.declare_partials('disp_aug', 'K', val=np.ones(size**2), rows=rows, cols=cols)

        elif partial_type == "dense":
            self.declare_partials('disp_aug', 'forces', val=-np.eye(size))

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        R = Ax - b.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        residuals['disp_aug'] = inputs['K'].dot(outputs['disp_aug']) - inputs['forces']

    def solve_nonlinear(self, inputs, outputs):
        """
        Use numpy to solve Ax=b for x.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        # lu factorization for use with solve_linear
        self._lup = linalg.lu_factor(inputs['K'])
        outputs['disp_aug'] = linalg.lu_solve(self._lup, inputs['forces'])

    def linearize(self, inputs, outputs, J):
        """
        Compute the non-constant partial derivatives.
        """
        partial_type = self.metadata['partial_type']
        if partial_type == "matrix_free":
            return

        x = outputs['disp_aug']
        size = self.metadata['size']
        if partial_type == "dense":
            dx_dA = np.zeros((size, size**2))
            for i in range(size):
                dx_dA[i, i * size:(i + 1) * size] = x
            J['disp_aug', 'K'] = dx_dA

            J['disp_aug', 'disp_aug'] = inputs['K']

            # constant, defined int initialize_partials
            # J['disp_aug', 'forces'] = -np.eye()

        elif partial_type == "sparse":

            J['disp_aug', 'K'] = np.tile(x, size)
            # J['disp_aug', 'K'].set_data(np.tile(x, size))
            J['disp_aug', 'disp_aug'] = inputs['K']

            # constant, defined int initialize_partials
            # J['disp_aug', 'forces'] = -np.ones(size)

    def _mat_vec_prod(self, inputs, outputs, d_inputs, d_outputs,
                      d_residuals, mode):
        """
        Compute jac-vector product.

        linear operator for the partial derivative jacobian, only used if the 'partial_type'
        metadata is set to 'matrix_free'.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        d_residuals : Vector
            see outputs
        mode : str
            either 'fwd' or 'rev'
        """
        if mode == 'fwd':

            if 'disp_aug' in d_outputs:
                d_residuals['disp_aug'] += inputs['K'].dot(d_outputs['disp_aug'])
            if 'K' in d_inputs:
                d_residuals['disp_aug'] += d_inputs['K'].dot(outputs['disp_aug'])
            if 'forces' in d_inputs:
                d_residuals['disp_aug'] -= d_inputs['forces']

        elif mode == 'rev':

            if 'disp_aug' in d_outputs:
                d_outputs['disp_aug'] += inputs['K'].T.dot(d_residuals['disp_aug'])
            if 'K' in d_inputs:
                d_inputs['K'] += np.outer(outputs['disp_aug'], d_residuals['disp_aug']).T
            if 'forces' in d_inputs:
                d_inputs['forces'] -= d_residuals['disp_aug']

    def solve_linear(self, d_outputs, d_residuals, mode):
        r"""
        Back-substitution to solve the derivatives of the linear system.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

        Parameters
        ----------
        d_outputs : Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'
        """
        if mode == 'fwd':
            sol_vec, rhs_vec = d_outputs, d_residuals
            t = 0
        else:
            sol_vec, rhs_vec = d_residuals, d_outputs
            t = 1

        # print("foobar", rhs_vec['disp_aug'])
        sol_vec['disp_aug'] = linalg.lu_solve(self._lup, rhs_vec['disp_aug'], trans=t)
