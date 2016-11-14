from __future__ import division
import numpy

from openmdao.api import Component



class MaterialsTube(Component):
    """ Compute geometric properties for a tube element.

    Parameters
    ----------
    r : array_like
        Radii for each FEM element.
    thickness : array_like
        Tube thickness for each FEM element.

    Returns
    -------
    A : array_like
        Areas for each FEM element.
    Iy : array_like
        Mass moment of inertia around the y-axis for each FEM element.
    Iz : array_like
        Mass moment of inertia around the z-axis for each FEM element.
    J : array_like
        Polar moment of inertia for each FEM element.

    """

    def __init__(self, surface):
        super(MaterialsTube, self).__init__()

        self.surface = surface

        self.ny = surface['num_y']
        self.nx = surface['num_x']
        self.n = self.nx * self.ny
        self.mesh = surface['mesh']
        name = surface['name']

        self.add_param('r', val=surface['r'])
        self.add_param('thickness', val=surface['t'])
        self.add_output('A', val=numpy.zeros((self.ny - 1)))
        self.add_output('Iy', val=numpy.zeros((self.ny - 1)))
        self.add_output('Iz', val=numpy.zeros((self.ny - 1)))
        self.add_output('J', val=numpy.zeros((self.ny - 1)))

        # self.deriv_options['type'] = 'cs'
        self.deriv_options['form'] = 'central'
        #self.deriv_options['extra_check_partials_form'] = "central"

        self.arange = numpy.arange((self.ny - 1))

    def solve_nonlinear(self, params, unknowns, resids):
        name = self.surface['name']
        pi = numpy.pi
        r1 = params['r'] - 0.5 * params['thickness']
        r2 = params['r'] + 0.5 * params['thickness']

        unknowns['A'] = pi * (r2**2 - r1**2)
        unknowns['Iy'] = pi * (r2**4 - r1**4) / 4.
        unknowns['Iz'] = pi * (r2**4 - r1**4) / 4.
        unknowns['J'] = pi * (r2**4 - r1**4) / 2.


    def linearize(self, params, unknowns, resids):
        name = self.surface['name']
        jac = self.alloc_jacobian()

        pi = numpy.pi
        r = params['r'].real
        t = params['thickness'].real
        r1 = r - 0.5 * t
        r2 = r + 0.5 * t

        dr1_dr = 1.
        dr2_dr = 1.
        dr1_dt = -0.5
        dr2_dt =  0.5

        r1_3 = r1**3
        r2_3 = r2**3

        a = self.arange
        jac['A', 'r'][a, a] = 2 * pi * (r2 * dr2_dr - r1 * dr1_dr)
        jac['A', 'thickness'][a, a] = 2 * pi * (r2 * dr2_dt - r1 * dr1_dt)
        jac['Iy', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        jac['Iy', 'thickness'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        jac['Iz', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        jac['Iz', 'thickness'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        jac['J', 'r'][a, a] = 2 * pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        jac['J', 'thickness'][a, a] = 2 * pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)

        return jac
