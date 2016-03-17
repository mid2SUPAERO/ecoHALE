from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.solvers.newton import Newton
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel


class HybridGSNewton(NonLinearSolver):

    def __init__(self): 
        super(HybridGSNewton, self).__init__()

        self.nlgs = NLGaussSeidel()
        self.newton = Newton()

        self.nlgs.options['maxiter'] = 5

    def solve(self, params, unknowns, resids, system, metadata=None): 

        self.nlgs.solve(params, unknowns, resids, system, metadata)
        self.newton.solve(params, unknowns, resids, system, metadata)
