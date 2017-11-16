import numpy as np
import scipy.sparse.linalg


class OASLinearSolver(object):

    def __init__(self, krylov_solver_name=None, maxiter=10, tol=1e-15):
        if krylov_solver_name == None:
            self.krylov_solver = None
        elif krylov_solver_name == 'cg':
            self.krylov_solver = scipy.sparse.linalg.cg
            self.callback_func = self._print_sol
        elif krylov_solver_name == 'bicgstab':
            self.krylov_solver = scipy.sparse.linalg.bicgstab
            self.callback_func = self._print_sol
        elif krylov_solver_name == 'gmres':
            self.krylov_solver = scipy.sparse.linalg.gmres
            self.callback_func = self._print_res

        self.maxiter = maxiter
        self.tol = tol

    def _print_res(self, res):
        print(self.counter, np.linalg.norm(res))
        self.counter += 1

    def _print_sol(self, sol):
        res = self.mtx.dot(sol) - self.rhs
        norm = np.linalg.norm(res)
        print(self.counter, np.linalg.norm(norm))
        self.counter += 1

    def _lu_solve_fwd(self, rhs):
        sol = scipy.linalg.lu_solve(self.lu, rhs, trans=0)
        return sol

    def _lu_solve_rev(self, rhs):
        sol = scipy.linalg.lu_solve(self.lu, rhs, trans=1)
        return sol

    def solve(self, rhs, mode='fwd'):
        if self.krylov_solver is None:
            if mode == 'fwd':
                return self._lu_solve_fwd(rhs)
            if mode == 'rev':
                return self._lu_solve_rev(rhs)

        if mode == 'fwd':
            pc_op = scipy.sparse.linalg.LinearOperator(self.mtx.shape, matvec=self._lu_solve_fwd)
        elif mode == 'rev':
            pc_op = scipy.sparse.linalg.LinearOperator(self.mtx.shape, matvec=self._lu_solve_rev)

        self.counter = 0
        self.rhs = rhs
        sol, _ = self.krylov_solver(self.mtx, rhs, M=pc_op, callback=self.callback_func,
            tol=self.tol, maxiter=self.maxiter)

        return sol
