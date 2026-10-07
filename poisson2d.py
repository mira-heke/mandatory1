import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        h = self.p.L / N
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1))
        D2x = D2 / h**2
        D2y = D2 / h**2
        return (sparse.kron(D2x, sparse.eye(N + 1)) +
                sparse.kron(sparse.eye(N + 1), D2y)).tolil()

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        xij, yij = self.create_mesh(N)
        A = self.laplace(N)
        b = self.meshfunction(f, xij, yij)
        uexact = self.meshfunction(ue, xij, yij)
        b = b.ravel()
        uexact = uexact.ravel()
        boundary = self.get_boundary_indices(N)
        for i in boundary:
            A.rows[i] = [i]
            A.data[i] = [1.0]
            b[i] = uexact[i]

        return A.tocsr(), b

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        f = sp.lambdify((x, y), u, "numpy")
        return np.asarray(f(xij, yij), dtype=float)
    

    def get_boundary_indices(self, N: int) -> np.ndarray:
        indices = np.arange((N + 1) ** 2).reshape((N + 1, N + 1))
        boundary = np.concatenate([
            indices[0, :],
            indices[-1, :],
            indices[1:-1, 0],
            indices[1:-1, -1],
        ])
        return boundary

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        N = u.shape[0] - 1
        h = self.p.L / N
        xij, yij = self.create_mesh(N)
        uexact = self.meshfunction(ue, xij, yij)
        return np.sqrt(h**2 * np.sum((uexact - u)**2))

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        A, b = self.assemble(
            N,
            sp.diff(ue, x, 2) + sp.diff(ue, y, 2), 
            ue)
        return sparse_linalg.spsolve(A, b).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        N = U.shape[0] - 1
        h = self.p.L / N
        x = np.clip(x, 0, self.p.L)
        y = np.clip(y, 0, self.p.L)

        i = min(int(x / h), N - 1)
        j = min(int(y / h), N - 1)

        tx = (x - i * h) / h
        ty = (y - j * h) / h

        u00 = U[i, j]
        u10 = U[i + 1, j]
        u01 = U[i, j + 1]
        u11 = U[i + 1, j + 1]

        return float(
            (1 - tx) * (1 - ty) * u00
            + tx * (1 - ty) * u10
            + (1 - tx) * ty * u01
            + tx * ty * u11
        )


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")

