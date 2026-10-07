import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        xi = np.linspace(0, 1, N + 1)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=sparse)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags(
            [1.0, -2.0, 1.0],
            [-1, 0, 1],
            (N + 1, N + 1)
        )

        D2 = (
            sparse.kron(D, sparse.eye(N + 1))
            + sparse.kron(sparse.eye(N + 1), D)
        ).tolil()

        return D2

    @property
    def w(self):
        return self.c *np.pi *np.sqrt(self.mx**2 + self.my**2)


    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        xij, yij = self.create_mesh(N)
        ue = self.ue(mx, my)
        f=sp.lambdify((x, y, t), ue, "numpy")
        u0=np.asarray(f(xij, yij, 0), dtype=float)  
        u1=np.asarray(f(xij, yij, self.dt), dtype=float)
        return u0, u1

    @property
    def dt(self) -> float:
       h=1.0/ self.N
       return self.cfl * h / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        xij, yij = self.create_mesh(self.N)
        ue = self.ue(self.mx, self.my)
        f=sp.lambdify((x, y, t), ue, "numpy")
        u_exact=np.asarray(f(xij, yij, t0), dtype=float)
        h=1.0/self.N
        return np.sqrt(h**2 * np.sum((u - u_exact)**2))

    def apply_bcs(self, u: np.ndarray):
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        self.N = N
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        h=1.0 / N
        dt=self.dt
        
        D2 = self.D2(N)

        unm1, un = self.initialize(N, mx, my)
        self.apply_bcs(unm1)
        self.apply_bcs(un)

        errors = [self.l2_error(unm1, 0.0)]

        if store_data > 0:
            data = {0:unm1.copy()}

        for n in range(1, Nt):
            unp1 = (
                2 * un - unm1 + (c*dt/h)**2 * (D2 @ un.ravel()).reshape((N+1, N+1))
            )

            self.apply_bcs(unp1)

            t0 = n * dt
            errors.append(self.l2_error(unp1, t0+dt))

            if store_data >0 and n % store_data == 0:
                data[n] = unp1.copy()

            unm1, un = un, unp1

        if store_data == -1:
             return h, np.array(errors)
            
        return data


    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags(
            [1.0, -2.0, 1.0],
            [-1, 0, 1],
            (N + 1, N + 1)
        ).tolil()

        D[0, 0] = -2.0
        D[0, 1] = 2.0
        D[N,N] = -2.0
        D[N,N-1] = 2.0

        D2 = (
            sparse.kron(D, sparse.eye(N + 1))
            + sparse.kron(sparse.eye(N + 1), D)
        ).tolil()
        return D2
        
    def ue(self, mx: int, my: int) -> sp.Expr:
        return (
            sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)
        )

    def apply_bcs(self, u: np.ndarray):
       pass 



def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    cfl=1/np.sqrt(2)
    sol = Wave2D()
    h, errors=sol(N=40, Nt=40, cfl=cfl, mx=2, my=2)
    assert errors[-1] < 1e-12
    solN = Wave2D_Neumann()
    h, errors = solN(N=40, Nt=40, cfl=cfl, mx=2, my=2)
    assert errors[-1] < 1e-12
    
if __name__ == "__main__":
    test_convergence_wave2d()
    print("dirichlet passed")