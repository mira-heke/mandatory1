import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Wave2D import Wave2D_Neumann

sol = Wave2D_Neumann()

N=40
Nt=100
cfl = 1/np.sqrt(2)

data= sol (
    N=N, Nt=Nt, cfl=cfl, mx=2, my=2, store_data=4,
)

xij, yij = sol.create_mesh(N)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

frames = []

for n, val in data.items():
    frame= ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2 )
    frames.append([frame])

ani= animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)

ani.save("report/neumannwave.gif", writer="pillow", fps=5)
