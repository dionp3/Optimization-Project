import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return (-2 * x) * np.sin(x)

class PSO:
    def __init__(self, x: list, v: list, c: list, r: list, w: float):
        self.x = x
        self.v = v
        self.c = c
        self.r = r
        self.w = w

        self.oldX = x.copy()
        self.pBest = x.copy()
        self.gBest = None

    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i]) < f(self.pBest[i]):
                self.pBest[i] = self.x[i]
            else:
                self.pBest[i] = self.oldX[i]

    def findGBest(self):
        fValues = [f(x) for x in self.x]
        self.gBest = self.x[np.argmin(fValues)]

    def updateV(self):
        for i in range(len(self.x)):
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * self.r[0] * (self.pBest[i] - self.x[i])) + (
                    self.c[1] * self.r[1] * (self.gBest - self.x[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] = self.x[i] + self.v[i]
    
    def print_info(self, i):
        print(f"Iteration {i}")
        print(f"x = {self.x}")
        print(f"vx = {self.vx}")
        print(f'pBest = {self.pBestX}')
        print(f'gBest = {self.gBestX}')
        print(f'f(gBest x = {f(self.gBestX)}')
        print()

    def plot_particles(self, ax):
        ax.scatter(self.x, [f(xi) for xi in self.x], c='b', marker='o', label='Particles')
        ax.scatter(self.gBest, f(self.gBest), c='r', marker='o', s=100, label='Global Best')

    def animate(self, i, ax):
        self.findPBest()
        self.findGBest()
        self.updateV()
        self.updateX()
        ax.clear()
        self.plot_particles(ax)
        ax.set_title(f'Iteration {i}')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('f(X)')
        ax.legend()

    def iterate(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

x = np.array([1, np.pi/2, np.pi])
v = np.array([0, 0, 0])
c = np.array([1/2, 1])
r = np.array([1, 1])
w = 1

pso = PSO(x, v, c, r, w)
pso.iterate(50)
