import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return (x + y**2 - 13)**2 + (x**2 + y - 9)**2

class PSO:
    def __init__(self, x: list, y: list, v: list, c: list, r: list, w: float):
        self.x = x
        self.y = y
        self.vx = v
        self.vy = v.copy()
        self.c = c
        self.r = r
        self.w = w

        self.oldX = x.copy()
        self.oldY = y.copy()
        self.pBestX = x.copy()
        self.pBestY = y.copy()
        self.gBestX = self.x[np.argmin([f(x, y) for x, y in zip(self.x, self.y)])]
        self.gBestY = self.y[np.argmin([f(x, y) for x, y in zip(self.x, self.y)])]

    def findPBest(self):
        for i in range(len(self.x)):
            value = f(self.x[i], self.y[i])
            pBestFValue = f(self.pBestX[i], self.pBestY[i])
            if value < pBestFValue:
                self.pBestX[i] = self.x[i]
                self.pBestY[i] = self.y[i]
            else:
                self.pBestX[i] = self.oldX[i]
                self.pBestY[i] = self.oldY[i]

    def findGBest(self):
        fValues = [f(x, y) for x, y in zip(self.x, self.y)]
        minimumIndex = np.argmin(fValues)
        self.gBestX = self.x[minimumIndex]
        self.gBestY = self.y[minimumIndex]

    def updateV(self):
        for i in range(len(self.x)):
            self.vx[i] = (self.w * self.vx[i]) + (self.c[0] * self.r[0] * (self.pBestX[i] - self.x[i])) + (
                    self.c[1] * self.r[1] * (self.gBestX - self.x[i]))
            self.vy[i] = (self.w * self.vy[i]) + (self.c[0] * self.r[0] * (self.pBestY[i] - self.y[i])) + (
                    self.c[1] * self.r[1] * (self.gBestY - self.y[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.oldY[i] = self.y[i]
            self.x[i] = self.x[i] + self.vx[i]
            self.y[i] = self.y[i] + self.vy[i]

    def print_info(self, i):
        print(f"Iteration {i}")
        print(f"x = {self.x}")
        print(f"y = {self.y}")
        print(f"vx = {self.vx}")
        print(f"vy = {self.vy}")
        print(f'pBest = {self.pBestX}')
        print(f'gBest = {self.gBestX}')
        print(f'pBest = {self.pBestY}')
        print(f'gBest = {self.gBestY}')
        print(f'f(gBest x, gBest y) = {f(self.gBestX, self.gBestY)}')
        print()

    def plot_particles(self, ax):
        ax.scatter(self.x, self.y, [f(xi, yi) for xi, yi in zip(self.x, self.y)], c='b', marker='o', label='Particles')
        ax.scatter(self.gBestX, self.gBestY, f(self.gBestX, self.gBestY), c='r', marker='o', s=100, label='Global Best')

    def animate(self, i, ax):
        self.findPBest()
        self.findGBest()
        self.updateV()
        self.updateX()
        self.print_info(i)
        ax.clear()
        self.plot_particles(ax)
        ax.set_title(f'Iteration {i}')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    def iterate(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

x = np.array([1.0, 1.0, 0.0])
y = np.array([1.0, -1.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
c = np.array([1.0, 1.0])
r = np.array([1.0, 0.5])
w = 1

pso = PSO(x, y, v, c, r, w)
pso.iterate(50)
