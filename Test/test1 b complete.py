import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return -2 * x * np.sin(x)

class PSO:
    def __init__(self, x, v, c, w):
        self.x = x
        self.v = v
        self.c = c
        self.w = w

        self.oldX = list(x)
        self.pBest = list(x)
        self.gBest = 0

    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i]) < f(self.pBest[i]):
                self.pBest[i] = self.x[i]
            else:
                self.pBest[i] = self.oldX[i]

    def findGBest(self):
        minVal = f(self.x[0])
        minIndex = 0

        for i in range(1, len(self.x)):
            fx = f(self.x[i])
            if fx < minVal:
                minVal = fx
                minIndex = i
        self.gBest = self.x[minIndex]

    def updateV(self, r1, r2):
        for i in range(len(self.x)):
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * r1 * (self.pBest[i] - self.x[i])) + (self.c[1] * r2 * (self.gBest - self.x[i]))
        print(f"r1 = {r1}, r2 = {r2}")

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] += self.v[i]

    def plot_particles(self, ax):
        ax.scatter(self.x, [f(xi) for xi in self.x], c='b', marker='o', label='Particles')
        ax.scatter(self.gBest, f(self.gBest), c='r', marker='o', s=100, label='Global Best')

    def animate(self, i, ax):
        if i == 0:
            print("Iterasi ke-0")
            print("x =", self.x)
            print("v =", self.v)
            print("pBest =", self.pBest)
            print("gBest =", self.gBest)
            print("f(x) sebelum =", [f(val) for val in self.x], "\n")

        self.findPBest()
        self.findGBest()
        self.updateV(np.random.rand(), np.random.rand())
        self.updateX()
        ax.clear()
        self.plot_particles(ax)
        ax.set_title(f'Iteration {i}')
        ax.set_xlim(-2, 4)
        ax.set_ylim(-15, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('f(X)')
        ax.legend()

        print(f"Iterasi {i+1}")
        print(f"x = {self.x}")
        print(f"v = {self.v}")
        print(f"pBest = {self.pBest}")
        print(f"gBest = {self.gBest}")
        print(f"f(gBest) = {f(self.gBest)}")
        print(f"f(x) = {[f(x) for x in self.x]}")
        print()

    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

# Nilai awal X0 sebagai 10 bilangan acak
initial_x = np.random.rand(10)
print("Bilangan acak untuk nilai awal X0:", initial_x)

v = np.zeros(10)
c = [1/2, 1]
w = 1

pso = PSO(initial_x, v, c, w)
pso.iterate_with_animation(50)
