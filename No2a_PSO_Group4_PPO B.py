import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return (x + y**2 - 13)**2 + (x**2 + y - 9)**2

class PSO:
    def __init__(self, x, y, v, c, r, w):
        self.x = x
        self.y = y
        self.vx = v
        self.vy = v.copy()
        self.c = c
        self.r = r
        self.w = w
        self.oldX = list(x)
        self.oldY = list(y)
        self.pBest = list(zip(x, y))
        self.gBest = (0, 0)
        self.first_iteration = True

    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i], self.y[i]) < f(self.pBest[i][0], self.pBest[i][1]):
                self.pBest[i] = (self.x[i], self.y[i])
            else:
                self.pBest[i] = (self.oldX[i], self.oldY[i])

    def findGBest(self):
        minVal = f(self.x[0], self.y[0])
        minIndex = 0
        for i in range(1, len(self.x)):
            fx = f(self.x[i], self.y[i])
            if fx < minVal:
                minVal = fx
                minIndex = i
        self.gBest = (self.x[minIndex], self.y[minIndex])

    def updateV(self):
        for i in range(len(self.x)):
            self.vx[i] = (self.w * self.vx[i]) + (self.c[0] * self.r[0] * (self.pBest[i][0] - self.x[i])) + (self.c[1] * self.r[1] * (self.gBest[0] - self.x[i]))
            self.vy[i] = (self.w * self.vy[i]) + (self.c[0] * self.r[0] * (self.pBest[i][1] - self.y[i])) + (self.c[1] * self.r[1] * (self.gBest[1] - self.y[i]))
        self.v = [(self.vx[i], self.vy[i]) for i in range(len(self.x))]

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.oldY[i] = self.y[i]
            self.x[i] = self.x[i] + self.v[i][0]
            self.y[i] = self.y[i] + self.v[i][1]

    def plot_particles(self, ax):
        ax.scatter(self.x, self.y, [f(xi, yi) for xi, yi in zip(self.x, self.y)], c='b', marker='o', label='Particles')
        ax.scatter(self.gBest[0], self.gBest[1], f(self.gBest[0], self.gBest[1]), c='r', marker='o', s=100, label='Global Best')

    def animate(self, i, ax):
        if self.first_iteration:
            self.first_iteration = False
        else:
            print(f"Iterasi {i+1}")
            print(f"x = {[round(val, 3) for val in self.x]}")
            print(f"y = {[round(val, 3) for val in self.y]}")
            print(f"f(x, y) = {[round(f(val[0], val[1]), 3) for val in zip(self.x, self.y)]}")
            self.findPBest()
            self.findGBest()
            self.updateV()
            self.updateX()
            print(f"pBest = {[(round(val[0], 3), round(val[1], 3)) for val in self.pBest]}")
            print(f"gBest x, y = {(round(self.gBest[0], 3), round(self.gBest[1], 3))}")
            print(f"gBest f(x, y) = {round(f(self.gBest[0], self.gBest[1]), 3)}")
            print(f"v = {[tuple(map(lambda x: round(x, 3), val)) for val in self.v]}")
            print(f"Update x = {[round(val, 3) for val in self.x]}")
            print(f"Update y = {[round(val, 3) for val in self.y]}")
            print(f"Update f(x, y) = {[round(f(val[0], val[1]), 3) for val in zip(self.x, self.y)]}")
            print()

        ax.clear()
        self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f'Iteration {i+1}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.legend()

    def plot_surface(self, ax):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, label='Objective Function')

    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

    def print_optimization_result(self):
        print("\nHasil Optimasi:")
        print("Nilai Optimal x =", self.gBest[0])
        print("Nilai Optimal y =", self.gBest[1])
        print("Nilai Optimal f(x, y) =", f(self.gBest[0], self.gBest[1]))

print("No2a Particle Swarm Optimization Group 4 PPO B\n")

x = [1, -1, 2]
y = [1, -1, 1]
v = np.zeros(3)
c = [1, 1/2]
r = [1, 1]
w = 1

print("f(x, y) = (x + y**2 - 13)**2 + (x**2 + y - 9)**2")
print("\nNilai Awal:")
print("x =", x)
print("y =", y)
print("v =", v)
print("C =", c)
print("R =", r)
print("w =", w,"\n")

pso = PSO(x, y, v, c, r, w)

num_iterations = int(input("Masukkan jumlah iterasi: "))
pso.iterate_with_animation(num_iterations)

pso.print_optimization_result()