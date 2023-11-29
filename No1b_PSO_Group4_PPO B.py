import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return (-2 * x) * (np.sin(x))

class PSO:
    def __init__(self, x, r1, r2, v, c, w):
        self.x = x
        self.r1 = r1
        self.r2 = r2
        self.v = v
        self.c = c
        self.w = w
        self.oldX = list(x)
        self.pBest = list(x)
        self.gBest = 0
        self.first_iteration = True

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

    def updateV(self):
        for i in range(len(self.x)):
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * self.r1 * (self.pBest[i] - self.x[i])) + (self.c[1] * self.r2 * (self.gBest - self.x[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] += self.v[i]

    def plot_particles(self, ax):
        ax.scatter(self.x, [f(xi) for xi in self.x], c='b', marker='o', label='Particles')
        ax.scatter(self.gBest, f(self.gBest), c='r', marker='o', s=100, label='Global Best')

    def animate(self, i, ax):
        if self.first_iteration:
            self.first_iteration = False
        else:
            print(f"Iterasi {i+1}")
            print(f"x = {[round(val, 3) for val in self.x]}")
            print(f"f(x) = {[round(f(val), 3) for val in self.x]}")
            self.findPBest()
            self.findGBest()
            self.updateV()
            self.updateX()
            print(f"pBest = {[round(val, 3) for val in self.pBest]}")
            print(f"gBest x = {round(self.gBest, 3)}")
            print("gBest f(x) =", round(f(pso.gBest), 3))
            print(f"v = {[round(val, 3) for val in self.v]}")
            print(f"Update x = {[round(val, 3) for val in self.x]}")
            print(f"Update f(x) = {[round(f(val), 3) for val in self.x]}")
            print()

        ax.clear()
        self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f'Iteration {i+1}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('f(X)')
        ax.legend()

    def plot_surface(self, ax):
        x = np.linspace(-5, 5, 100)
        y = f(x)
        ax.plot(x, y, color='purple', alpha=0.5, label='Objective Function')

    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

    def print_optimization_result(pso):
        print("Hasil Optimasi:")
        print("Nilai Optimal x =", pso.gBest)
        print("Nilai Optimal f(x) =", f(pso.gBest))

print("No1b Particle Swarm Optimization Group 4 PPO B\n")

num_iterations = int(input("Masukkan jumlah iterasi: "))

x = np.random.uniform(0, np.pi, 3)
r1 = np.random.rand()
r2 = np.random.rand()
v = np.zeros(3)
c = [1/2, 1]
w = 1

print("\nNilai Awal:")
print("x =", x)
print("r1 =", r1)
print("r2 =", r2)
print("v =", v)
print("c =", c)
print("w =", w,"\n")

pso = PSO(x, r1, r2, v, c, w)

pso.iterate_with_animation(num_iterations)

pso.print_optimization_result()