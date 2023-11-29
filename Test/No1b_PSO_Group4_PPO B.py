import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return (-2 * x) * (np.sin(x))

class PSO:
    def __init__(self, x, v, c, r, w):
        self.x = x
        self.v = v
        self.c = c
        self.r = r
        self.w = w
        self.oldX = list(x)
        self.pBest = list(x)
        self.gBest = 0
        self.tolerance = 1e-5
        self.animation = None        
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
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * self.r[0] * (self.pBest[i] - self.x[i])) + (self.c[1] * self.r[1] * (self.gBest - self.x[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] += self.v[i]

    def plot_particles(self, ax):
        particles = ax.scatter(self.x, [f(xi) for xi in self.x], c='b', marker='o', label='Particles')
        global_best = ax.scatter(self.gBest, f(self.gBest), c='r', marker='o', s=100, label='Global Best')
        return particles, global_best

    def plot_surface(self, ax):
        x = np.linspace(-5, 5, 100)
        y = f(x)
        ax.plot(x, y, color='purple', alpha=0.5, label='Objective Function')

    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

    def print_optimization_result(self):
        print("\nHasil Optimasi:")
        print("Nilai Optimal x =", self.gBest)
        print("Nilai Optimal f(x) =", f(self.gBest))

    def animate(self, i, ax):
        if self.first_iteration:
            self.prev_best = f(self.gBest)
            self.unchanged_count = 0
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
            print(f"gBest f(x) = {round(f(self.gBest), 3)}")
            print(f"v = {[round(val, 3) for val in self.v]}")
            print(f"Update x = {[round(val, 3) for val in self.x]}")
            print(f"Update f(x) = {[round(f(val), 3) for val in self.x]}")
            print()

        current_best = f(self.gBest)
        if abs(current_best - self.prev_best) < self.tolerance:
            self.unchanged_count += 1
        else:
            self.unchanged_count = 0

        self.prev_best = current_best 

        if self.unchanged_count >= 5:
            print("Iterasi dihentikan karena konvergensi telah tercapai.")
            if self.animation is not None:
                self.animation.event_source.stop()
                pso.print_optimization_result()
            return None

        ax.clear()
        particles, global_best = self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f'Iteration {i+1}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()

        if i >= num_iterations-1:
            print("Batas iterasi telah tercapai")
            if self.animation is not None:
                self.animation.event_source.stop()
                pso.print_optimization_result()
            return None

        if i == 0:
            ax.legend(handles=[particles, global_best], loc='upper right')       

print("No1b Particle Swarm Optimization Group 4 PPO B\n")

x = np.random.uniform(0, np.pi, 3)
v = np.zeros(3)
c = [1/2, 1]
r = [np.random.rand(), np.random.rand()]
w = 1

print("f(x) = (-2 * x) * (np.sin(x))")
print("\nNilai Awal:")
print("x =", x)
print("v =", v)
print("C =", c)
print("R =", r)
print("w =", w,"\n")

pso = PSO(x, v, c, r, w)

num_iterations = int(input("Masukkan jumlah iterasi: "))
pso.iterate_with_animation(num_iterations)