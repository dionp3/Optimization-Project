# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Definisikan fungsi objektif yang akan dioptimalkan
def f(x):
    return (-2 * x) * (np.sin(x))

# Definisikan kelas PSO (Particle Swarm Optimization)
class PSO:
    def __init__(self, x, v, c, r, w):
        # Inisialisasi parameter PSO
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

    # Fungsi untuk mencari pBest
    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i]) < f(self.pBest[i]):
                self.pBest[i] = self.x[i]
            else:
                self.pBest[i] = self.oldX[i]

    # Fungsi untuk mencari gBest
    def findGBest(self):
        minVal = f(self.x[0])
        minIndex = 0
        for i in range(1, len(self.x)):
            fx = f(self.x[i])
            if fx < minVal:
                minVal = fx
                minIndex = i
        self.gBest = self.x[minIndex]

    # Fungsi untuk mengupdate kecepatan partikel
    def updateV(self):
        for i in range(len(self.x)):
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * self.r[0] * (self.pBest[i] - self.x[i])) + (self.c[1] * self.r[1] * (self.gBest - self.x[i]))

    # Fungsi untuk mengupdate posisi partikel
    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] += self.v[i]

    # Fungsi untuk plot partikel pada grafik 3D
    def plot_particles(self, ax):
        particles = ax.scatter(self.x, [f(xi) for xi in self.x], c='b', marker='o', label='Particles')
        global_best = ax.scatter(self.gBest, f(self.gBest), c='r', marker='o', s=100, label='Global Best')
        return particles, global_best

    # Fungsi untuk plot fungsi objektif pada grafik 3D
    def plot_surface(self, ax):
        x = np.linspace(-5, 5, 100)
        y = f(x)
        ax.plot(x, y, color='purple', alpha=0.5, label='Objective Function')

    # Fungsi untuk iterasi dengan animasi
    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

    # Fungsi untuk print hasil optimasi
    def print_optimization_result(self):
        print("\nHasil Optimasi:")
        print("Nilai Optimal x =", self.gBest)
        print("Nilai Optimal f(x) =", f(self.gBest))

    # Fungsi untuk animasi pada setiap iterasi
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

        # Memeriksa perubahan nilai terbaik terakhir
        current_best = f(self.gBest)
        if abs(current_best - self.prev_best) < self.tolerance:
            self.unchanged_count += 1
        else:
            self.unchanged_count = 0

        self.prev_best = current_best 

        # Jika nilai terbaik tidak berubah dalam 5 iterasi terakhir, hentikan iterasi
        if self.unchanged_count >= 5:
            print("Iterasi dihentikan karena konvergensi telah tercapai.")
            if self.animation is not None:
                self.animation.event_source.stop()
                pso.print_optimization_result()
            return None
        
        # Bersihkan plot sebelum menggambar 3D yang baru
        ax.clear()
        particles, global_best = self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f'Iteration {i+1}') 
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()

        # Jika sudah mencapai batas iterasi, hentikan iterasi dan tampilkan hasil optimasi
        if i >= num_iterations-1:
            print("Batas iterasi telah tercapai")
            if self.animation is not None:
                self.animation.event_source.stop()
                pso.print_optimization_result()
            return None

        # Pada iterasi pertama, tampilkan legenda partikel dan nilai terbaik global
        if i == 0:
            ax.legend(handles=[particles, global_best], loc='upper right')        

# Print judul
print("No1a Particle Swarm Optimization Group 4 PPO B\n")

# Inisialisasi nilai awal
x = [1.0, np.pi/2, np.pi]
v = np.zeros(3)
c = [1/2, 1]
r = [1, 1]
w = 1

# Menampilkan informasi nilai awal
print("f(x) = (-2x)(sin(x))  ;  0<=x<=phi")
print("\nNilai Awal:")
print("x =", x)
print("v =", v)
print("C =", c)
print("R =", r)
print("w =", w,"\n")

# Inisialisasi objek PSO
pso = PSO(x, v, c, r, w)

# Input jumlah iterasi dari pengguna
num_iterations = int(input("Masukkan jumlah iterasi: "))
pso.iterate_with_animation(num_iterations)