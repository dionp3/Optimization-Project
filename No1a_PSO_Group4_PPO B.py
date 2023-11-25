# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Definisi fungsi objektif (fungsi f)
def f(x):
    return (-2 * x) * (np.sin(x))

# Definisi kelas PSO (Particle Swarm Optimization)
class PSO:
    def __init__(self, x, v, c, w):
        # Inisialisasi posisi partikel (x), kecepatan partikel (v), koefisien percepatan kognitif dan sosial (c), dan inertia weight (w)
        self.x = x
        self.v = v
        self.c = c
        self.w = w

        # Inisialisasi variabel untuk menyimpan posisi partikel sebelumnya, pBest (posisi partikel terbaik), dan gBest (posisi terbaik secara global)
        self.oldX = list(x)
        self.pBest = list(x)
        self.gBest = 0

        # Flag untuk menandai iterasi pertama
        self.first_iteration = True

    # Fungsi untuk mencari pBest untuk setiap partikel
    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i]) < f(self.pBest[i]):
                self.pBest[i] = self.x[i]
            else:
                self.pBest[i] = self.oldX[i]

    # Fungsi untuk mencari gBest dari semua partikel
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
    def updateV(self, r1, r2):
        for i in range(len(self.x)):
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * r1 * (self.pBest[i] - self.x[i])) + (self.c[1] * r2 * (self.gBest - self.x[i]))

    # Fungsi untuk mengupdate posisi partikel
    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] += self.v[i]

    # Fungsi untuk plotting partikel dan animasi
    def plot_particles(self, ax):
        ax.scatter(self.x, [f(xi) for xi in self.x], c='b', marker='o', label='Particles')
        ax.scatter(self.gBest, f(self.gBest), c='r', marker='o', s=100, label='Global Best')

    # Fungsi untuk animasi iterasi
    def animate(self, i, ax):
        if self.first_iteration:
            # Print informasi untuk iterasi pertama
            print("Iterasi 0")
            print(f"r1 = {round(self.v[0], 3)}, r2 = {round(self.v[1], 3)}")
            print("x sebelum =", [round(val, 3) for val in self.oldX])
            print("x =", [round(val, 3) for val in self.x])
            print("v =", [round(val, 3) for val in self.v])
            print("pBest =", [round(val, 3) for val in self.pBest])
            print("gBest =", round(self.gBest, 3))
            print("f(x) sebelum =", [round(f(val), 3) for val in self.oldX])
            print("f(x) =", [round(f(val), 3) for val in self.x], "\n")
            self.first_iteration = False
        else:
            # Update pBest, gBest, kecepatan, dan posisi untuk iterasi selanjutnya
            self.findPBest()
            self.findGBest()
            r1 = 1  # Set r1 dan r2 menjadi 1
            r2 = 1
            self.updateV(r1, r2)
            self.updateX()

            # Print informasi untuk iterasi selanjutnya
            print(f"Iterasi {i+1}")
            print(f"r1 = {round(r1, 3)}, r2 = {round(r2, 3)}")
            print(f"x sebelum = {[round(val, 3) for val in self.oldX]}")
            print(f"x = {[round(val, 3) for val in self.x]}")
            print(f"v = {[round(val, 3) for val in self.v]}")
            print(f"pBest = {[round(val, 3) for val in self.pBest]}")
            print(f"gBest = {round(self.gBest, 3)}")
            print(f"f(x) sebelum = {[round(f(val), 3) for val in self.oldX]}")
            print(f"f(x) = {[round(f(val), 3) for val in self.x]}")
            print()

        # Menghapus plot sebelumnya dan memplot partikel serta gunung untuk iterasi saat ini
        ax.clear()
        self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f'Iteration {i}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('f(X)')
        ax.legend()

    # Fungsi untuk plotting permukaan fungsi objektif sebagai gunung
    def plot_surface(self, ax):
        x = np.linspace(-5, 5, 100)
        y = f(x)
        ax.plot(x, y, color='purple', alpha=0.5, label='Objective Function')

    # Fungsi untuk iterasi dengan animasi
    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

# Menampilkan judul
print("Particle Swarm Optimization Group 4 PPO B\n")

# Input jumlah iterasi dari pengguna
num_iterations = int(input("Masukkan jumlah iterasi: "))

# Inisialisasi nilai awal X0 sebagai [1.0, np.pi/2, np.pi]
initial_x = [1.0, np.pi/2, np.pi]
initial_x_rounded = [round(val, 3) for val in initial_x]
print("\nNilai awal X:", initial_x_rounded, "\n")

# Inisialisasi nilai awal r1 dan r2 sebagai bilangan acak dengan interval dari 0 sampai 1
r1 = np.random.rand()
r2 = np.random.rand()

# Inisialisasi nilai awal V sebagai 10 bilangan 0, koefisien c, dan inertia weight w
v = np.zeros(3)  # Ubah menjadi 3 karena hanya 3 partikel
c = [1/2, 1]
w = 1

# Membuat objek PSO
pso = PSO(initial_x, v, c, w)

# Melakukan iterasi dengan animasi sebanyak num_iterations iterasi
pso.iterate_with_animation(num_iterations)

# Menampilkan hasil optimasi setelah semua iterasi
print("Hasil Optimasi:")
print("Nilai Optimal X:", pso.gBest)
print("Nilai Optimal f(X):", f(pso.gBest))