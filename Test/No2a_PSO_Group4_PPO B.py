# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Definisi fungsi objektif (fungsi f)
def f(x, y):
    return (x + y**2 - 13)**2 + (x**2 + y - 9)**2

# Definisi kelas PSO (Particle Swarm Optimization)
class PSO:
    def __init__(self, x, y, v, c, w):
        # Inisialisasi posisi partikel (x, y), kecepatan partikel (v), koefisien percepatan kognitif dan sosial (c), dan inertia weight (w)
        self.x = x
        self.y = y
        self.vx = v
        self.vy = v.copy()
        self.c = c
        self.w = w
        # Inisialisasi variabel untuk menyimpan posisi partikel sebelumnya, pBest (posisi partikel terbaik), dan gBest (posisi terbaik secara global)
        self.oldX = list(x)
        self.oldY = list(y)
        self.pBest = list(zip(x, y))
        self.gBest = (0, 0)
        # Inisialisasi v di sini
        self.v = [(self.vx[i], self.vy[i]) for i in range(len(self.x))]

        # Flag untuk menandai iterasi pertama
        self.first_iteration = True

    # Fungsi untuk mencari pBest untuk setiap partikel
    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i], self.y[i]) < f(self.pBest[i][0], self.pBest[i][1]):
                self.pBest[i] = (self.x[i], self.y[i])
            else:
                self.pBest[i] = (self.oldX[i], self.oldY[i])

    # Fungsi untuk mencari gBest dari semua partikel
    def findGBest(self):
        minVal = f(self.x[0], self.y[0])
        minIndex = 0

        for i in range(1, len(self.x)):
            fx = f(self.x[i], self.y[i])
            if fx < minVal:
                minVal = fx
                minIndex = i
        self.gBest = (self.x[minIndex], self.y[minIndex])

    # Fungsi untuk mengupdate kecepatan partikel
    def updateV(self, r1, r2):
        for i in range(len(self.x)):
            self.vx[i] = (self.w * self.vx[i]) + (self.c[0] * r1 * (self.pBest[i][0] - self.x[i])) + (self.c[1] * r2 * (self.gBest[0] - self.x[i]))
            self.vy[i] = (self.w * self.vy[i]) + (self.c[0] * r1 * (self.pBest[i][1] - self.y[i])) + (self.c[1] * r2 * (self.gBest[1] - self.y[i]))

        # Gabungkan vx dan vy menjadi satu array v
        self.v = [(self.vx[i], self.vy[i]) for i in range(len(self.x))]


            
    # Fungsi untuk mengupdate posisi partikel
    def updateX(self):
    # Menggabungkan vx dan vy menjadi satu array v
        self.v = [(self.vx[i], self.vy[i]) for i in range(len(self.x))]

        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.oldY[i] = self.y[i]

            # Menggunakan nilai v untuk memperbarui x dan y
            self.x[i] = self.x[i] + self.v[i][0]
            self.y[i] = self.y[i] + self.v[i][1]


    # Fungsi untuk plotting partikel dan animasi
    def plot_particles(self, ax):
        ax.scatter(self.x, self.y, [f(xi, yi) for xi, yi in zip(self.x, self.y)], c='b', marker='o', label='Particles')
        ax.scatter(self.gBest[0], self.gBest[1], f(self.gBest[0], self.gBest[1]), c='r', marker='o', s=100, label='Global Best')

    # Fungsi untuk animasi iterasi
    def animate(self, i, ax):
        if self.first_iteration:
            self.first_iteration = False
        else:
            # Update pBest, gBest, kecepatan, dan posisi untuk iterasi selanjutnya
            self.findPBest()
            self.findGBest()
            r1, r2 = self.r[0], self.r[1]  # Menggunakan nilai r1 dan r2 dari input
            self.updateV(r1, r2)
            self.updateX()

            # Print informasi untuk iterasi selanjutnya
            print(f"Iterasi {i+1}")
            print(f"r1 = {round(r1, 3)}, r2 = {round(r2, 3)}")
            print(f"x sebelum = {[round(val, 3) for val in self.oldX]}")
            print(f"x = {[round(val, 3) for val in self.x]}")
            print(f"y sebelum = {[round(val, 3) for val in self.oldY]}")
            print(f"y = {[round(val, 3) for val in self.y]}")
            print("v =", [tuple(map(lambda x: round(x, 3), val)) for val in self.v])
            print(f"pBest = {[(round(val[0], 3), round(val[1], 3)) for val in self.pBest]}")
            print(f"gBest = {(round(self.gBest[0], 3), round(self.gBest[1], 3))}")
            print(f"f(x, y) sebelum = {[round(f(val[0], val[1]), 3) for val in zip(self.oldX, self.oldY)]}")
            print(f"f(x, y) = {[round(f(val[0], val[1]), 3) for val in zip(self.x, self.y)]}")
            print()

        # Menghapus plot sebelumnya dan memplot partikel untuk iterasi saat ini
        ax.clear()
        self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f'Iteration {i+1}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        ax.legend()

    # Fungsi untuk plotting permukaan fungsi objektif sebagai gunung
    def plot_surface(self, ax):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, label='Objective Function')

    # Fungsi untuk iterasi dengan animasi
    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        animation = FuncAnimation(fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False)
        plt.show()

        # Fungsi untuk menampilkan hasil optimasi setelah semua iterasi
    def print_optimization_result(self):
        print("\nHasil Optimasi:")
        print("Nilai Optimal X:", round(self.gBest[0], 3))
        print("Nilai Optimal Y:", round(self.gBest[1], 3))
        print("Nilai Optimal f(X, Y):", round(f(self.gBest[0], self.gBest[1]), 3))


# Menampilkan judul
print("Particle Swarm Optimization Group 4 PPO B\n")

# Input jumlah iterasi dari pengguna
num_iterations = int(input("Masukkan jumlah iterasi: "))

# Inisialisasi nilai awal X dan Y sesuai dengan permintaan
x = [1.0, -1.0, 2.0]
y = [1.0, -1.0, 1.0]

# Inisialisasi nilai awal r1 dan r2 sesuai dengan permintaan
r1 = [1.0, 1.0]
r2 = [1.0, 1.0]

# Inisialisasi nilai awal V sebagai 3 bilangan 0, koefisien c, dan inertia weight w sesuai dengan permintaan
v = np.zeros(3)
c = [1.0, 0.5]
w = 1.0

# Membuat objek PSO
pso = PSO(x, y, v, c, w)
pso.r = r1

# Melakukan iterasi dengan animasi sebanyak num_iterations iterasi
pso.iterate_with_animation(num_iterations)

# Menampilkan hasil optimasi setelah semua iterasi
pso.print_optimization_result()