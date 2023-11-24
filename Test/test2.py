import numpy as np

def f(x, y):
    return (x + y**2 - 13)**2 + (x**2 + y - 9)**2

class PSO:
    def __init__(self, x_vals, y_vals, v_vals, c_vals, r_vals, w_val):
        self.x = np.array(x_vals)
        self.y = np.array(y_vals)
        self.vx = np.array(v_vals)
        self.vy = np.array(v_vals)
        self.c = np.array(c_vals)
        self.r = np.array(r_vals)
        self.w = w_val

        self.oldX = self.x.copy()
        self.oldY = self.y.copy()
        self.pBestX = self.x.copy()
        self.pBestY = self.y.copy()
        self.gBestX = self.x[np.argmin(self.x)]
        self.gBestY = self.y[np.argmin(self.y)]

    def find_pBest(self):
        for i in range(len(self.x)):
            value = f(self.x[i], self.y[i])
            pBestValue = f(self.pBestX[i], self.pBestY[i])
            if value < pBestValue:
                self.pBestX[i] = self.x[i]
                self.pBestY[i] = self.y[i]
            else:
                self.pBestX[i] = self.oldX[i]
                self.pBestY[i] = self.oldY[i]

    def find_GBest(self):
        f_values = [f(xi, yi) for xi, yi in zip(self.x, self.y)]
        minimum_index = np.argmin(f_values)

        self.gBestX = self.x[minimum_index]
        self.gBestY = self.y[minimum_index]

    def update_V(self):
        for i in range(len(self.x)):
            self.vx[i] = (self.w * self.vx[i]) + (self.c[0] * self.r[0] * (self.pBestX[i] - self.x[i])) + (self.c[1] * self.r[1] * (self.gBestX - self.x[i]))
            self.vy[i] = (self.w * self.vy[i]) + (self.c[0] * self.r[0] * (self.pBestY[i] - self.y[i])) + (self.c[1] * self.r[1] * (self.gBestY - self.y[i]))

    def update_X(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.oldY[i] = self.y[i]
            self.x[i] += self.vx[i]
            self.y[i] += self.vy[i]

    def iterate(self, n):
        print("Iterasi ke-0")
        print("x =", self.x)
        print("y =", self.y)
        print("vx =", self.vx)
        print("vy =", self.vy)
        print("pBestX =", self.pBestX)
        print("gBestX =", self.gBestX)
        print("pBestY =", self.pBestY)
        print("gBestY =", self.gBestY)
        print("f(gBestX, gBestY) =", f(self.gBestX, self.gBestY))
        print("f(x,y) =", [f(xi, yi) for xi, yi in zip(self.x, self.y)])
        print()

        for i in range(n):
            print(f"Iterasi ke-{i + 1}")
            self.find_pBest()
            self.find_GBest()
            self.update_V()
            self.update_X()

            print("x =", self.x)
            print("y =", self.y)
            print("vx =", self.vx)
            print("vy =", self.vy)
            print("pBestX =", self.pBestX)
            print("gBestX =", self.gBestX)
            print("pBestY =", self.pBestY)
            print("gBestY =", self.gBestY)
            print("f(gBestX, gBestY) =", f(self.gBestX, self.gBestY))
            print("f(x,y) =", [f(xi, yi) for xi, yi in zip(self.x, self.y)])
            print()

# Main function
x = [1.0, -1.0, 2.0]
y = [1.0, -1.0, 1.0]
vx = [0, 0, 0]
vy = [0, 0, 0]
c = [1.0, 0.5]
r = [1.0, 1.0]
w = 1.0

pso = PSO(x, y, vx, c, r, w)
pso.iterate(50)
