import numpy as np

def f(x):
    return -2 * x * np.sin(x)

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

    def iterate(self, n):
        print("Iterasi ke-0")
        print("x =", self.x)
        print("v =", self.v)
        print("pBest =", self.pBest)
        print("gBest =", self.gBest)
        print("f(x) =", [f(val) for val in self.x], "\n")

        for i in range(n):
            print(f"Iterasi ke-{i + 1}")
            self.findPBest()
            self.findGBest()
            self.updateV()
            self.updateX()

            print("x sebelum =", self.oldX)
            print("x sesudah =", self.x)
            print("v =", self.v)
            print("pBest =", self.pBest)
            print("gBest =", self.gBest)
            print("f(x) sebelum =", [f(val) for val in self.oldX])
            print("f(x) sesudah =", [f(val) for val in self.x], "\n")

x = [1.0, np.pi/2, np.pi]
v = [0, 0, 0]
c = [1/2, 1]
r = [1, 1]
w = 1

pso = PSO(x, v, c, r, w)
pso.iterate(3)
