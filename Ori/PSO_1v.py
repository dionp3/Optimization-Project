import numpy as np
def f(x):
    return 1/3 * np.sqrt(x**2 + 25)

class PSO():
    def __init__(self, x: list, v: list, c: list, r: list, w:float):
        self.x = x
        self.v = v
        self.c = c
        self.r = r
        self.w = w

        self.oldX = x.copy()
        self.pBest = x.copy()
        self.gBest = None

    def findPBest(self):
        for i in range(len(self.x)):
            if (f(self.x[i]) < f(self.pBest[i])):
                self.pBest[i] = self.x[i]
            else:
                self.pBest[i] = self.oldX[i]
    
    def findGBest(self):
        fValues = []
        for x in self.x:
            fValues.append(f(x))
        self.gBest = self.x[np.argmin(fValues)]

    def updateV(self):
        for i in range(len(self.x)):
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * self.r[0] * (self.pBest[i] - self.x[i])) + (self.c[1] * self.r[1] * (self.gBest - self.x[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.x[i] = self.x[i] + self.v[i]

    def iterate(self, n):
        print(f"Iterasi 0")
        print(f"x = {self.x}")
        print(f"v = {self.v}")
        print()

        for i in range(n):
            print(f"Iterasi", i+1)
            self.findPBest()
            self.findGBest()
            self.updateV()
            self.updateX()
        
            print(f'x = {self.x}')
            print(f'v = {self.v}')
            print(f'pBest = {self.pBest}')
            print(f'gBest = {self.gBest}')
            print(f'f(gBest) = {f(self.gBest)}')
            print(f'f(x) = {[f(x) for x in self.x]}')
            print()

x = np.array([-1.0, 1.5, 2.0])
v = np.array([0.0, 0.0, 0.0])
c = np.array([0.5, 1])
r = np.array([0.5, 0.5])
w = 1

pso = PSO(x, v, c, r, w)
pso.iterate(3)