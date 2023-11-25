import numpy as np

def f(x, y):
    return (x + y**2 - 13)**2 + (x**2 + y - 9)**2

class PSO():
    def __init__(self, x: list, y: list,v: list, c: list, r: list, w:float):
        self.x = x
        self.y = y
        self.vx = v
        self.vy = v.copy()
        self.c = c
        self.r = r
        self.w = w

        self.oldX = x.copy()
        self.oldY = y.copy()
        self.pBestX = x.copy()
        self.pBestY = y.copy()
        self.gBestX = self.x[np.argmin([f(x, y) for x, y in zip(self.x, self.y)])]
        self.gBestY = self.y[np.argmin([f(x, y) for x, y in zip(self.x, self.y)])]

    def findPBest(self):
        for i in range(len(self.x)):
            value = f(self.x[i], self.y[i])
            pBestFValue = f(self.pBestX[i], self.pBestY[i])
            if (value < pBestFValue):
                self.pBestX[i] = self.x[i]
                self.pBestY[i] = self.y[i]
            else:
                self.pBestX[i] = self.oldX[i]
                self.pBestY[i] = self.oldY[i]
    
    def findGBest(self):
        fValues = []
        for x, y in zip(self.x, self.y):
            fValues.append(f(x, y))

        minimumIndex = np.argmin(fValues)
        self.gBestX = self.x[minimumIndex]
        self.gBestY = self.y[minimumIndex]

    def updateV(self):
        for i in range(len(self.x)):
            self.vx[i] = (self.w * self.vx[i]) + (self.c[0] * self.r[0] * (self.pBestX[i] - self.x[i])) + (self.c[1] * self.r[1] * (self.gBestX - self.x[i]))
            self.vy[i] = (self.w * self.vy[i]) + (self.c[0] * self.r[0] * (self.pBestY[i] - self.y[i])) + (self.c[1] * self.r[1] * (self.gBestY - self.y[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.oldX[i] = self.x[i]
            self.oldY[i] = self.y[i]
            self.x[i] = self.x[i] + self.vx[i]
            self.y[i] = self.y[i] + self.vy[i]

    def iterate(self, n):
        print(f"Iterasi 0")
        print(f"x = {self.x}")
        print(f"y = {self.y}")
        print(f"vx = {self.vx}")
        print(f"vy = {self.vy}")
        print(f'pBest = {self.pBestX}')
        print(f'gBest = {self.gBestX}')
        print(f'pBest = {self.pBestY}')
        print(f'gBest = {self.gBestY}')
        print(f'f(gBest x, gBest y) = {f(self.gBestX, self.gBestY)}')
        print()

        for i in range(n):
            print(f"Iterasi", i+1)
            self.findPBest()
            self.findGBest()
            self.updateV()
            self.updateX()
            print(f'x = {self.x}')
            print(f'y = {self.y}')
            print(f'vx = {self.vx}')
            print(f'vy = {self.vy}')
            print(f'pBest = {self.pBestX}')
            print(f'gBest = {self.gBestX}')
            print(f'pBest = {self.pBestY}')
            print(f'gBest = {self.gBestY}')
            print(f'f(gBest x, gBest y) = {f(self.gBestX, self.gBestY)}')
            print(f'f(x, y) = {[f(x, y) for x, y in zip(self.x, self.y)]}')
            print()

x = np.array([1, -1, 2])
y = np.array([1, -1, 1])
v = np.array([0, 0, 0])
c = np.array([1, 1/2])
r = np.array([1, 1])
w = 1

pso = PSO(x, y, v, c, r, w)
pso.iterate(50)