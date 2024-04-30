import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from numba import int64, float64, complex128
from numba.experimental import jitclass
from numba import njit


R_weak = 0.000001
N = 256
M = 15000
@njit
def Mandelbrot(M, N, R):
    
    set_c = np.zeros((M,M))
    y = np.linspace(-0.3, -0.1, M)
    x = np.linspace(-0.85, -0.79, M)
    for i in range(0, M):
        for j in range(0, M):
            c = x[i] + 1j*y[j]
            Z = np.zeros((N), dtype = np.complex128)
            for n in range(1, N):
                Z[n] = Z[n-1]**2 + c
                if np.abs(Z[n]) > R:
                    set_c[i,j] = 0
                else:
                    set_c[i,j] = 1
                
                
    return set_c

start = datetime.datetime.now()
print('Время старта: ' + str(start))
nn = Mandelbrot(M, N, R_weak)
finish = datetime.datetime.now()
print('Время окончания: ' + str(finish))
# вычитаем время старта из времени окончания
print('Время работы: ' + str(finish - start))
plt.imshow((nn))


