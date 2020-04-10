#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite difference iterative relaxation method
Created on Mon Mar 23 09:48:39 2020

@author: mario borunda
"""
def updateFunction (phi,i,j):
    phi[i][j] = 0.25*(phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1])
    
def sFunction (phi,N):
    s=0.0
    for i in range(0,N-1):
        for j in range(0,N-1):
            s = s + ((phi[i+1][j] - phi[i][j])**2 + (phi[i][j+1] - phi[i][j])**2)
    return(s)
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

cmap = cm.PRGn

a = 1.0
eps = a/10.0
N = 11
print (a,eps,N)

phi = [[0.5 for x in range(N)] for y in range(N)] 
"""
We initialized the phi 2D array to 1/2
"""
for i in range(N):
    phi[i][0] = 0.0
    phi[i][N-1] = 0.0
    phi[0][i] = 1.0
    phi[N-1][i] = 1.0
    

"""
We initialized the corners of the array to 1/2
where we assumed they take the average of the two plates and then
the boundaries (top,bottom) = 1 (left,right) = 0
the next lines plot phi
"""
plt.figure(1)

plt.subplot(211)
color_map = plt.imshow(phi)
color_map.set_cmap(cmap)
plt.title('Initialization')

s1 = []
s1.append(sFunction(phi,N))

phiPoint = []
phiPoint.append(phi[7][5])
"""
We want to store the value of the sum in the s1 array
We also want to store the value of phi at one lattice point
The next line prints those two values
"""
print(s1,phiPoint)

for i in range(1,N-1):
    for j in range(1,N-1):
        updateFunction(phi, i, j)

s1.append(sFunction(phi,N))
phiPoint.append(phi[7][5])
print(s1,phiPoint)
   
plt.subplot(212)
color_map = plt.imshow(phi)
color_map.set_cmap(cmap)
plt.title('One iteration')

for counter in range(2, 41):
    for i in range(1,N-1):
        for j in range(1,N-1):
            updateFunction(phi, i, j)
    s1.append(sFunction(phi,N))
    phiPoint.append(phi[7][5])

plt.figure(2)
color_map = plt.imshow(phi)
color_map.set_cmap(cmap)
plt.title('100 iterations')
plt.colorbar()

plt.figure(3)
plt.title('Convergence of the relaxation')
plt.plot(s1, marker='o')
plt.ylabel('S')
for i in range(0,41):
    print(i,s1[i])

plt.figure(4)
plt.title('Potential at selected point')
plt.plot(phiPoint, marker='o')
plt.ylabel(r'$\phi (0.0,0.2)$')  #phi[5][7]

x = np.arange(-0.5, 0.5+eps, eps)
y1 = []
y2 = []
for i in range(0,N):
    y1.append(phi[5][i])
    y2.append(phi[i][5])
print(x,y1)
plt.figure(5)
plt.title('Potential from iterative solution')
plt.plot(x,y1, marker='o', label='y=0')
plt.plot(x,y2, marker='o', label='x=0')
plt.legend()
plt.ylabel(r'$\phi (0.0,0.2)$')  #phi[5][7]

