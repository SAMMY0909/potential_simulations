#regular imports

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import cm   #cm stands for colour map


# _define potential function
#this function should do recursive updates to potential in the mesh grid
#i,j are running grid labels, n is the total number of either i or j pts in the square mesh grid
#sum over nearest neighbours
def upfunc(v,i,j):
    v[i][j] = 0.25 * (v[i + 1][j] + v[i - 1][j] + v[i][j + 1] + v[i][j - 1])

#Crimp function to set the crimp to V=1    
def crimp(v,a,divider,n):
    for i in range(0,int(divider-divider*(0.3*a+0.5))+1):
        for j in range(int(divider*(0.2*a+0.5)),n):
                       v[i][j]=1.0

#we also define the functional sum, to indeed see it is minimised after we plot it
def functional(v,n):
    sum=0.0
    for i in range(0,n-1):
        for j in range(0,n-1):
            sum+= ((v[i+1][j] - v[i][j])**2 + (v[i][j+1] - v[i][j])**2)
    return sum



#specify colour map
cmap=cm.jet

#initialise physical params
a=1.0
divider=100 #this is the step division into as many intervals
h=a/divider #this is the step size of each interval
n=divider+1 #number of points along one linear direction of the meshgrid, since it's a square, it's the same
#cmap=cm.cool #not too cool to be a colourmap
#-------------------------------------------Run Initialisation---------------------------------------------------------#
#initialise the potential to uniform values, this will change later on
v = [[0.5 for x in range(n)] for y in range(n)]

#initialise boundaries
for i in range(n):
    v[i][0] = 0.0; v[i][n-1] = 0.0;v[0][i] = 1.0;v[n-1][i] = 1.0

#initialise corners
v[0][0] = 0.5;v[0][n-1] = 1;v[n-1][0] = 0.5;v[n-1][n-1] = 0.5 #here we can keep 3 corners at 0.5, but the rightmost corner should be at 1 because of crimp condition.

#initialise crimp
crimp(v,a,divider,n)

#plot initialisation pic
plt.figure(1)

plt.subplot(131)
color_map = plt.imshow(v)
color_map.set_cmap(cmap)
plt.title('Init Pot Map',fontname='Comic Sans MS', fontsize=10)

#cf is the cumulative functional 1-D array used for storing those values
cf=[]
cf.append(functional(v,n))

#-------------------------------------------RUN FIRST ITER---------------------------------------------------------#
#This part must be modified because crimp part IS NOT UPDATED
#So update rectangle minus the crimp in the crimp line
#n-1 because range function runs up to n-2 and we don't update the boundaries because they must stay put at given #Dirichlet condition
for i in range(1,n-1):
    for j in range(1,n-1):
        upfunc(v, i, j)
        crimp(v,a,divider,n) #this has to be run in every iteration to reset the crimp to V=1.0
        #this is because the updation changes all mesh points, so we have to undo the crimp part
        #it is like a boundary condition, so can't change

cf.append(functional(v,n))

#plot 1st iter results
plt.subplot(132)
color_map = plt.imshow(v)
color_map.set_cmap(cmap)
plt.title('1st Iter Pot Map',fontname='Comic Sans MS', fontsize=10)

#-------------------------------------------RUN ALL ITER---------------------------------------------------------#
for token in range(2, 1001):
    for i in range(1,n-1):
        for j in range(1,n-1):
            upfunc(v, i, j)
            crimp(v,a,divider,n)
    cf.append(functional(v,n))

plt.subplot(133)
color_map = plt.imshow(v)
color_map.set_cmap(cmap)
plt.colorbar()
plt.title('1000 iterations',fontname='Comic Sans MS', fontsize=10)

plt.savefig("Potential_runs_q3.pdf", dpi=600)
#---------------------------------------------OTHER PLOTS---------------------------------------------------------#
#---------------------------------------Relaxation Iterative Convergence------------------------------------------#
plt.figure(3)
plt.title('Convergence of the relaxation')
plt.plot(cf,  marker='o',color='mediumvioletred',markersize=4)
plt.ylabel('Convergence Sum')
plt.xlabel('Number of iters')
for i in range(0,200):
    print(i,cf[i])
plt.savefig("Convergence_q3.pdf", dpi=600)

#--------------------------------------------Potential along selected axis-------------------------------------------#
x = np.arange(-0.5, 0.5+h, h)
y1 = []
y2 = []
for i in range(0,n):
    y1.append(v[50][i])
    y2.append(v[i][50])

plt.figure(4)
plt.title('Potential from iterative solution')
plt.plot(x,y1, marker='o',color='green', markersize=4, label='y=0')
plt.plot(x,y2, marker='o',color='mediumvioletred', markersize=4, label='x=0')
plt.legend()
plt.ylabel(r'$\Phi (x@y=0-OR-y@x=0)$')
plt.savefig("Potential_selected_q3.pdf", dpi=600)