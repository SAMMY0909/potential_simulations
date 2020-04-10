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

#To set the line function   
def line_ch(v,a,divider,n):
    for j in range(int(divider*(-0.25*a+0.5)),int(divider*(0.25*a+0.5))+1):
                       v[int((0.1*a+0.5)*divider)][j]=-1.0
    for j in range(int(divider*(-0.25*a+0.5)),int(divider*(0.25*a+0.5))+1):
                       v[int((-0.1*a+0.5)*divider)][j]=1.0

#for efield , two components
def efield(ex,ey,v,n):
    for i in range(1,n):
        for j in range(1,n):
            ey[i][j]=0.5*(v[i-1][j]-v[i+1][j]) #field along y
            ex[i][j]=0.5*(v[i][j-1]-v[i][j+1]) #field along x

'''
def efield(ex,ey,v):
    [ex,ey]=np.gradient(v) #np.gradient has similar function as our difference method for gradient, it's faster

#since we want faster computing time, I defined the function this way
'''

def egden(u,ex,ey,n):
    for i in range(0,n):
        for j in range(0,n):
            u[i][j]=(ex[i][j])**2+(ey[i][j])**2


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
#----------------------------------------RunInitialisation---------------------------------------------------------#
#initialise all the 2-D arrays to uniform values, this will change later on
v  = np.zeros((n+1,n+1))
ex = np.zeros((n+1,n+1))
ey = np.zeros((n+1,n+1))
u  = np.zeros((n+1,n+1))


#initialise line charges
line_ch(v,a,divider,n)

#initialise efield fn
efield(ex,ey,v,n)

#initialise energy
egden(u,ex,ey,n)

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
#This part must be modified because line_ch part IS NOT UPDATED
#n-1 because range function runs up to n-2 and we don't update the boundaries because they must stay put at given #Dirichlet condition
for i in range(1,n-1):
    for j in range(1,n-1):
        upfunc(v, i, j)
        line_ch(v,a,divider,n) #this has to be run in every iteration to reset the line_ch to V=+/-1.0
        #this is because the updation changes all mesh points, so we have to undo the line_ch part

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
            line_ch(v,a,divider,n)
    cf.append(functional(v,n))


plt.subplot(133)
color_map = plt.imshow(v)
color_map.set_cmap(cmap)
plt.colorbar()
plt.title('1000 iterations',fontname='Comic Sans MS', fontsize=10)

plt.savefig("Potential_runs_q4.pdf", dpi=600)

#---------------------------------------------OTHER PLOTS---------------------------------------------------------#
#---------------------------------------Relaxation Iterative Convergence------------------------------------------#

plt.figure(3)
plt.title('Convergence of the relaxation')
plt.plot(cf, marker='o',color='mediumvioletred', markersize=6)
plt.ylabel('Convergence Sum')
plt.xlabel('Number of iters')
plt.savefig("Convergence_q4.pdf", dpi=600)

#------------------------------------------Potential along selected axis-------------------------------------------#

x = np.arange(-0.5, 0.5+h, h)
y1 = []
y2 = []
for i in range(0,n):
    y1.append(v[50][i])
    y2.append(v[i][50])

plt.figure(4)
plt.title('Potential from iterative solution')
plt.plot(x,y1, marker='o',color='green', markersize=6, label='y=0')
plt.plot(x,y2, marker='o',color='mediumvioletred', markersize=6, label='x=0')
plt.legend()
plt.ylabel(r'$\Phi (x@y=0-OR-y@x=0)$')
plt.savefig("Potential_selected_q4.pdf", dpi=600)

#------------------------------------------Field along selected axis-------------------------------------------#

x1 = np.arange(-0.5, 0.5+h, h)
y3 = []
y4 = [] 
for i in range(0,n):
    y3.append(ey[50][i])
    y4.append(ey[i][50])
    
plt.figure(5)
plt.title('Ey field along y=0/x=0 line')
plt.ylabel('Ey field along y=0/x=0 line')
plt.plot(x1,y3, marker='o',color='green', markersize=4, label='Ey along y=0')
#plt.plot(x1,y4, marker='o',color='maroon', markersize=4,label='Ey along x=0')
plt.legend()
plt.savefig("Field_selected_q4.pdf", dpi=600)

#--------------------------------------------Energy Densty Plot-------------------------------------------------#
line_ch(v,a,divider,n) # has to be updated whenever you run
efield(ex,ey,v,n)
egden(u,ex,ey,n)

plt.figure(6)
color_map = plt.imshow(u)
color_map.set_cmap(cmap)
plt.colorbar()
plt.title('1000 iterations Electric Field Energy Density',fontname='Comic Sans MS', fontsize=10)
plt.savefig("EnergyDensityPlot_q4.pdf", dpi=600)

#-------------------------Potential and Field Plot simultaneously-------------------------------------------------#
line_ch(v,a,divider,n)
efield(ex,ey,v,n) 

nx=0.5;ny=0.5

nrows,ncols=ex.shape
X=np.linspace(-nx,nx,nrows)
Y=np.linspace(-ny,ny,ncols)
Xi, Yi = np.meshgrid(X, Y,indexing='ij')

plt.figure(7)
color_map = plt.imshow(v,extent=[-0.5,0.5,-0.5,0.5])
color_map.set_cmap(cmap)
plt.quiver(Xi, Yi, ex, ey)
plt.colorbar()
plt.title('1000 iterations Potential(CMAP) + Electric Field (Quiver)',fontname='Comic Sans MS', fontsize=10)

plt.savefig("Potential_Efield_combined_q4.pdf", dpi=600)
