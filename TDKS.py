# This program solves the 1D Kohn-Sham equation for two electrons in a
# harmonic-oscillator potential. It first calculates the self-consistent
# ground state, and then does a time propagation, assuming a short "kick".
#--------------------------------------------------------------------------
#
#   HERE ARE THE PARAMETERS FOR THE GROUND-STATE CALCULATION:
#
#--------------------------------------------------------------------------
NGRID = 101    # number of grid points (always an odd number)
XMAX = 5.      # the numerical grid goes from -XMAX < x < XMAX
KSPRING = 1.   # spring constant of the harmonic oscillator potential
A = 0.5        # Coulomb softening parameter
TOL = 1.e-10   # numerical tolerance (the convergence criterion)
MIX = 0.75     # mixing parameter for the self-consistency iterations
#--------------------------------------------------------------------------
#
#   HERE ARE THE PARAMETERS FOR THE TIME_DEPENDENT CALCULATION:
#
#--------------------------------------------------------------------------
TMAX = 10.     # total propagation time, 0 < T < TMAX
DT = 0.1       # time step
EFIELD = 0.1   # strength of the electric field kick
TKICK = 0.5    # duration of the kick
DYN = 1        # If DYN=1, use time-dependent Hartree and exchange
#              # If DYN=0, use ground-state Hartree and exchange
#--------------------------------------------------------------------------
#
DX = 2.*XMAX/(NGRID-1)  # grid spacing
PI = 3.141592653589793  # define pi here
IONE = complex(0., 1.)
#
# Define the numerical grid as an array
#
import matplotlib.pyplot as plt # Plotting library
import numpy as np
x = np.linspace(-XMAX,XMAX,NGRID)
# 
# Initialize the density of the noninteracting 1D harmonic oscillator.
#
import math
def h(i):
    return 2.*math.exp(-i**2)/math.sqrt(PI)
#
# initialize a bunch of arrays
#
n = np.zeros(NGRID)
n1 = np.zeros(NGRID)
psi = np.zeros(NGRID)
PHI = np.zeros(NGRID, dtype=complex)
VH = np.zeros(NGRID) 
VX = np.zeros(NGRID)
vint = np.zeros(NGRID)
HMAT = np.zeros((NGRID,NGRID))
TMAT = np.zeros((NGRID,NGRID), dtype=complex)
HKIN = np.zeros((NGRID,NGRID))
ONE = np.zeros((NGRID,NGRID))
VEXT = np.zeros(NGRID)
VT = np.zeros(NGRID)
RHS = np.zeros(NGRID, dtype=complex)
#
for i in range(NGRID):
    n[i]=h(x[i])
n1 = n.copy()
#
from pylab import *
#
# Define the quadratic function of the spring, and use this to 
# define the external potential Hamiltonian:
#
def g(i):
    return 0.5*KSPRING*i**2
for i in range(NGRID):
    VEXT[i]=g(x[i]) 
    ONE[i,i] = 1.
#
#  define the kinetic energy operator:
#    
for i in range(NGRID):
    HKIN[i,i] = 490./(360.*DX**2)
for i in range(NGRID-1):
    HKIN[i,i+1] = -270./(360.*DX**2)
    HKIN[i+1,i] = -270./(360.*DX**2)
for i in range(NGRID-2):
    HKIN[i,i+2] = 27./(360.*DX**2)
    HKIN[i+2,i] = 27./(360.*DX**2)
for i in range(NGRID-3):
    HKIN[i,i+3] = -2./(360.*DX**2)
    HKIN[i+3,i] = -2./(360.*DX**2)   
#
# We will need numerical integration routines.
#
from scipy.integrate import simps   # integration with Simpson's rule
from scipy.integrate import quad    # integration with Gaussian quadrature
from scipy.special import kn        # The modified Bessel functions
#------------------------------------------------------------------------
# Here is the start of the self-consistency loop. We initialize the
# energy as that of two noninteacting electrons in a 1D harmonic potential
#------------------------------------------------------------------------
crit = 1.
EKS_previous = math.sqrt(KSPRING) 
counter = 0
while crit > TOL:
    counter += 1
    print(counter)
    for i in range(NGRID):
        for j in range(NGRID):  
            HMAT[i,j] = HKIN[i,j]
#
#  mix with the density of the previous iteration
#
        n = MIX*n + (1.-MIX)*n1
        n1 = n.copy()
#
# Calculate the Hartree potential VH:
#   
    for i in range(NGRID):
        for j in range(NGRID):
            vint[j]=n[j]/math.sqrt(A**2 + (x[i]-x[j])**2)
        result = simps(vint,x)
        VH[i] = result
        HMAT[i,i] = HMAT[i,i] + VEXT[i] + VH[i]   
#
# Calculate the exchange potential VX:
#    
    for i in range(NGRID):
        upper = n[i]*PI*A  
        def f(j):
            return kn(0,j)
        result, _ = quad(f,0.,upper)
        VX[i] = -result/(PI*A)
        HMAT[i,i] = HMAT[i,i] +VX[i]    
#
# Now find the eigenvalues and eigenvectors of the matrix MAT,
# and sort them to make sure we keep the lowest one only.
#
    vals, vecs = np.linalg.eig(HMAT)
    sorted = np.argsort(vals)
    lowest = sorted[0]
    EKS = vals[lowest]
    for i in range(NGRID):
        psi[i]=vecs[i,lowest]   
#
# Now we need to normalize our solution. 
#
    norm = simps(psi**2,x)
    psi = psi/math.sqrt(norm)
    n = 2*psi**2  
    n0 = n
    PHI = IONE*psi
#
#  We define the convergence criterion (crit) as the difference between the 
#  Kohn-Sham lowest energy eigenvalue in this iteration step to the one of  
#  the previous step. The criterion needs to be < TOL for the 
#  iteration to end.
#
    crit = abs(EKS_previous - EKS)
    EKS_previous = EKS
# --------------------------------------------------------------------------
# End of the self-consistency loop
# --------------------------------------------------------------------------
print('converged')
#
print(' lowest four Kohn-Sham eigenvalues:')
print(EKS,vals[sorted[1]],vals[sorted[2]],vals[sorted[3]])
#
# --------------------------------------------------------------------------
# Now that we have calculated the ground state, we can begin with the
# time propagation. We take psi as the initial wave function and 
# propagate it forward in time.
# --------------------------------------------------------------------------
#
f = open("dip.txt", "w")
T=0
while T < TMAX:  
    T = T + DT
    print('time =',T)
    
#   First, define the time-dependent perturbation VT. We assume that it
#   is a short "kick" in the form of a uniform electric field, which lasts
#   for 5 time steps.
#
    if T <= TKICK:
        VT = EFIELD*x
    else:
        VT = np.zeros(NGRID)
           
    HMAT = HKIN.copy()
   
#   If DYN=1 calculate the time-dependent Hartree and exchange potential

    if DYN == 1:
        for i in range(NGRID):
            for j in range(NGRID):
                vint[j]=n[j]/math.sqrt(A**2 + (x[i]-x[j])**2)
            result = simps(vint,x)
            VH[i] = result

        for i in range(NGRID):
            upper = n[i]*PI*A  
            def ff(j):
                return kn(0,j)
            result, _ = quad(ff,0.,upper)
            VX[i] = -result/(PI*A)       
#
#   Now construct the time-dependent Hamiltonian
#
    
    
    for i in range(NGRID):
        HMAT[i,i] = HMAT[i,i] + VEXT[i] + VH[i] + VX[i] + VT[i]
    
#   Next, do the time propagation step by solving a linear equation
#   (Crank-Nicolson algorithm)
#        
    TMAT = ONE - 0.5*DT*IONE*HMAT
    
    RHS = TMAT.dot(PHI)
        
    TMAT = ONE + 0.5*DT*IONE*HMAT

    PHI = linalg.solve(TMAT,RHS)
    
    n = 2.*abs(PHI)**2

#
#   Real-time plotting
#   https://stackoverflow.com/a/63702988
#
    plt.cla() # Clear previous plots
    plt.axis([x[0],x[-1],0,1]) # Set the axis limits to always be the same
    plt.text(x[NGRID//10],0.7,f"t={T:1.2f}") # Display the time as text
    plt.plot(x,n0, label='ground-state density', color='blue') # Plot the ground-state density
    plt.plot(x,n, label='time-dependent density', color='red') # Plot the time-dependent density
    plt.xlabel('x')
    plt.ylabel('n')
    plt.legend() # Create the plot legend
    plt.pause(0.01) # Animation magic. Change the parameter to alter the "framerate"


    norm = simps(n,x)
#    print('norm',norm)
    
    n1 = n*x
    dip = simps(n1,x)
#    print('dipole moment',dip)
#    sys.exit()

    f.write(str(T) + "  " + str(dip) + '\n')
f.close()



plt.show() # Keep the plot open after the calculation is finished
