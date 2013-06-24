'''
Created on Feb 13, 2013
@author: gutshall
This is a ODE L1-norm Minimization Majorization implementation in Python
'''

# System Library imports
from numpy import zeros,ones,max,sum,abs,sqrt,mean,std,spacing,count_nonzero,ceil
from numpy.random import rand,permutation
from numpy.linalg import norm
from time import sleep as pause
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl

# Soft Thresholding
def wthresh(x_sig, t):
    from numpy import sign
    
    tmp = abs(x_sig) - t
    tmp = (tmp + abs(tmp)) / 2
    x_sig = sign(x_sig)*tmp
    
    return x_sig

# -------------------------------------------------------------
# Define Constants
sparsity = 5                # Number of non-zeros
n = 100                     # Length of unknowns
m = int(ceil(n*1))          # Length of observation measurements
niter = 50000                 # Number of regression iterations
plotflag = True
plotModulo = 1000

# Section:01 ----------------------------------------------------
# Generate Test Data

# Generate the Convolution Kernel
A = rand(m,n) / sqrt(m)                        # Convolution matrix A, we assume each coeff in [-1/sqrt(m),1/sqrt(m)]
if plotflag == True:
    fig = plt.figure(1)
    pyl.pcolor(A)
    pyl.ylabel('m')
    pyl.xlabel('n')
    pyl.title('Convolution Matrix')
    pyl.show(block=False)

# Speed-up variable
AT = A.T

# Generate Sparse x vector (same non-zero indexes)
x = zeros((n,1))
x[permutation(n)[0:sparsity]] = 5*rand(sparsity,1) + 5 #Uniform random values assigned to fixed x values

# Populate the observation space
y = A.dot(x) + 1e-5*rand(m,1)

# Section:02 ----------------------------------------------------
# Calculate nessary parameters
tau = mean(abs(AT.dot(y)))                             # L1-norm regularization
#tau = 1e-2
dt = 1 / norm(A)*1e-2
T = tau*dt                                              # Soft thresholding parameter
x_est = zeros(x.shape)*1e-6                                  # Intialize the X_est with zero entries
J = zeros((niter,1))

# Intialize FISTA parameters
z = zeros(y.shape) 

# Section:03 ----------------------------------------------------
# Run Iterations
for k in range(0,niter):
    
    error = y - A.dot(x_est)
    
    J[k] = sum(abs(y - A.dot(x_est))**2) + tau*sum(abs(x_est))      # Calculate Objective Function
    
    znew = error*dt + z
    
    ww = AT.dot(error + z)*dt + x_est
    ww[ww <= 0] = 0                                            #non-zero constraint
    
    x_est = wthresh(ww,T)          # ISTA

    z = znew 
    
    # Online Debug Plotting
    if plotflag == True:
        if (k == 0):
            # First, need to create the template for the figure() on the first iteration
            fig = plt.figure(99)
            
            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(x_est,'.')
            ax1.set_title('x_est for iteration: ' + str(k))
            ax1.set_xlabel('Sample Index [j]')
            ax1.set_ylabel('x_est')
            
            ax2 = fig.add_subplot(2,1,2)
            ax2.loglog(J)
            ax2.set_xlabel('iteration [k]')
            ax2.set_ylabel('J(x)')
            ax2.grid(True)
            
            plt.show(block=False) # block = False, means that it will draw and continue
            
        elif not(k % plotModulo):
            # Update the figure
            print('k =',k)
            
            # Clear the previous plots
            ax1.clear()
            ax2.clear()
            
            # Replot the update
            ax1.plot(x_est,'.')
            ax1.set_title('x_est for iteration: ' + str(k))
            ax2.loglog(J,'b')
            
            # Draw and Pause
            fig.canvas.draw()
            pause(5e-2)
            
    elif not(k % plotModulo): # Just to make sure that everything is running
        print('k =',k)
        
# Remove Circular Artifact
# Algorithm always places a signal at index-m,
x_est[-1] = 0 # same thing as x_est(end)

# Calculate L0-norm difference
# Form a single vector for the result
true_x = zeros(x.shape)
x_est_prime = zeros(x.shape)
true_x[x > spacing(1)] = 1             # Note, in Python spacing(1) is used in place of eps
threshold = (mean(x_est) + 0.5*std(x_est))
x_est_prime[x_est > threshold] = 1 

# Save the L0-norm distance
L0_norm_diff = count_nonzero(x_est_prime - true_x)
print('L_0 norm difference = ',L0_norm_diff)

# Plot using Matplotlib
fig = plt.figure(2)
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x_est,'r*',x,'bo',y*((max(x) + 1) / max(y)),'b--',threshold*ones(x.shape),'r--')
ax1.set_xlabel('index [i]')
ax1.set_ylabel('x intensity')
ax1.set_title('Recovered Results \n l0-norm difference = ' + str(L0_norm_diff))
leg = ax1.legend(('Estimated', 'True Unknown', 'Measured','Threshold'))
plt.show()

# End of Program
