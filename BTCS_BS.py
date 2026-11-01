import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

def BTCS_BS(S0:float|int = 100, K:float|int= 80, r:int = 0.05, sigma:int=.15, T:int = 1, J:int=100, N:int=10_000,z:int = 4):
    
    #define step size
    dt = T/N
    
    #define boundaries for prices
    S_max = S0*np.exp((r-sigma**2/2)*T + z*sigma*np.sqrt(T))
    S_min = 0
    
    #setting up the price vector to make sure S0 is included
    j0 = max(1, int(round(S0 / (S_max/J)))) #solve for the node of S0 
    dS = S0 / j0                            #ds = Sj/j so solving with j-th value of S0
    S_max = J * dS                          #Smax = J*dS since Smin=0
    S_vector = np.linspace(S_min, S_max, J+1) #construct the vector with all the prices
    S_interior = S_vector[1:-1]
    
    #construct the payoff vector
    payoff_vector = np.maximum(S_vector-K, 0)
    
    #construct the A matrix
    m = J-1
    
    alpha = (r*S_interior)/(2*dS)
    beta = ((sigma**2)*S_interior**2)/(2*dS**2)
    main_diag = -(2*beta+r) #diagonals for matrix A
    sub_diag = beta-alpha
    sup_diag = beta + alpha
    
    A = sp.diags([sub_diag[1:], main_diag, sup_diag[:-1]], offsets=[-1,0,1], 
                 format = "csc") #building a sparse matrix with the diagonals
    I = sp.diags(np.ones(m), offsets = 0, format = "csc") #identity matrix to compute I-/+A
    M = I - dt*A
    
    #we perform LU decomposition to our matrix M
    M_lu = splu(M.tocsc())
    
    Cn = payoff_vector[1:-1] #interior points of the initial condition
    coef_up = dt * sup_diag[-1] #coefficient used for the boundary condition at j=J-1
    
    for n in range(N):      #n=0,...,N-1
        tau_new = (n+1) * dt  #tau used for the boundary condition              
        bn_new = np.zeros(m) #vector of the boundary condition
        bn_new[-1] = coef_up * (S_max - K * np.exp(-r * tau_new)) #boundary condition
        Cn = M_lu.solve(Cn + bn_new) #construct the next step of Cn with solve to make it efficient
    
    C_today = np.concatenate([[0], Cn, [S_max - K*np.exp(-r*T)]])
    
    price = C_today[j0]
    
    return price

BTCS_BS()
    