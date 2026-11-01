import numpy as np
import scipy.sparse as sp

def FTCS_BS(S0:float|int = 100, K:float|int= 80, r:int = 0.05, sigma:int=.15, T:int = 1, J:int=100, N:int=10_000,z:int = 4):
    
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
    M = I + dt*A
    
    Cn = payoff_vector[1:-1]
    coef_up = dt * sup_diag[-1]
    
    for n in range(N):              # n = 0 .. N-1
        tau = n * dt                # FTCS uses the OLD level, so tau_n
        bn = np.zeros(m)
        bn[-1] = coef_up * (S_max - K * np.exp(-r * tau)) #boundary condition
        Cn = M @ Cn + bn #construct the next step of Cn
    
    C_today = np.concatenate([[0], Cn, [S_max - K*np.exp(-r*T)]])
    
    price = C_today[j0]
    
    return price

FTCS_BS()
    