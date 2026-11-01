import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

def FD_BS(S0:float|int = 100, K:float|int= 80, r:int = 0.05, sigma:int=.3, T:int = 1, J:int=100, N:int=10_000,z:int = 4, theta:float|int = 1/2):
    """
    Finite Difference approximation using Forward Difference, Backward Difference and Crank-Nicolson schemes 
    to solve the Black-Scholes PDE for European Call Options.
    Crank-Nicolson is used by default. 

    Args:
        S0 (float | int, optional): Price of the underlying stock, today. Defaults to 100.
        K (float | int, optional): Strike Price. Defaults to 80.
        r (int, optional): risk-free rate of the market. Defaults to 0.05.
        sigma (int, optional): std_dev of underlyign stock. Defaults to .3.
        T (int, optional): time until maturity. Defaults to 1.
        J (int, optional): Number of space intervals. Defaults to 100.
        N (int, optional): Number of time steps. Defaults to 10_000.
        z (int, optional): number of standard deviations defining the value of S_max,
        under the assumption the stock follows a GBM. Defaults to 4 to make the probability of 
        the stock reaching a higher value negligible.
        theta(float | int, optional): parameter to specify the scheme. 0=FTCS, 0.5=C-N, 1=BTCS
    Returns:
        Price (float): price of the option today considering S=S0. 
        S_vector(array): Vector with prices of stocks from 0 up to S_max including S0.
        C_today(array): Vector with option prices for all the stock prices in S_vector.
        stability(Bool): False if the sufficient stability condition is not satisfied.
    """
    theta_values = [0,0.5,1]
    if theta not in theta_values:
        raise ValueError("Parameter theta has to be 0, 0.5 or 1.")
    
    if sigma<0 or r<0 or S0<0:
        raise ValueError("Volatility, risk-free rate and initial price have to be positive.")
    
    if N<0 or J<0: 
        raise ValueError("The number of space intervals and time steps has to be positive.")
    
    if sigma**2 < r:
        raise ValueError("It is deprecated to use a volatility lower than the risk free rate.")
    
    if z < 4:
        raise ValueError("It is not recommended to use less than 4 std deviations to estimated the Maximum Price reached.")

    #define step size
    dt = T/N
    
    stability = True
    if theta == 0 and dt > 1/((sigma**2)*((J-1)**2)+r):
        stability = False

    
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
    
    #construct the left and right matrices for all schemes
    M_left = I - theta*dt*A
    M_right = I + (1-theta)*dt*A
    
    #we perform LU decomposition to our matrix M_left which is the matrix applied to Cn+1
    if theta != 0:
        M_left_lu = splu(M_left.tocsc())
    
    Cn = payoff_vector[1:-1] #interior points of the initial condition
    coef_up = dt * sup_diag[-1] #coefficient used for the boundary condition at j=J-1
    
    for n in range(N):      #n=0,...,N-1
        tau_old = n*dt
        tau_new = (n+1) * dt  #tau used for the boundary condition  
        bn_old = np.zeros(m)            
        bn_new = np.zeros(m) #vector of the boundary condition
        bn_old[-1] = coef_up*(S_max - K * np.exp(-r * tau_old))
        bn_new[-1] = coef_up * (S_max - K * np.exp(-r * tau_new)) #boundary condition
        bn_total = theta*bn_new + (1-theta)*bn_old
        
        if theta!= 0:
            Cn = M_left_lu.solve(M_right@Cn + bn_total) #construct the next step of Cn with solve to make it efficient
        else:  
            Cn = M_right @ Cn + bn_total
    
    C_today = np.concatenate([[0], Cn, [S_max - K*np.exp(-r*T)]])
    price = C_today[j0]
    
    return price, S_vector, C_today, stability

    