
def bs_price(S0:int|float, K:int|float, r:float, T:int|float, sigma:float, q:float=0.0, option=1):
    
    import numpy as np
    from scipy.stats import norm
    
    """
    Closed form solution to compute the price of a call/put European
    option under the Black-Scholes model with the Merton
    extension to include the dividend rate parameter q. 

    Args:
        S0 (int | float): Initial price
        K (int | float): Strike price
        r (float): risk-free rate
        T (int | float): maturity time
        sigma (float): volatility
        option (str, optional): Parameter to choose either put or
        call option. Defaults to "call".

    Returns:
        _type_: _description_
    """
    d1 = (np.log(S0/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == 1:
        return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1)
