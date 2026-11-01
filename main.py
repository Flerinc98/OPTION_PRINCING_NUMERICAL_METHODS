from OPTION_PRINCING_NUMERICAL_METHODS.FD_BS import FD_BS
from OPTION_PRINCING_NUMERICAL_METHODS.convergence_analysis import convergence_data, print_table, latex_table, plot_convergence
from OPTION_PRINCING_NUMERICAL_METHODS.BS_closedform import bs_price

theta_values = [0,0.5,1]

# ---- Results for changes in N for each scheme ---

ref_price = FD_BS(J=70, N=200_000, theta=0.5)[0]
N_sizes = [10*(2**n) for n in range(8)]

data = convergence_data(FD_BS, ref_price, N_sizes, theta_values, J=70)
print_table(data)
print(latex_table(data))
plot_convergence(data)

# ---- Results for changes in J for each scheme ---
S0 = 100
K= 80
r = 0.05
sigma=.3

ref_price = bs_price(S0=S0, K=K, r=r, T = 1.0,sigma=sigma)


