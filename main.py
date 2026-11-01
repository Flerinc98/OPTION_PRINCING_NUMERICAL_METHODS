from FD_BS import FD_BS
from convergence_analysis import convergence_data, print_table, latex_table, plot_convergence
from BS_closedform import bs_price

theta_values = [0,0.5,1]

# ---- Results for changes in N for each scheme ---
ref_price = FD_BS(J=70, N=200_000, theta=0.5)[0]
print(ref_price)
values = [50*(2**n) for n in range(8)]
var_name = "N"
data = convergence_data(FD_BS, ref_price, var_name, values, theta_values, J=70)
print_table(data)
print(latex_table(data))
plot_convergence(data)

# ---- Results for changes in J for each scheme ---
S0 = 100
K= 80
r = 0.05
sigma=.3
ref_price = bs_price(S0=S0, K=K, r=r, T = 1.0,sigma=sigma)
print(ref_price)
values = [20*(2**n) for n in range(8)]
var_name = "J"
data = convergence_data(FD_BS, ref_price, var_name, values, theta_values, N=100_000, S_max=400)
print_table(data, var_label="J", step_label="dS", step_fn=lambda J: 400.0/J)
print(latex_table(data, var_label="J", step_label=r"\Delta S", step_fn=lambda J: 400.0/J,
                  caption="Space convergence at fixed $N$", label="tab:space-conv"))
plot_convergence(data, step_fn=lambda J: 400.0/J, xlabel=r"$\Delta S$",
                 title="Space convergence at fixed $N$",
                 fname="space_convergence.png")


#--- Results for simultaneous testing of J and N ----

data = convergence_data(FD_BS, ref_price, "J", values, theta_values,
                        S_max=400, N=lambda J: 5*J)
print_table(data, var_label="J", step_label="dS", step_fn=lambda J: 400.0/J)
print(latex_table(data, var_label="J", step_label=r"\Delta S", step_fn=lambda J: 400.0/J,
                  caption=r"Simultaneous refinement, $N = 5J$", label="tab:joint-conv"))
plot_convergence(data, step_fn=lambda J: 400.0/J, xlabel=r"$\Delta S$",
                 title=r"Simultaneous refinement, $\Delta\tau \propto \Delta S$",
                 fname="joint_convergence.png")