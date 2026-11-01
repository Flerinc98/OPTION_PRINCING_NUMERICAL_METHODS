import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMES = {0: "FTCS (explicit)", 0.5: "Crank-Nicolson", 1: "BTCS (implicit)"}
STYLE = {0:   dict(color="#E8710A", marker="o", ls="--"),
         0.5: dict(color="#00998C", marker="s", ls="-"),
         1:   dict(color="#5A4FCF", marker="^", ls="-.")}

### the following function lets you define the variable either N or J, it's values, the three
### theta values and then thorugh **kw lets you unpack the rest of the parameters for FD_BS
def convergence_data(FD_BS, ref_price, var_name, values, theta_values, **kw):
    """errors[t] = list of (N or J, error, stable) ; unstable entries have error=nan."""
    out = {}
    for t in theta_values:
        rows = []
        for v in values: #for loop of values in the list of N or J 
            args = {var_name:v, "theta":t, **kw} #create a dictionary to unpack into FD_BS function 
            price, _, _, stable = FD_BS(**args) #returns the price and if the method is stable
            err = abs(ref_price - price) #if stable else np.nan
            rows.append((n, err, stable))
        out[t] = rows
    return out


def fitted_order(Ns, errs):
    """Least-squares slope of log(error) vs log(dt).  dt ~ 1/N, so order = slope."""
    Ns, errs = np.asarray(Ns, float), np.asarray(errs, float)
    ok = np.isfinite(errs) & (errs > 0)
    if ok.sum() < 2:
        return np.nan
    p = np.polyfit(np.log(1.0/Ns[ok]), np.log(errs[ok]), 1)
    return p[0]


def print_table(data):
    for t, rows in data.items():
        Ns   = [r[0] for r in rows]
        errs = [r[1] for r in rows]
        print(f"\n--- theta = {t}  ({SCHEMES[t]}) ---")
        print(f"{'N':>7} {'dt':>12} {'error':>13} {'ratio':>8}   stability")
        prev = None
        for n, e, stable in rows:
            if not stable:
                print(f"{n:7d} {1/n:12.3e} {'--':>13} {'--':>8}   UNSTABLE")
                prev = None
                continue
            ratio = prev/e if (prev and np.isfinite(e) and e > 0) else np.nan
            rs = f"{ratio:8.2f}" if np.isfinite(ratio) else f"{'--':>8}"
            print(f"{n:7d} {1/n:12.3e} {e:13.3e} {rs}   stable")
            prev = e
        p = fitted_order(Ns, errs)
        print(f"  least-squares fitted order:  p = {p:.3f}")


def latex_table(data, caption="Time convergence at fixed $J$", label="tab:time-conv"):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{r r r r l}", r"\hline",
             r"$N$ & $\Delta\tau$ & error & ratio & stability \\", r"\hline"]
    for t, rows in data.items():
        lines.append(r"\multicolumn{5}{l}{\textbf{%s} ($\theta=%s$)} \\" % (SCHEMES[t], t))
        prev = None
        for n, e, stable in rows:
            if not stable:
                lines.append(f"{n} & {1/n:.3e} & -- & -- & unstable \\\\")
                prev = None
                continue
            ratio = prev/e if (prev and np.isfinite(e) and e > 0) else None
            rs = f"{ratio:.2f}" if ratio else "--"
            lines.append(f"{n} & {1/n:.3e} & {e:.3e} & {rs} & stable \\\\")
            prev = e
        p = fitted_order([r[0] for r in rows], [r[1] for r in rows])
        lines.append(r"\multicolumn{5}{l}{\quad fitted order $p = %.3f$} \\ \hline" % p)
    lines += [r"\end{tabular}",
              r"\caption{%s}" % caption, r"\label{%s}" % label, r"\end{table}"]
    return "\n".join(lines)


def plot_convergence(data, fname="time_convergence.png"):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for t, rows in data.items():
        Ns   = np.array([r[0] for r in rows], float)
        errs = np.array([r[1] for r in rows], float)
        ok = np.isfinite(errs) & (errs > 0)
        if ok.sum() == 0:
            continue
        dt = 1.0/Ns[ok]
        p = fitted_order(Ns, errs)
        ax.loglog(dt, errs[ok], markersize=6, linewidth=1.6,
                  label=f"{SCHEMES[t]}  ($p={p:.2f}$)", **STYLE[t])

    # reference slopes, anchored to the plotted range
    all_dt = np.concatenate([1.0/np.array([r[0] for r in rows], float) for rows in data.values()])
    all_e  = np.concatenate([np.array([r[1] for r in rows], float) for rows in data.values()])
    fin = np.isfinite(all_e) & (all_e > 0)
    if fin.sum() == 0:
        return fname
    x = np.array([all_dt[fin].min(), all_dt[fin].max()])
    for order, lab in [(1, r"slope 1"), (2, r"slope 2")]:
        c = all_e[fin].max() / (all_dt[fin].max()**order)
        ax.loglog(x, 0.6*c*x**order, color="#9aa0a6", ls=":", lw=1.1, zorder=0)
        ax.annotate(lab, xy=(x[0], 0.6*c*x[0]**order), fontsize=8.5,
                    color="#6a6a6a", xytext=(4, 2), textcoords="offset points")

    ax.set_xlabel(r"$\Delta\tau$", fontsize=11)
    ax.set_ylabel("absolute error", fontsize=11)
    ax.set_title("Time convergence at fixed $J=100$", loc="left", fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0d0d0"); ax.spines["left"].set_color("#d0d0d0")
    ax.tick_params(colors="#6a6a6a", labelsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=170, facecolor="white")
    return fname