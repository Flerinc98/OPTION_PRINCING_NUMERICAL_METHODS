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
            args = {var_name: v, "theta": t,
                    **{k: (f(v) if callable(f) else f) for k, f in kw.items()}} #create a dictionary to unpack into FD_BS function 
            price, _, _, stable = FD_BS(**args) #returns the price and if the method is stable
            err = abs(ref_price - price)
            usable =  np.isfinite(err) and err < abs(ref_price) #make sure the error is usable in case the sufficient 
            #condition is too conservative
            rows.append((v, err, stable, usable)) #for each value returns the value, error and if stable or not
        out[t] = rows #dictionary with theta as the keys and the list of resuts as the element in each key
    return out

### function to obtain the least-squares slope of the error vs the step/space size
def fitted_order(values, errs, usable):
    """Least-squares slope of log(error) vs log(dt/ds).  dt and dS ~ 1/N, so order = slope."""
    values, errs, usable  = np.asarray(values, float), np.asarray(errs, float), np.asarray(usable, bool)
    ok = np.isfinite(errs) & (errs > 0) & usable #check if the errors didn't explode or 
    if ok.sum() < 2: #are not lower than machine epsilon if at least one of them is false return nan
        return np.nan
    p = np.polyfit(np.log(1.0/values[ok]), np.log(errs[ok]),1) #degree 1 of fitting polynomial
    return p[0]


def print_table(data, var_label="N", step_label="dt", step_fn=lambda v: 1.0/v):
    for t, rows in data.items(): #unpacks for every value of theta and the respective data
        values   = [r[0] for r in rows] #value of N or J
        errs = [r[1] for r in rows] #the errors of each value of J or N 
        usable = [r[3] for r in rows]
        print(f"\n--- theta = {t}  ({SCHEMES[t]}) ---") #from the dictionary defined above 
        print(f"{var_label:>7} {step_label:>12} {'error':>13} {'ratio':>8}  {'condition':>10} {'observed':>9}") #columns of the table
        prev = None #no previous error for the first value of N or J 
        for v, e, stable, use in rows:
            cond = "satisfied" if stable else "violated"
            obs  = "ok" if use else "blew up" 
            if not use:
                print(f"{v:7d} {step_fn(v):12.3e} {'--':>13} {'--':>8} {cond:>10} {obs:>9}")
                prev = None
                continue
            ratio = prev/e if (prev is not None and e > 0) else np.nan
            rs = f"{ratio:8.2f}" if np.isfinite(ratio) else f"{'--':>8}"
            print(f"{v:7d} {step_fn(v):12.3e} {e:13.3e} {rs} {cond:>10} {obs:>9}")
            prev = e
        p = fitted_order(values, errs, usable)
        print(f"  least-squares fitted order:  p = {p:.3f}")


def latex_table(data, var_label="N", step_label=r"\Delta\tau",
                step_fn=lambda v: 1.0/v, show_stability=True,
                caption="Time convergence at fixed $J$", label="tab:time-conv"):
    ncol = 6 if show_stability else 4
    spec = "r r r r l l" if show_stability else "r r r r"
    head = f"${var_label}$ & ${step_label}$ & error & ratio"
    head += " & condition & observed \\\\" if show_stability else " \\\\"

    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{%s}" % spec, r"\hline", head, r"\hline"]
    for t, rows in data.items():
        lines.append(r"\multicolumn{%d}{l}{\textbf{%s} ($\theta=%s$)} \\"
                     % (ncol, SCHEMES[t], t))
        prev = None
        for v, e, stable, use in rows:
            tail = ""
            if show_stability:
                cond = "satisfied" if stable else "violated"
                obs  = "ok" if use else "blew up"
                tail = f" & {cond} & {obs}"
            if not use:
                lines.append(f"{v} & {step_fn(v):.3e} & -- & --{tail} \\\\")
                prev = None
                continue
            ratio = prev/e if (prev is not None and e > 0) else None
            rs = f"{ratio:.2f}" if ratio is not None and np.isfinite(ratio) else "--"
            lines.append(f"{v} & {step_fn(v):.3e} & {e:.3e} & {rs}{tail} \\\\")
            prev = e
        p = fitted_order([r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows])
        lines.append(r"\multicolumn{%d}{l}{\quad fitted order $p = %.3f$} \\ \hline"
                     % (ncol, p))
    lines += [r"\end{tabular}", r"\caption{%s}" % caption,
              r"\label{%s}" % label, r"\end{table}"]
    return "\n".join(lines)


def plot_convergence(data, step_fn=lambda v: 1.0/v, xlabel=r"$\Delta\tau$",
                     title="Time convergence at fixed $J$",
                     fname="time_convergence.png"):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    all_x, all_e = [], []

    for t, rows in data.items():
        values = np.array([r[0] for r in rows], float)
        errs   = np.array([r[1] for r in rows], float)
        usable = np.array([r[3] for r in rows], bool)
        ok = np.isfinite(errs) & (errs > 0) & usable
        if ok.sum() == 0:
            continue
        x = step_fn(values[ok])
        p = fitted_order(values, errs, usable)
        ax.loglog(x, errs[ok], markersize=6, linewidth=1.6,
                  label=f"{SCHEMES[t]}  ($p={p:.2f}$)", **STYLE[t])
        all_x.append(x); all_e.append(errs[ok])

    if not all_x:
        fig.savefig(fname, dpi=170, facecolor="white")
        return fname

    all_x = np.concatenate(all_x); all_e = np.concatenate(all_e)
    xr = np.array([all_x.min(), all_x.max()])
    for order, lab in [(1, "slope 1"), (2, "slope 2")]:
        c = all_e.max() / (all_x.max()**order)
        ax.loglog(xr, 0.6*c*xr**order, color="#9aa0a6", ls=":", lw=1.1, zorder=0)
        ax.annotate(lab, xy=(xr[0], 0.6*c*xr[0]**order), fontsize=8.5,
                    color="#6a6a6a", xytext=(4, 2), textcoords="offset points")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("absolute error", fontsize=11)
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0d0d0"); ax.spines["left"].set_color("#d0d0d0")
    ax.tick_params(colors="#6a6a6a", labelsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=170, facecolor="white")
    return fname