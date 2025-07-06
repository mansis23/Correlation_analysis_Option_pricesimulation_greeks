import numpy as np
import matplotlib.pyplot as plt

# --- BASE VALUES ---
S0 = 102768.2         # Spot Price
K = S0              # At-the-money strike
r = 0.05            # Risk-free rate
sigma = 0.41        # Volatility
T = 1.0             # Time to maturity
M = 252             # Time steps
N = 5000            # Simulations
dividend_yield = 0.0
z = 1.96

# --- MONTE CARLO SIMULATION ---
def simulate_paths(S0, r, sigma, T, M, N, dividend_yield):
    dt = T / M
    drift = (r - dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.randn(N // 2, M)
    Z = np.vstack((Z, -Z))  # Antithetic variates
    S = np.zeros((N, M + 1))
    S[:, 0] = S0
    for t in range(1, M + 1):
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])
    return S

# --- LOOKBACK PAYOFFS ---
def floating_strike_call(S):
    return np.maximum(S[:, -1] - np.min(S, axis=1), 0)

def fixed_strike_call(S, K):
    return np.maximum(np.max(S, axis=1) - K, 0)

# --- PRICING FUNCTION ---
def estimate_price(S0, sigma, r, T, M=252, N=10000, kind='floating'):
    S_paths = simulate_paths(S0, r, sigma, T, M, N, dividend_yield)
    if kind == 'floating':
        payoff = floating_strike_call(S_paths)
    elif kind == 'fixed':
        payoff = fixed_strike_call(S_paths, K)
    else:
        raise ValueError("Invalid kind")
    return np.exp(-r * T) * np.mean(payoff)

# --- GREEKS ---
def calc_all_greeks(S0, sigma, r, T, K, M=252, N=5000, h=1.0, dh=0.01, dy=0.01):
    base = estimate_price(S0, sigma, r, T, M, N, kind='fixed')
    delta = (estimate_price(S0 + h, sigma, r, T, M, N, kind='fixed') -
             estimate_price(S0 - h, sigma, r, T, M, N, kind='fixed')) / (2 * h)
    gamma = (estimate_price(S0 + h, sigma, r, T, M, N, kind='fixed') -
             2 * base +
             estimate_price(S0 - h, sigma, r, T, M, N, kind='fixed')) / (h ** 2)
    vega = (estimate_price(S0, sigma + dh, r, T, M, N, kind='fixed') -
            estimate_price(S0, sigma - dh, r, T, M, N, kind='fixed')) / (2 * dh)
    rho = (estimate_price(S0, sigma, r + dy, T, M, N, kind='fixed') -
           estimate_price(S0, sigma, r - dy, T, M, N, kind='fixed')) / (2 * dy)
    theta = (estimate_price(S0, sigma, r, T + dh, M, N, kind='fixed') -
             estimate_price(S0, sigma, r, T - dh, M, N, kind='fixed')) / (2 * dh)
    return base, delta, gamma, vega, rho, theta

# --- PARAMETER RANGES ---
r_vals = np.linspace(0.01, 0.10, 4)
sigma_vals = np.linspace(0.1, 0.5, 4)
div_yield_vals = np.linspace(0.00, 0.05, 4)
T_vals = np.linspace(0.25, 2.0, 4)
K_vals = np.linspace(S0 * 0.8, S0 * 1.2, 4)
S0_vals = np.linspace(S0 * 0.8, S0 * 1.2, 4)

results_dict = {
    "r": r_vals,
    "sigma": sigma_vals,
    "dividend_yield": div_yield_vals,
    "T": T_vals,
    "K": K_vals,
    "S0": S0_vals
}

# --- FINAL PLOTTING ---
fig, axs = plt.subplots(2, 3, figsize=(30, 20))
params = ["r", "sigma", "dividend_yield", "T", "K", "S0"]
titles = ["Risk-free Rate (r)", "Volatility (σ)", "Dividend Yield", "Time to Maturity (T)", "Strike Price (K)", "Spot Price (S₀)"]

for i, param in enumerate(params):
    x_vals = results_dict[param]
    price_vals, delta_vals, gamma_vals, vega_vals, rho_vals, theta_vals = [], [], [], [], [], []

    for val in x_vals:
        local_div_yield = dividend_yield
        if param == "r":
            args = (S0, sigma, val, T, K)
        elif param == "sigma":
            args = (S0, val, r, T, K)
        elif param == "dividend_yield":
            local_div_yield = val
            args = (S0, sigma, r, T, K)
        elif param == "T":
            args = (S0, sigma, r, val, K)
        elif param == "K":
            args = (S0, sigma, r, T, val)
        elif param == "S0":
            args = (val, sigma, r, T, K)

        # Overwrite dividend_yield globally ONLY if necessary
        dividend_yield = local_div_yield
        price, d, g, v, rh, th = calc_all_greeks(*args)
        price_vals.append(price)
        delta_vals.append(d)
        gamma_vals.append(g)
        vega_vals.append(v)
        rho_vals.append(rh)
        theta_vals.append(th)

    ax = axs[i // 3, i % 3]
    ax.plot(x_vals, price_vals, label="Price", color="black", linewidth=2)
    ax.plot(x_vals, delta_vals, label="Delta", linestyle="--")
    ax.plot(x_vals, gamma_vals, label="Gamma", linestyle="--")
    ax.plot(x_vals, vega_vals, label="Vega", linestyle="--")
    ax.plot(x_vals, rho_vals, label="Rho", linestyle="--")
    ax.plot(x_vals, theta_vals, label="Theta", linestyle="--")
    ax.set_title(f"Impact of {titles[i]}")
    ax.set_xlabel(param)
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()