import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import pandas as pd

# --- PARAMETERS FOR BITCOIN ---
ticker = "BTC-USD"
r = 0.05
T = 1.0
M = 252
N = 10000
dividend_yield = 0.0
z = 1.96

# Get historical BTC data and calculate volatility
btc_data = yf.download(ticker, period="1y", interval="1d")['Close']  # This is now a Series
btc_returns = np.log(btc_data / btc_data.shift(1)).dropna()          # Also a Series
daily_vol = float(btc_returns.std())                                 # Force to scalar
sigma = daily_vol * np.sqrt(252)                                     # Annualized volatility
print(f"Estimated Annual Volatility for BTC: {sigma:.4f}")


# Get latest BTC price
data = yf.download(ticker, period="5d", interval="1d")['Close']
S0 = float(data.iloc[-1])
K = S0  # Strike = spot price


# --- Monte Carlo GBM Path Simulation with Antithetic Variates ---
def simulate_paths(S0, r, sigma, T, M, N, dividend_yield):
    dt = T / M
    drift = (r - dividend_yield - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.randn(N // 2, M)
    Z = np.vstack((Z, -Z))
    S = np.zeros((N, M + 1))
    S[:, 0] = S0
    for t in range(1, M + 1):
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])
    return S


# --- Lookback Option Payoffs ---
def floating_strike_call(S):
    return np.maximum(S[:, -1] - np.min(S, axis=1), 0)

def fixed_strike_call(S, K):
    return np.maximum(np.max(S, axis=1) - K, 0)


# --- Generic Price Estimator ---
def estimate_price(S0, sigma, r, T, K=100, kind='floating'):
    S_paths = simulate_paths(S0, r, sigma, T, M, N, dividend_yield)
    if kind == 'floating':
        payoff = floating_strike_call(S_paths)
    elif kind == 'fixed':
        payoff = fixed_strike_call(S_paths, K)
    else:
        raise ValueError("kind must be 'floating' or 'fixed'")
    return np.exp(-r * T) * np.mean(payoff)


# --- Greeks Calculation ---
def calc_delta(S0, h=1.0):
    return (estimate_price(S0 + h, sigma, r, T) - estimate_price(S0 - h, sigma, r, T)) / (2 * h)

def calc_gamma(S0, h=1.0):
    return (estimate_price(S0 + h, sigma, r, T) - 2 * estimate_price(S0, sigma, r, T) + estimate_price(S0 - h, sigma, r, T)) / (h ** 2)

def calc_vega(sigma, h=0.01):
    return (estimate_price(S0, sigma + h, r, T) - estimate_price(S0, sigma - h, r, T)) / (2 * h)

def calc_rho(r, h=0.01):
    return (estimate_price(S0, sigma, r + h, T) - estimate_price(S0, sigma, r - h, T)) / (2 * h)

def calc_theta(T, h=1/252):
    return (estimate_price(S0, sigma, r, T + h) - estimate_price(S0, sigma, r, T - h)) / (2 * h)


# --- Control Variate Pricing ---
def control_variate_price(S_paths, K, r, T):
    avg_geom = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
    payoff = floating_strike_call(S_paths)
    sigma_hat = sigma / np.sqrt(3)
    mu_hat = 0.5 * (r - dividend_yield - 0.5 * sigma*2) + 0.5 * sigma_hat*2
    d1 = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)
    exact_geo = np.exp(-r * T) * (S0 * np.exp(mu_hat * T) * norm.cdf(d1) - K * norm.cdf(d2))
    cov = np.cov(payoff, avg_geom)[0, 1]
    var = np.var(avg_geom)
    b_star = cov / var
    adj_payoff = payoff - b_star * (avg_geom - np.mean(avg_geom))
    price_cv = np.exp(-r * T) * np.mean(adj_payoff)
    error_cv = np.std(adj_payoff) / np.sqrt(N)
    return price_cv, error_cv


# --- Convergence Analysis ---
def convergence_analysis_both():
    N_vals = [1000, 2000, 5000, 10000, 15000, 20000]
    prices_floating, errors_floating = [], []
    prices_fixed, errors_fixed = [], []

    for n in N_vals:
        S_paths = simulate_paths(S0, r, sigma, T, M, n, dividend_yield)

        payoff_floating = floating_strike_call(S_paths)
        price_f = np.exp(-r * T) * np.mean(payoff_floating)
        error_f = np.std(payoff_floating) / np.sqrt(n)
        prices_floating.append(price_f)
        errors_floating.append(error_f)

        payoff_fixed = fixed_strike_call(S_paths, K)
        price_fx = np.exp(-r * T) * np.mean(payoff_fixed)
        error_fx = np.std(payoff_fixed) / np.sqrt(n)
        prices_fixed.append(price_fx)
        errors_fixed.append(error_fx)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(N_vals, prices_floating, 'o-', label='Floating')
    plt.plot(N_vals, prices_fixed, 'o--', label='Fixed')
    plt.fill_between(N_vals,
                     [p - z * e for p, e in zip(prices_floating, errors_floating)],
                     [p + z * e for p, e in zip(prices_floating, errors_floating)],
                     alpha=0.2, label='CI Floating', color='blue')
    plt.fill_between(N_vals,
                     [p - z * e for p, e in zip(prices_fixed, errors_fixed)],
                     [p + z * e for p, e in zip(prices_fixed, errors_fixed)],
                     alpha=0.2, label='CI Fixed', color='orange')
    plt.title("Convergence of Option Price")
    plt.xlabel("Simulations (N)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(N_vals, errors_floating, 'o-', label='Floating Error', color='blue')
    plt.plot(N_vals, errors_fixed, 'o--', label='Fixed Error', color='orange')
    plt.title("Convergence of Standard Error")
    plt.xlabel("Simulations (N)")
    plt.ylabel("Standard Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Final Execution Block ---
S_paths = simulate_paths(S0, r, sigma, T, M, N, dividend_yield)
payoff_floating = floating_strike_call(S_paths)
payoff_fixed = fixed_strike_call(S_paths, K)
price_floating = np.exp(-r * T) * np.mean(payoff_floating)
price_fixed = np.exp(-r * T) * np.mean(payoff_fixed)
error_floating = np.std(payoff_floating) / np.sqrt(N)
error_fixed = np.std(payoff_fixed) / np.sqrt(N)
price_cv, error_cv = control_variate_price(S_paths, K, r, T)

# Confidence Intervals
ci_floating = (price_floating - z * error_floating, price_floating + z * error_floating)
ci_fixed = (price_fixed - z * error_fixed, price_fixed + z * error_fixed)
ci_cv = (price_cv - z * error_cv, price_cv + z * error_cv)

# Greeks
delta = calc_delta(S0)
gamma = calc_gamma(S0)
vega = calc_vega(sigma)
rho = calc_rho(r)
theta = calc_theta(T)

# Print Results
print("\n--- Lookback Option Pricing (Monte Carlo) for BTC ---")
print(f"Floating Strike Call Price       : {price_floating:.4f} ± {error_floating:.4f}")
print(f"  95% CI: ({ci_floating[0]:.4f}, {ci_floating[1]:.4f})")
print(f"Fixed Strike Call Price          : {price_fixed:.4f} ± {error_fixed:.4f}")
print(f"  95% CI: ({ci_fixed[0]:.4f}, {ci_fixed[1]:.4f})")
print(f"Floating (Control Variate) Price : {price_cv:.4f} ± {error_cv:.4f}")
print(f"  95% CI: ({ci_cv[0]:.4f}, {ci_cv[1]:.4f})")

print("\n--- Greeks (Floating Strike) ---")
print(f"Delta : {delta:.4f}")
print(f"Gamma : {gamma:.4f}")
print(f"Vega  : {vega:.4f}")
print(f"Rho   : {rho:.4f}")
print(f"Theta : {theta:.4f}")
print("------------------------------------------------")

# Plot GBM Paths
plt.plot(S_paths[:100].T, lw=0.8)
plt.title("Sample Simulated GBM Paths (BTC)")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Convergence Plots
convergence_analysis_both()