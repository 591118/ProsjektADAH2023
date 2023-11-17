import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
alpha = 0.5     # Propensity to consume out of disposable income
beta = 0.5      # Adjustment speed of spending habits
g = 0.02        # Growth rate of income
U = 180000       # Fixed expenses
P = 96000       # Payment amount for loans
r = 0.05        # Interest rate on loans
L0 = 2000000     # Initial loan amount
A0 = 480000     # Initial income
F0 = 72000      # Initial discretionary spending

# ODE system
def dFdt(F, t, alpha, beta, g, U, P, r, L0, A0):
    A = A0 * np.exp(g * t)       # Income growing exponentially
    L = L0 * np.exp(-r * t) - P * t  # Loan decreasing over time
    dF = alpha * (A - U - r * L) - beta * F
    return dF

# Time vector from 0 to 10 years
t = np.linspace(0, 10, 100)

# Solve ODE
F = odeint(dFdt, F0, t, args=(alpha, beta, g, U, P, r, L0, A0))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, F, label='Forbruk [kr]')
plt.title('Simulering av forbrukervaner utgifter over tid')
plt.xlabel('Tid (år)')
plt.ylabel('Forbruk kostnader')
plt.legend()
plt.grid(True)
plt.show()
