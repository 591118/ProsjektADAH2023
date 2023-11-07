import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
k1 = -0.3  # How quickly inflation responds to the policy rate (negative sign to show inverse relationship)
k2 = 0.5   # How aggressively the policy rate is adjusted
n = 0.02   # Natural rate of inflation
I_target = 0.02  # Target inflation rate

# The system of ODEs
def model(y, t, k1, k2, n, I_target):
    I, R = y
    dIdt = k1 * (R - I) + n  # Adjusted the sign to (-) for k1 to correct the relationship
    dRdt = k2 * (I - I_target) - 0.3 * (R - I)  # Added a dampening factor proportional to R
    return [dIdt, dRdt]

# Initial conditions
I0 = 0.01  # Initial inflation rate
R0 = 0.01  # Initial policy rate
y0 = [I0, R0]

# Time points to solve the ODEs for
t = np.linspace(0, 50, 500)

# Solve ODEs
solution = odeint(model, y0, t, args=(k1, k2, n, I_target))

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(t, solution[:, 0], 'b', label='Inflation rate I(t)')
plt.plot(t, solution[:, 1], 'r', label='Policy rate R(t)')
plt.xlabel('Time')
plt.ylabel('Rates')
plt.title('Inflation and Policy Rate Over Time')
plt.legend(loc='best')
plt.grid()
plt.show()
