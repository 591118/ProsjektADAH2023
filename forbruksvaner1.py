import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)

# Parameters
alpha = 0.25     # Propensity to consume out of disposable income
beta = 0.5      # Adjustment speed of spending habits
g = 0.02        # Growth rate of income
U = 180000       # Fixed expenses
P = 96000       # Payment amount for loans
r = 0.05        # Interest rate on loans
L0 = 1622400     # Initial loan amount
A0 = 480000     # Initial income
F0 = 100000      # Initial discretionary spending

# ODE system
def dFdt(F, t, alpha, beta, g, U, P, r, L0, A0):
    A = A0 * np.exp(g * t)       # Income growing exponentially
    L = L0 * np.exp(-r * t) - P * t  # Loan decreasing over time

    #Bruker A0, siden inflasjonen vil spise opp økningen av lønning
    dF = alpha * (A0 - U - r * L) - beta * F
    return dF

# Time vector from 0 to 10 years
t = np.linspace(0, 10, 12)

# Solve ODE
F = odeint(dFdt, F0, t, args=(alpha, beta, g, U, P, r, L0, A0))

print(df.loc[df.index[0]])
print((df.loc[df.index[0]])-(df.loc[df.index[4]]))

'''ny = []
for element in df.loc[df.index[0]]:
    ny = element-df.loc[df.index[4]]'''
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df.columns.astype(int), F, label='Forbruk [kr]')
plt.plot(df.columns.astype(int), df.loc[df.index[0]]-df.loc[df.index[1]]-df.loc[df.index[6]]-df.loc[df.index[7]]-df.loc[df.index[8]]-df.loc[df.index[4]]-df.loc[df.index[10]], marker='o', label=df.index[0])
#plt.plot(df.columns.astype(int),df.loc[df.index[0]])
plt.title('Simulering av forbrukervaner utgifter over tid')
plt.xlabel('Tid (år)')
plt.ylabel('Forbruk kostnader')
plt.legend()
plt.grid(True)
plt.show()
