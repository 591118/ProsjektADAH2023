import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)

# Parameters
alpha = 0.69     # Propensity to consume out of disposable income
beta = 0.9      # Adjustment speed of spending habits
#g = 0.02        # Growth rate of income (Må ha inflasjon i modell for å kunne bruke denne)
U = 270000       # Fixed expenses 
P = 96000       # Payment amount for loans (8k i mnd)
r = 0.045        # Interest rate on loans (Dagens rentesats)
L0 = 1622400     # Initial loan amount (338% av årslønnen som er gj. snitt ifølge Gjensidige)
A0 = 480000     # Initial income (Gj. i Norge er rundt 53k)
F0 = 100000      # Initial discretionary spending (Dette var gj. snitt i 1999 i følge SSB)

# ODE system
def dFdt(F, t, alpha, beta, U, P, r, L0, A0):
    #A = A0 * np.exp(g * t)       # Income growing exponentially
    L = L0 * np.exp(-r * t) - P * t  # Loan decreasing over time

    #Bruker A0, siden inflasjonen vil spise opp økningen av lønning
    dF = alpha * (A0 - U - r * L) - beta * F
    return dF

# Time vector from 0 to 10 years
t = np.linspace(0, 10, 10)
t1 = np.linspace(1999, 2012, 10)

# Solve ODE
F = odeint(dFdt, F0, t, args=(alpha, beta, U, P, r, L0, A0))

print(df.loc[df.index[0]])
print((df.loc[df.index[0]])-(df.loc[df.index[4]]))

'''ny = []
for element in df.loc[df.index[0]]:
    ny = element-df.loc[df.index[4]]'''
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t1, F, label='Forbruk [kr]')
plt.plot(df.columns.astype(int), df.loc[df.index[0]]-df.loc[df.index[1]]-df.loc[df.index[6]]-df.loc[df.index[7]]-df.loc[df.index[8]]-df.loc[df.index[4]]-df.loc[df.index[10]], marker='o', label=df.index[0])
plt.title('Simulering av forbrukervaner utgifter over tid')
plt.xlabel('Tid (år)')
plt.ylabel('Forbruk kostnader')
plt.legend()
plt.grid(True)
plt.show()
