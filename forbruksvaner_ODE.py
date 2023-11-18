import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)

# Parameters
alpha = 0.90   # Propensity to consume out of disposable income
beta = 0.4      # Adjustment speed of spending habits (MPC)

#g = 0.02        # Growth rate of income (Må ha inflasjon i modell for å kunne bruke denne. Antar i denne modellen at inflasjonen nuller ut økning i lønn.)
U = 210000       # Fixed expenses (Ca. Gj. i norge mellom 1999-2012 i følge forbrukerundersøkelsen SSB) (Antar at denne også er konstant)
P = 68000       # Payment amount for loans (5,6k i mnd)
r = 0.03        # Interest rate on loans (Modellen antar fast rentesats)
L0 = 960200     # Initial loan amount (338% av årslønnen som er gj. snitt ifølge Gjensidige)
A0 = 284100    # Initial income (Gj. nettolønn i Norge er 402 000kr - 284 100kr i 2012)
F0 = 105000      # Initial discretionary spending (Dette var gj. snitt i 1999 i følge dataen til SSB)

# ODE system
def system(t, y, alpha, beta, U, P, r):
    #A = A0 * np.exp(g * t)       # Income growing exponentially
    #L = L0 * np.exp(-r * t) - P * t  # Loan decreasing over time
    F, L = y
    dL = -P + r * L
    dF = alpha * (A0 - U - r * L) - beta * F
    return [dF, dL]


years = np.arange(1999,2013)
t_span = (0, years[-1] - years[0])  # Time span for solve_ivp
t_eval = years - years[0]  # Specific time points for evaluation

# Solve ODE using solve_ivp
result = solve_ivp(system, t_span, [F0, L0], args=(alpha, beta, U, P, r), t_eval=t_eval)

# Extracting the solution
forbruk, lån = result.y

# Time vector for plotting (1999 to 2012)


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(years, forbruk, label='Forbruk [kr]')
plt.plot(years, lån, label="lån")

# Plotting additional data as in your original code
plt.plot(df.columns.astype(int), df.loc[df.index[0]]-df.loc[df.index[1]]-df.loc[df.index[7]]-df.loc[df.index[8]]-df.loc[df.index[4]]-df.loc[df.index[10]], marker='o', label=df.index[0])
plt.title('Simulering av forbrukervaner')
plt.xlabel('Tid (år)')
plt.ylabel('Forbruk kostnader')
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
