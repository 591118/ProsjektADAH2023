import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)

# Parameters
alpha = 0.9   # Propensitet til å konsumere av disponibel inntekt (90% forbruk og 10% sparing)
beta = 0.33    # Justeringshastighet for forbruksvaner (MPC)
delta = 0.3    # Hvor mye renta på lånet avhenger av inflasjonen

g = 0.03      # Vekstrate for inntekt (Må ha inflasjon i modellen for å kunne bruke denne. Antar i denne modellen at inflasjonen nuller ut økning i lønn.)
U0 = 136600    # (210 000kr)Faste utgifter (Ca. Gj. i Norge mellom 1999-2012 i følge forbrukerundersøkelsen SSB) (Antar at denne også er konstant)
P = 44200      # Betalingsbeløp for lån (3,6k i mnd)
r_base = 0.02       # Rentesats på lån (Modellen antar fast rentesats)
L0 = 624600    # Opprinnelig lånebeløp (338% av årslønnen som er gj. snitt ifølge Gjensidige)
A0 = 195000    # Opprinnelig inntekt (Gj. nettolønn i Norge er 402 000kr - 284 100kr i 2012 - ca 184 800kr i 1999)
F0 = 105000    # Opprinnelig diskresjonær(valgfritt) forbruk (Dette var gj. snitt i 1999 i følge dataen til SSB)

def inflation(t):
    avg_inflation = 0.02  # Average inflation rate
    amplitude = 0.01      # Amplitude of the sinusoidal function
    period = 8            # Approximate period in years
    frequency = (2 * np.pi) / period
    return avg_inflation + amplitude * np.sin(-frequency * t)


# ODE system
def system(t, y, alpha, beta, P, r, g):
    #A = A0 * np.exp(g * t)       # Income growing exponentially
    #L = L0 * np.exp(-r * t) - P * t  # Loan decreasing over time
    F, L, A, U = y
    I = inflation(t)
    #r = delta * (r_base + I)
    r = r_base + delta*I

    # Har med denne koden fordi mye av betalingene til lånet er bare rentene, og ikke lånet selv.
    interest_payment = r*L
    principal_payment = P - interest_payment

    dU = I*U
    dA = g+I * A
    dL = -principal_payment
    dF = alpha * (A - U - interest_payment) - beta * F
    return [dF, dL, dA, dU]


years = np.arange(1999,2015)
t_span = (0, years[-1] - years[0])  # Time span for solve_ivp
t_eval = years - years[0]  # Specific time points for evaluation

# Solve ODE using solve_ivp
result = solve_ivp(system, t_span, [F0, L0, A0, U0], args=(alpha, beta, P, r_base, g), t_eval=t_eval)

# Extracting the solution
forbruk, lån, lønn, fast_forbruk = result.y

# Time vector for plotting (1999 to 2012)
#plt.plot(t_span,inflation(t_span))
plt.plot(t_eval,inflation(t_eval))
plt.show()

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(years, lån, label="lån")
plt.ylabel('[Kr]')
plt.legend()
plt.grid()

# Plotting additional data as in your original code
plt.subplot(212)
plt.plot(years, forbruk, label='Forbruk [kr]')
plt.plot(df.columns.astype(int), df.loc[df.index[0]]-df.loc[df.index[1]]-df.loc[df.index[7]]-df.loc[df.index[8]]-df.loc[df.index[4]]-df.loc[df.index[10]], marker='o', label=df.index[0])
#plt.title('Simulering av forbrukervaner')
plt.xlabel('Tid (år)')
plt.ylabel('[Kr]')
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
