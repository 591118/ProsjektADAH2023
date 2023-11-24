import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)

# Parameters
alpha = 0.68   # Propensitet til å konsumere av disponibel inntekt (90% forbruk og 10% sparing)
beta = 1    # Justeringshastighet for forbruksvaner (MPC)




g = 0.017 # Vekstrate for inntekt (Må ha inflasjon i modellen for å kunne bruke denne. Antar i denne modellen at inflasjonen nuller ut økning i lønn.)
r = 0.03  # Rentesats på lån (Modellen antar fast rentesats)
I = 0.0185 # gj.snitt inflasjon på 2%

# Startverdier (ratioene er beregnet ut ifra gj. snitt i Norge):
 
A0 = 184800    # Opprinnelig inntekt (Gj. nettolønn i Norge er 402 000kr - 284 100kr i 2012 - ca 184 800kr i 1999)

# Ratio mellom lønn og faste utgifter:
ratio_fasteU_lønn = 2.406
U0 = A0/ratio_fasteU_lønn   # (76808kr)Faste utgifter (Ca. Gj. i Norge mellom 1999-2012 i følge forbrukerundersøkelsen SSB) (Antar at denne også er konstant)

# Ratio mellom lønn og nedbetaling på lån:
ratio_nedbetaling_lønn = 7.957
P = A0/ratio_nedbetaling_lønn   # (48774 kr)

# Ratio mellom lån og lønn:
ratio_lån_lønn = 1.86
L0 = A0*ratio_lån_lønn    # (721828.8) Opprinnelig lånebeløp (338% av årslønnen som er gj. snitt ifølge Gjensidige, 186% i 1999)

antallPersonerPerHusholdning = 2.1
F0 = 105000/antallPersonerPerHusholdning    # Opprinnelig diskresjonær(valgfritt) forbruk (105 000kr var gj. snitt i 1999 i følge dataen til SSB per husholdning))

#Standardavvik for rnd verdier (normaldistribusjon)
# Burde kanskje sørge for at forholdet mellom startverdiene og standardavviket er likt på alle
std_ratio = 4.62
A0_std = A0/std_ratio 
U0_std = U0/std_ratio
L0_std = L0/std_ratio 



def inflation(t):
    avg_inflation = 0.02  # Average inflation rate
    amplitude = 0.01      # Amplitude of the sinusoidal function
    period = 8            # Approximate period in years
    frequency = (2 * np.pi) / period
    return avg_inflation + amplitude * np.sin(-frequency * t)


#plt.plot(KPI_2006_Prosessert["Dato"],popt)
# ODE system
def system(t, y, alpha, beta,P, r, g, I):

    F, L, A, U = y

    # Har med denne koden fordi mye av betalingene til lånet er bare rentene, og ikke lånet selv.
    principal_payment = P - r*L
    
    dU = I*U
    dA = g*A
    dL = -principal_payment
    dF = alpha * (A - U - P) - beta * F
    return [dF, dL, dA, dU]


years = np.arange(1999,2020)
t_span = (0, years[-1] - years[0])  # Time span for solve_ivp
t_eval = years - years[0]  # Specific time points for evaluation


# Solve ODE using solve_ivp
result = solve_ivp(system, t_span, [F0, L0, A0, U0], args=(alpha, beta, P, r, g, I), t_eval=t_eval)

# Extracting the solution
forbruk, lån, lønn, fasteUtgifter  = result.y

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(211)
#plt.plot(years, lån, label="lån")
plt.plot(years, lønn-forbruk-fasteUtgifter, label="Sparing")
plt.plot(years, fasteUtgifter, label="Faste utgifter" )
plt.plot(years, lån, label="Gjeld" )
plt.plot(years, lønn, label="Inntekt" )
#plt.yscale("log")
plt.ylabel('[Kr]')
plt.legend()
plt.grid()

# Plotting additional data as in your original code
plt.subplot(212)
plt.plot(years, forbruk, label='Luksusforbruk [kr]')
plt.plot(df.columns.astype(int), (df.loc[df.index[0]]-df.loc[df.index[1]]-df.loc[df.index[7]]-df.loc[df.index[8]]-df.loc[df.index[4]]-df.loc[df.index[10]])/2.1, marker='o', label="SSB DATA")
#plt.title('Simulering av forbrukervaner')
plt.xlabel('Tid (år)')
plt.ylabel('[Kr]')
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()


''' ADA511 delen av prosjektet:'''

num_simulations = 1000
simulation_data = []
loan_payback_times = []

for _ in range(num_simulations):
    # Generate random parameters
    A0_rand = np.random.normal(A0, A0_std)
    L0_rand = np.random.normal(L0, L0_std)
    U0_rand = np.random.normal(U0, U0_std)

    P = A0_rand/ratio_nedbetaling_lønn
 
    # Run the simulation with random parameters
    #result = solve_ivp(system, t_span, [F0, L0_rand, A0_rand, U0_rand], args=(alpha, beta, P, r_base, g), t_eval=t_eval)
    result = solve_ivp(system, t_span, [F0, L0_rand, A0, U0], args=(alpha, beta, P, r, g, I), t_eval=t_eval)
    
    # Extract loan amount data and determine payback time
    loan_data = result.y[1]
    payback_time = next((time for time, loan in zip(t_eval, loan_data) if loan <= 0), None)


    # Save the parameters and outcomes for this iteration
    simulation_data.append({
        'A0': A0,
        'L0': L0_rand,
        'U0': U0,
        'payback_time': payback_time
    })



# Convert the data to a DataFrame
dope = pd.DataFrame(simulation_data)

# Optionally, save to a CSV file
dope.to_csv('simulation_data.csv', index=False)


plt.subplot(211)
plt.title("Nedbetalingstid lån")

plt.scatter(dope['payback_time'],dope['L0'], marker='o', label="Gjeld")
plt.legend()
plt.subplot(212)
plt.scatter(dope['payback_time'],dope['A0'], marker='o',label = "Lønn")

plt.xlabel("Antall år")
plt.ylabel("kr")
plt.legend()
plt.show()