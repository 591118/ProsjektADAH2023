import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from styringsrenta_fordeling import bin_edges, probabilities

df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)
df = df.drop(df.columns[-1], axis=1)

#
num_simulations = 7000 # For ADA511-delen

sluttÅr = 2030 # StartÅr = 1999. Angi hvor lenge simulasjonen skal kjøre

# Parameters
alpha = 0.68   # Propensitet til å konsumere av disponibel inntekt (90% forbruk og 10% sparing)
beta = 0.9    # Justeringshastighet for forbruksvaner (MPC)


g = 0.017 # Vekstrate for inntekt (Må ha inflasjon i modellen for å kunne bruke denne. Antar i denne modellen at inflasjonen nuller ut økning i lønn.)
r = 0.03  # Rentesats på lån (Modellen antar fast rentesats)


I = 0.0185 # gj.snitt inflasjon på 2%

# Startverdier (ratioene er beregnet ut ifra gj. snitt i Norge):
 
#A0 = 160000   # Opprinnelig inntekt (Gj. nettolønn i Norge er 402 000kr - 284 100kr i 2012 - ca 184 800kr i 1999)
A0 = 400000
# Ratio mellom lønn og faste utgifter:
ratio_fasteU_lønn = 0.48
U0 = A0*ratio_fasteU_lønn   # (76808kr)Faste utgifter (Ca. Gj. i Norge mellom 1999-2012 i følge forbrukerundersøkelsen SSB) (Antar at denne også er konstant)

# Ratio mellom lønn og nedbetaling på lån:
ratio_nedbetaling_lønn = 0.145 #14,5%
#P = A0*ratio_nedbetaling_lønn   # (23224 kr) en tanke kunne vært å ha en P som varierer med en parameter som sier noe om hvor mye av lønningen en person er villig til å betale ned på gjelden (større for folk som tjener mindre)
P = 180000
# Ratio mellom lån og lønn:
ratio_lån_lønn = 1.86
#L0 = 350000    # (343728) Opprinnelig lånebeløp (338% av årslønnen som er gj. snitt ifølge Gjensidige, 186% i 1999)
L0 = 1360000
antallPersonerPerHusholdning = 2.1
F0 = 105000/antallPersonerPerHusholdning    # Opprinnelig diskresjonær(valgfritt) forbruk (105 000kr var gj. snitt i 1999 i følge dataen til SSB per husholdning))

S0 = 0
#Standardavvik for rnd verdier (normaldistribusjon)
# Burde kanskje sørge for at forholdet mellom startverdiene og standardavviket er likt på alle
# Kan diskutere om hva de burde være / om de burde være forskjellige
std_ratio = 4.62 #4.62
A0_std = A0/std_ratio 
U0_std = U0/std_ratio
L0_std = L0/std_ratio 
r_std = 0.01



'''def inflation(t):
    avg_inflation = 0.02  # Average inflation rate
    amplitude = 0.01      # Amplitude of the sinusoidal function
    period = 8            # Approximate period in years
    frequency = (2 * np.pi) / period
    return avg_inflation + amplitude * np.sin(-frequency * t)
'''

#plt.plot(KPI_2006_Prosessert["Dato"],popt)
# ODE system
def system(t, y, alpha, beta,P, r, g, I):

    F, L, A, U, S = y
    
    dU = I*U
    dA = g*A
    dL = r*L - P
    dF = alpha * (A - U - P) - beta * F
    dS = (1-alpha)*(A-U-P) + r*S
    return [dF, dL, dA, dU, dS]


years = np.arange(1999,sluttÅr)
t_span = (0, years[-1] - years[0])  # Time span for solve_ivp
t_eval = years - years[0]  # Specific time points for evaluation


# Solve ODE using solve_ivp
result = solve_ivp(system, t_span, [F0, L0, A0, U0, S0], args=(alpha, beta, P, r, g, I), t_eval=t_eval)

# Extracting the solution
forbruk  = result.y[0]

learning_rate = 1e-7  # Smaller learning rate
epsilon = 1e-8  # Smaller epsilon for gradient calculation
max_gradient = 1e4  # Maximum allowable gradient magnitude
threshold = 1e-6
max_iterations = 1000

#region gradient descent for alpha
actual_data = (df.loc[df.index[0]]-df.loc[df.index[1]]-df.loc[df.index[7]]-df.loc[df.index[8]]-df.loc[df.index[4]]-df.loc[df.index[10]])/2.1

def cost_function(alpha,beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data):
    y0 = [F0, L0, A0, U0, S0]
    result = solve_ivp(system, t_span, y0, args=(alpha, beta, P, r, g, I), t_eval=t_eval)
    model_output = result.y[0][0:11]  # Assuming y[0] is 'forbruk'
    return np.sum((model_output - actual_data) ** 2)


cost = cost_function(alpha, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data)
# Modified gradient function to calculate gradients for both alpha and beta
def gradient(alpha, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data):
    epsilon = 1e-5
    gradient_alpha = (cost_function(alpha + epsilon, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data) - 
                      cost_function(alpha, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data)) / epsilon

    gradient_beta = (cost_function(alpha, beta + epsilon, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data) - 
                     cost_function(alpha, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data)) / epsilon

    return gradient_alpha, gradient_beta

# Gradient descent loop with dual parameter optimization
for iteration in range(max_iterations):
    grad_alpha, grad_beta = gradient(alpha, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data)

    # Clip the gradients
    grad_alpha = np.clip(grad_alpha, -max_gradient, max_gradient)
    grad_beta = np.clip(grad_beta, -max_gradient, max_gradient)

    # Update alpha and beta
    #alpha -= learning_rate * grad_alpha
    #beta -= learning_rate * grad_beta

    new_cost = cost_function(alpha, beta, P, r, g, I, F0, L0, A0, U0, t_span, t_eval, actual_data)

    # Debugging print statements
    print(f"Iteration {iteration}: Alpha = {alpha}, Beta = {beta}, Cost = {new_cost}, Gradient Alpha = {grad_alpha}, Gradient Beta = {grad_beta}")

    # Check for overshooting or convergence
    if new_cost > cost or abs(new_cost - cost) < threshold or alpha < 0 or beta < 0:
        print(f"Stopping iteration at {iteration}.")
        break

    cost = new_cost

# Ensure alpha and beta stay within reasonable ranges
#alpha = max(0, min(alpha, 1))
#beta = max(0, min(beta, 1))  # Assuming a similar range for beta
print(f"Calibrated alpha: {alpha}, Calibrated beta: {beta}")
print(f'forholdet:{alpha/beta}')
#endregion


result = solve_ivp(system, t_span, [F0, L0, A0, U0, S0], args=(alpha, beta, P, r, g, I), t_eval=t_eval, method='RK45')

forbruk, lån, lønn, fasteUtgifter, sparing  = result.y

result1 = solve_ivp(system, t_span, [F0, L0, A0, U0, S0], args=(alpha, beta, P, r, g, I), t_eval=t_eval, method='RK23')

forbruk1, lån1, lønn1, fasteUtgifter1, sparing1  = result1.y

result2 = solve_ivp(system, t_span, [F0, L0, A0, U0, S0], args=(alpha, beta, P, r, g, I), t_eval=t_eval, method='DOP853')

forbruk2, lån1, lønn1, fasteUtgifter1, sparing1  = result1.y

# Plot results

plt.subplot(221)
plt.plot(years,forbruk-forbruk1, label = "RK45 - RK23")
plt.legend()
plt.subplot(222)
plt.plot(years,forbruk-forbruk2,label = "RK45 - DOP853")

plt.legend()
plt.subplot(212)
plt.plot(years,forbruk1-forbruk2, label = "RK23 - DOP853")
plt.legend()

plt.show()
#plt.plot(years,forbruk-forbruk2)
plt.figure(figsize=(10, 6))
plt.subplot(211)
#plt.plot(years, lån, label="lån")
plt.plot(years, sparing, label="Sparing")
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


simulation_data = []
loan_payback_times = []

for _ in range(num_simulations):
    # Generate random parameters
    #A0_rand = np.random.normal(A0, A0_std)
    #L0_rand = np.random.normal(L0, L0_std)
    #U0_rand = np.random.normal(U0, U0_std)
    #r_rand = np.round(np.random.normal(r, r_std)*100).astype(int)/100


    # rente ut ifra fordeling av styringsrenten:
    r_rand = np.random.choice(bin_edges[:-1],p=probabilities) # pluss 0.005 for å simulere rente på lån og ikke styringsrente!

    #Pe = A0*ratio_nedbetaling_lønn
    Pe = 120000

    P = np.round(np.random.normal(Pe,0.1*Pe)/1000).astype(int)*1000
 
    # Run the simulation with random parameters
    #result = solve_ivp(system, t_span, [F0, L0_rand, A0_rand, U0_rand], args=(alpha, beta, P, r_base, g), t_eval=t_eval)
    y0 = [F0, L0, A0, U0, S0]#, S0]
    result = solve_ivp(system, t_span, y0, args=(alpha, beta, P, r_rand, g, I), t_eval=t_eval)
    
    # Extract loan amount data and determine payback time
    loan_data = result.y[1]
    payback_time = next((time for time, loan in zip(t_eval, loan_data) if loan <= 0), 1000)


    # Save the parameters and outcomes for this iteration
    simulation_data.append({
        #'A0': y0[2],
        #'L0': y0[1],
        'r': f"r{r_rand}",
        'P': f"p{P}", #antall prosent av A0 som er P.
        'payback_time': f"pt{payback_time}"
    })



# Convert the data to a DataFrame
dope = pd.DataFrame(simulation_data)

# Optionally, save to a CSV file
dope.to_csv('simulation_data.csv', index=False)

plt.figure()
'''plt.subplot(411)
plt.title("Nedbetalingstid lån")

plt.scatter(dope['payback_time'],dope['L0'], marker='o', label="Gjeld")
plt.legend()
plt.subplot(412)
plt.scatter(dope['payback_time'],dope['A0'], marker='o',label = "Lønn")
plt.legend()'''
plt.subplot(413)
plt.scatter(dope['payback_time'],dope['r'], marker='o',label = "Rente")
plt.subplot(414)
plt.scatter(dope['payback_time'],dope['P'], marker='o',label = "P")

plt.xlabel("Antall år")
plt.ylabel("kr")
plt.legend()
plt.show()