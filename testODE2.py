import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parametere
alpha = 0.1  # reaksjonskoeffisient for styringsrente
beta = 0.1   # reaksjonskoeffisient for inflasjon
gamma = 0.01  # hvor raskt styringsrenten justeres
delta = 0.01  # hvor raskt inflasjonen justeres
i_target = 0.02  # inflasjonsmålet

# Differensialligninger
def model(y, t):
    i, r = y
    di_dt = delta * (r - i)
    dr_dt = gamma * (i_target - i)
    return [di_dt, dr_dt]

# Initialbetingelser
i0 = 0.025  # initial inflasjon
r0 = 0.035  # initial styringsrente
y0 = [i0, r0]

# Tidsintervaller
t = np.linspace(0, 1000, 1000)

# Løse ODE-systemet
solution = odeint(model, y0, t)

# Plotte løsningene
plt.figure(figsize=(12, 6))
plt.plot(t, solution[:, 0], 'orange', label='Inflasjon')
plt.plot(t, solution[:, 1], 'blue', label='Styringsrente')
plt.title('Dynamikken mellom styringsrente og inflasjon over tid')
plt.xlabel('Tid')
plt.ylabel('Verdi')
plt.legend()
plt.grid(True)
plt.show()
