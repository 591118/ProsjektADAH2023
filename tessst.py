import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parametere
gamma = 0.03  # Justeringshastighet for styringsrenten
delta = 0.01  # Justeringshastighet for inflasjonen
i_target = 0.025  # Inflasjonsmålet
A = 0.01  # Amplitude for sinusfunksjonen til inflasjon
B = 2*np.pi/24  # Frekvens for sinusfunksjonen til inflasjon (en syklus per 12 enheter av tid)

# Differensialligninger
def model(y, t):
    i, r = y
    di_dt = delta * (r-i) + A * np.sin(B * t)  # Legger til sinusfunksjonen i inflasjonsraten
    dr_dt = gamma * (i_target - i)   # Styringsrenten justerer seg mot inflasjonsmålet
    return [di_dt, dr_dt]

# Initialbetingelser
i0 = i_target  # Begynner med inflasjon ved målet
r0 = 0.035  # En antatt initial styringsrente
y0 = [i0, r0]

# Tidsintervaller
t = np.linspace(0, 1000, 1000)  # Løper modellen over en lengre tidsperiode

# Løse ODE-systemet
solution = odeint(model, y0, t)

# Plotte løsningen
plt.figure(figsize=(12, 6))
plt.plot(t, solution[:, 0], label='Inflasjon (i)')
plt.plot(t, solution[:, 1], label='Styringsrente (r)')
plt.title('Dynamikken mellom styringsrente og inflasjon med Sinusfunksjon')

plt.legend()
plt.grid(True)
plt.show()


