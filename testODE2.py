import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parametere
gamma = 0.03  # hvor raskt styringsrenten justeres
delta = 0.01  # hvor raskt inflasjonen justeres
i_target = 0.025  # inflasjonsmålet

# Differensialligninger
def model(y, t):
    i, r = y
    di_dt = delta * (r - i)
    dr_dt = gamma * (i_target - i) 
    return [di_dt, dr_dt]

# Initialbetingelser
i0 = 0.025  # initial inflasjon
r0 = 0.09  # initial styringsrente
y0 = [i0, r0]

# Tidsintervaller
t = np.linspace(0, 1000, 1000)

# Løse ODE-systemet
solution = odeint(model, y0, t)

# Opprette figur og akser
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotte styringsrente
color = 'tab:blue'
ax1.set_xlabel('Tid')
ax1.set_ylabel('Styringsrente', color=color)
ax1.plot(t, solution[:, 1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Opprette en annen y-akse for inflasjon
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Inflasjon', color=color)
ax2.plot(t, solution[:, 0], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Tittel og grid
plt.title('Dynamikken mellom styringsrente og inflasjon over tid')
fig.tight_layout()
plt.grid(True)
plt.show()
