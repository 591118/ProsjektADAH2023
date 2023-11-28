import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

styringsrenten = pd.read_csv('datasett_KPI_Styringsrente/styringsrenten.csv', sep=";", decimal=",")


styringsrenten = styringsrenten.iloc[140:]
styringsrenten = styringsrenten.reset_index(drop=True)

styringsrenten["TIME_PERIOD"] = pd.to_datetime(styringsrenten['TIME_PERIOD'])
styringsrenten["OBS_VALUE"] = styringsrenten["OBS_VALUE"].astype(float)

print(styringsrenten['TIME_PERIOD'])
totalt_antall_datapunkter = styringsrenten['OBS_VALUE'].size

plt.subplot(121)

# Lager scatterplot på den primære aksen
plt.scatter(styringsrenten['TIME_PERIOD'], styringsrenten['OBS_VALUE'], alpha=0.3)
plt.xlabel('Tid')
plt.ylabel('Styringsrente')

plt.subplot(122)
rounded_values = np.round(styringsrenten['OBS_VALUE'] * 10) / 10
bins = np.arange(start=np.floor(styringsrenten['OBS_VALUE'].min()), stop=np.ceil(styringsrenten['OBS_VALUE'].max()), step=0.1)

plt.hist(rounded_values, bins = bins, color='red', orientation='horizontal', alpha=0.5)
frequencies, bin_edges = np.histogram(rounded_values,bins=bins)

bin_edges = np.round((bin_edges+0.5)*10)/1000

probabilities = frequencies / totalt_antall_datapunkter
probabilities /= probabilities.sum() # gjøres for at summen av sannsynligheter blir 1. etter avrunding

print(probabilities) # sannsynlighetsfordeling av 0.1 intervaller fra 0.0 til 6.8
'''plt.title('Scatter plot og histogram av styringsrenten')
plt.show()'''

for i in range(0,1000):
    e = np.random.choice(bin_edges[:-1],p=probabilities)
    print(e)
