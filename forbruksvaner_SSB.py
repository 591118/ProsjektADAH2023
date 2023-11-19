# Clean and format the provided data for plotting
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Data as provided
df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)

print(df.loc[df.index[1]]+df.loc[df.index[7]]+df.loc[df.index[8]]+df.loc[df.index[4]]+df.loc[df.index[10]])
plt.figure(figsize=(12, 6))
for idx, category in enumerate(df.index):
    #if idx != 0:
        plt.plot(df.columns.astype(int), df.loc[category], marker='o', label=category)

# Adjusting the x-axis
plt.xticks(range(min(df.columns.astype(int)), max(df.columns.astype(int)) + 1, 1))

# Adding labels and title
plt.xlabel('År')
plt.ylabel('Beløp (kr)')
plt.title('Årlige utgifter etter kategori')

# Adding a legend
plt.legend(title='Utgiftskategorier', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.yscale("log")
plt.tight_layout()
plt.show()
