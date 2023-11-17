# Clean and format the provided data for plotting
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Data as provided
df = pd.read_csv("forbruksvaner.csv", delimiter=";", header=1, index_col=0)


plt.figure(figsize=(12, 6))
for idx, category in enumerate(df.index):
    #if idx != 0:
        plt.plot(df.columns.astype(int), df.loc[category], marker='o', label=category)

# Adjusting the x-axis
plt.xticks(range(min(df.columns.astype(int)), max(df.columns.astype(int)) + 1, 1))

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Amount (kr)')
plt.title('Yearly Expenditures by Category')

# Adding a legend
plt.legend(title='Expenditure Categories', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
