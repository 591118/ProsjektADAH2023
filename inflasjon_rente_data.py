import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def hent_csv():
    styringsrente_df = pd.read_csv('datasett_KPI_Styringsrente/styringsrenten.csv',delimiter=";",decimal=",")
    styringsrente_df2 = pd.read_csv("datasett_KPI_Styringsrente/IR.csv",delimiter=";", decimal = ",")
    kpi_2006 = pd.read_csv('datasett_KPI_Styringsrente/kpi_tab_no.csv',delimiter=";",decimal=",")
    kpi_1991 = pd.read_csv("datasett_KPI_Styringsrente/kpi_1991_2023.csv", delimiter=";", decimal=",")
    kpi_2019 = pd.read_csv("datasett_KPI_Styringsrente/kpi_test.csv", delimiter=";", decimal=",")
    return styringsrente_df,styringsrente_df2,kpi_2006,kpi_1991,kpi_2019
styringsrente_df, ds, df, kpi_1991, kpi_test = hent_csv()
# processing csv files
def process():
    ds["TIME_PERIOD"] = pd.to_datetime(ds["TIME_PERIOD"], dayfirst=True)
    kpi_1991['Dato'] = pd.to_datetime(kpi_1991['Dato'], dayfirst=True, format="%YM%m")
    kpi_test['Dato'] = pd.to_datetime(kpi_1991['Dato'], dayfirst=True, format="%YM%m")
    styringsrente_df["TIME_PERIOD"] = pd.to_datetime(styringsrente_df["TIME_PERIOD"], dayfirst=True)
def process_dataframe(df, date_column='Dato', start_date='2018-10-31'):
    """
    Process the given dataframe by:
    1. Converting the specified date_column to datetime format.
    2. Filtering rows based on a start_date.
    3. Resetting the index.
    
    Parameters:
    - df: DataFrame to be processed.
    - date_column: Column name with dates. Default is 'Dato'.
    - start_date: Filter rows based on this date. Default is '2018-10-31'.
    
    Returns:
    - Processed DataFrame.
    """
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
    df = df.loc[df[date_column] >= start_date]
    return df.reset_index(drop=True)
process()
da = process_dataframe(df)


"""
#calculating correlation
correlation = styringsrente_df['OBS_VALUE'].corr(kpi_df['Månedsendring (prosent)'])
print(f'Correlation between Interest Rate and KPI: {correlation:.2f}')
"""


def plot_data(subplot_positions, titles, x_values, y_values, labels=None, plot_type="plot"):
    """
    A reusable function to plot data with enhanced aesthetics.
    """
    plt.figure(figsize=(12, 8))
    plt.tight_layout(pad=5.0)  # adjust the spacing between subplots

    for idx, pos in enumerate(subplot_positions):
        ax = plt.subplot(pos)
        
        # Adding gridlines
        ax.grid(which='both', linestyle='--', linewidth=0.5)

        # Setting title and labels with increased font sizes
        plt.title(titles[idx], fontsize=16)
        
        if plot_type == "plot":
            plt.plot(x_values[idx], y_values[idx], linewidth=2, color='royalblue')
        elif plot_type == "scatter":
            plt.scatter(x_values[idx], y_values[idx], color='darkblue', edgecolor='white', s=50)
            if labels and labels[idx]:
                plt.xlabel(labels[idx][0], fontsize=14)
                plt.ylabel(labels[idx][1], fontsize=14)

        # Increase tick font size
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.show()

# Lager en modell for inflasjon:

# norges bank innførte i 2001 et inflasjonsmål på 2,5% per år:
kpi_2001 = process_dataframe(kpi_1991,'Dato','2000-12-31')
# x-verdier til x% inflasjon
def modell_KPI_x_prosent(xProsent):
    linear = []
    x = []
    for i in range(len(kpi_2001)):
        x.append((i+12)/12 - 1)
    # Compute the linear curve based on the years
    
    for element in x:
        linear.append(kpi_2001["Konsumprisindeks(2015=100)"][0]*(xProsent/100+1) ** element)

    return linear


# plots
plt.plot(kpi_1991["Dato"], kpi_1991["Konsumprisindeks(2015=100)"], color="red", label="Reell KPI")
plt.plot(kpi_2001["Dato"], modell_KPI_x_prosent(2.5), color = "blue", label="KPI-mål (2,5% økning pr. år)")
plt.plot(kpi_2001["Dato"], modell_KPI_x_prosent(2), color = "green", label="2% KPI økning pr. år")
plt.xlabel("År")
plt.ylabel("KPI")
plt.title("Inflasjon prognose & reell")
plt.legend()
plt.show()

# KPI og styringsrente fra 1991
plot_data(
    [221, 222, 212],
    ["KPI", "Styringsrente", ""],
    [kpi_1991["Dato"], styringsrente_df["TIME_PERIOD"], kpi_1991["Konsumprisindeks(2015=100)"]],
    [kpi_1991["Konsumprisindeks(2015=100)"], styringsrente_df["OBS_VALUE"], styringsrente_df["OBS_VALUE"]],
    labels=[None, None, ["Styringsrente", "KPI"]],
    plot_type="scatter"
)

# KPI og styringsrente fra 2019
plot_data(
    [231, 232, 233, 212],
    ["KPI", "Styringsrente", "", ""],
    [kpi_test["Dato"], ds["TIME_PERIOD"], da["Dato"], da["KPI"]],
    [kpi_test["Konsumprisindeks(2015=100)"], ds["OBS_VALUE"], da["KPI"], ds["OBS_VALUE"]],
    labels=[None, None, None, ["Styringsrente", "KPI"]],
    plot_type="scatter"
)



# calculate average KPI increase last 32 years
def calcAverageKPIIncrease():

    # Compute average yearly inflation %
    totalkpi = 0
    telle = 0
    kpi = 0
    for i in range(len(kpi_df)):
        if (i%12 != 0):
            kpi = kpi + kpi_df["Månedsendring (prosent)"][i]
        else:
            totalkpi = totalkpi + kpi
            kpi = 0
            telle = telle+1

    #print(totalkpi/telle)

