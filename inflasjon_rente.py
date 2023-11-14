import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def hent_csv():
    styringsrente_1991 = pd.read_csv('datasett_KPI_Styringsrente/styringsrenten.csv',delimiter=";",decimal=",")
    styringsrente_2018 = pd.read_csv("datasett_KPI_Styringsrente/IR.csv",delimiter=";", decimal = ",")
    kpi_2006 = pd.read_csv('datasett_KPI_Styringsrente/kpi_tab_no.csv',delimiter=";",decimal=",")
    kpi_1991 = pd.read_csv("datasett_KPI_Styringsrente/kpi_1991_2023.csv", delimiter=";", decimal=",")
    kpi_2019 = pd.read_csv("datasett_KPI_Styringsrente/kpi_test.csv", delimiter=";", decimal=",")
    return styringsrente_1991,styringsrente_2018,kpi_2006,kpi_1991,kpi_2019
styringsrente_1991, styringsrente_2018, KPI_2006, kpi_1991, kpi_test = hent_csv()
# processing csv files
def process():
    styringsrente_2018["TIME_PERIOD"] = pd.to_datetime(styringsrente_2018["TIME_PERIOD"], dayfirst=True)
    kpi_1991['Dato'] = pd.to_datetime(kpi_1991['Dato'], dayfirst=True, format="%YM%m")
    kpi_test['Dato'] = pd.to_datetime(kpi_1991['Dato'], dayfirst=True, format="%YM%m")
    styringsrente_1991["TIME_PERIOD"] = pd.to_datetime(styringsrente_1991["TIME_PERIOD"], dayfirst=True)
def process_dataframe(KPI_2006, date_column='Dato', start_date='2018-10-31'):
    """
    Process the given dataframe by:
    1. Converting the specified date_column to datetime format.
    2. Filtering rows based on a start_date.
    3. Resetting the index.
    
    Parameters:
    - KPI_2006: DataFrame to be processed.
    - date_column: Column name with dates. Default is 'Dato'.
    - start_date: Filter rows based on this date. Default is '2018-10-31'.
    
    Returns:
    - Processed DataFrame.
    """
    KPI_2006 = KPI_2006.loc[:, ~KPI_2006.columns.str.startswith('Unnamed')]
    KPI_2006[date_column] = pd.to_datetime(KPI_2006[date_column], dayfirst=True)
    KPI_2006 = KPI_2006.loc[KPI_2006[date_column] >= start_date]
    return KPI_2006.reset_index(drop=True)
process()
KPI_2006_Prosessert = process_dataframe(KPI_2006,"Dato","2006-10-31")
styringsrente_1991_Prosessert = process_dataframe(styringsrente_1991,"TIME_PERIOD","2006-10")
da = process_dataframe(KPI_2006)

def forskyvNiMåneder(KPI_datasett,Styringsrente_datasett,dateColumnKPI = "Dato",dateColumnStyringsrente= "Dato"):
    KPI_datasett[dateColumnKPI] = pd.to_datetime(KPI_datasett[dateColumnKPI], dayfirst=True)
    KPI_datasett = KPI_datasett.iloc[9:]
    KPI_datasett.reset_index(drop=True)

    Styringsrente_datasett[dateColumnStyringsrente] = pd.to_datetime(Styringsrente_datasett[dateColumnStyringsrente], dayfirst=True)
    Styringsrente_datasett = Styringsrente_datasett.iloc[:-9]
    Styringsrente_datasett.reset_index(drop=True)


    return KPI_datasett,Styringsrente_datasett

daa,styringsrente_2018s = forskyvNiMåneder(da,styringsrente_2018,"Dato","TIME_PERIOD")
#KPI_2006_Prosessert,styringsrente_1991_Prosessert = forskyvNiMåneder(KPI_2006_Prosessert,styringsrente_1991_Prosessert,"Dato","TIME_PERIOD")





#calculating correlation
correlation = styringsrente_1991_Prosessert["OBS_VALUE"].corr(KPI_2006_Prosessert["KPI"])
print(f'Correlation between Interest Rate and KPI: {correlation:.2f}')





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

"""# KPI og styringsrente fra 1991
plot_data(
    [221, 222, 212],
    ["KPI", "Styringsrente", ""],
    [kpi_1991["Dato"], styringsrente_1991["TIME_PERIOD"], kpi_1991["Konsumprisindeks(2015=100)"]],
    [kpi_1991["Konsumprisindeks(2015=100)"], styringsrente_1991["OBS_VALUE"], styringsrente_1991["OBS_VALUE"]],
    labels=[None, None, ["Styringsrente", "KPI"]],
    plot_type="scatter"
)"""
plot_data(
    [111],
    ["Endring i KPI"],
    [KPI_2006_Prosessert["Dato"]],
    [KPI_2006_Prosessert["KPI"]],
    labels=[["Styringsrente", "Endring i KPI"]],
    plot_type="plot"
)
# KPI og styringsrente fra 2019
plot_data(
    [131, 132, 133],
    ["KPI", "Styringsrente", "Endring i KPI"],
    [kpi_test["Dato"], styringsrente_2018["TIME_PERIOD"], KPI_2006_Prosessert["Dato"]],
    [kpi_test["Konsumprisindeks(2015=100)"], styringsrente_2018["OBS_VALUE"], KPI_2006_Prosessert["KPI"]],
    labels=[None, None, None, ["Styringsrente", "Endring i KPI"]],
    plot_type="plot"
)

#Plotter effekten styringsrenten har på KPI:
#9 mnd effekt er inflasjonen plottet opp mot styringsrente 9 mnd tidligere.
# "Om KPI-endring er 4 denne måneden, så var styringsrenten ... for 9 måneder siden"
plot_data(
    [221,222,212],
    ["Direkte effekt(2018-)", "9 mnd effekt(2018-)","Siden 2006-"],
    [da["KPI"],daa["KPI"],KPI_2006_Prosessert["KPI"]],
    [styringsrente_2018["OBS_VALUE"], styringsrente_2018s["OBS_VALUE"],styringsrente_1991_Prosessert["OBS_VALUE"]],
    labels=[[None, "Styringsrente"],[None, "Styringsrente"],["Endring KPI", "Styringsrente"]],
    plot_type="scatter"
)




'''# calculate average KPI increase last 32 years
def calcAverageKPIIncrease():

    # Compute average yearly inflation %
    totalkpi = 0
    telle = 0
    kpi = 0
    for i in range(len(kpi_KPI_2006)):
        if (i%12 != 0):
            kpi = kpi + kpi_KPI_2006["Månestyringsrente_2018endring (prosent)"][i]
        else:
            totalkpi = totalkpi + kpi
            kpi = 0
            telle = telle+1

    #print(totalkpi/telle)'''

