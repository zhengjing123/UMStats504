import pandas as pd
import numpy as np

# To obtain the data:
# wget http://2014.afghanistanelectiondata.org/data/results/download/2014_afghanistan_preliminary_runoff_election_results.csv
# wget http://2014.afghanistanelectiondata.org/data/results/download/2014_afghanistan_election_results.csv

fname1 = "2014_afghanistan_election_results.csv"
fname2 = "2014_afghanistan_preliminary_runoff_election_results.csv"

# Round 1 and round 2 raw data
df1 = pd.read_csv(fname1, encoding='latin1')
df2 = pd.read_csv(fname2, encoding='latin1')

# Merge on these columns
df1 = df1.set_index(["PC_number", "PS_number"])
df2 = df2.set_index(["PC_number", "PS_number"])

# Clean up the round 1 column names
df1 = df1.rename(columns={'province': 'Province',
                          'district': 'District',
                          'Eng-Qutbuddin Hilal': "Hilal",
                          'Dr. Abdullah Abdullah': 'Abdullah',
                          'Zalmai Rassoul': "Rassoul",
                          'Abdul Rahim Wardak': 'Wardak',
                          'Quayum Karzai': "Karzai",
                          'Prof-Abdo Rabe Rasool Sayyaf': "Sayyaf",
                          'Dr. Mohammad Ashraf Ghani Ahmadzai': "Ghani",
                          'Mohammad Daoud Sultanzoy': 'Sultanzoy',
                          'Mohd. Shafiq Gul Agha Sherzai': "Sherzai",
                          'Mohammad Nadir Naeem': "Naeem",
                          'Hedayat Amin Arsala': "Arsala",
                          'Total': 'Total_1'
                          })

# Rename the round 2 columns, so they don't collide with the round 1 column names
df2 = df2.rename(columns={'Total': 'Total_2',
                          'Ghani': 'Ghani_2',
                          'Abdullah': 'Abdullah_2',
                          })

df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
df = df.reset_index()

df = df.rename(columns={"District_x": "District", "Province_x": "Province"})
df = df.drop(["District_y", "Province_y"], axis=1)
