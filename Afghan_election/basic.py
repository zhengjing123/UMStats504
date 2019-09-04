import numpy as np
import pandas as pd
import statsmodels.api as sm
from data import df

df = df.dropna()

# Candidate names for round 1 and round 2
candidates1 = ['Hilal', 'Abdullah', 'Rassoul', 'Wardak', 'Karzai', 'Sayyaf',
               'Ghani', 'Sultanzoy', 'Sherzai', 'Naeem', 'Arsala']
candidates2 = ['Abdullah_2', 'Ghani_2']

# Total votes per round
totals = ["Total_1", "Total_2"]

# Fit models for the round 2 turnout in terms
# of the round 1 vote totals for all candidates.

fml0 = "Total_2 ~ " + " + ".join(candidates1)
model0 = sm.OLS.from_formula(fml0, data=df)
result0 = model0.fit()

# Fit models for the log round 2 turnout in terms
# of the log round 1 vote totals for all candidates.

dl = df.copy()
for c in candidates1, candidates2, totals:
    dl[c] = np.log(1 + dl[c])

fml1 = "Total_2 ~ " + " + ".join(candidates1)
model1 = sm.OLS.from_formula(fml1, data=dl)
result1 = model1.fit()

# Fit models for each round 2 candidate's vote totals in round 2 in terms
# of the round 1 vote totals for all candidates.

fml2 = candidates2[0] + " ~ " + " + ".join(candidates1)
model2 = sm.OLS.from_formula(fml2, data=df)
result2 = model2.fit()

fml3 = candidates2[1] + " ~ " + " + ".join(candidates1)
model3 = sm.OLS.from_formula(fml3, data=df)
result3 = model3.fit()

pa = pd.DataFrame({"Abdullah": result2.params, "Ghani": result3.params})

# Fit models for each round 2 candidate's log vote totals in round 2 in terms
# of the log round 1 vote totals for all candidates.

dl = df.copy()
for c in candidates1, candidates2:
    dl[c] = np.log(1 + dl[c])

fml4 = candidates2[0] + " ~ " + " + ".join(candidates1)
model4 = sm.OLS.from_formula(fml4, data=dl)
result4 = model4.fit()

fml5 = candidates2[1] + " ~ " + " + ".join(candidates1)
model5 = sm.OLS.from_formula(fml5, data=dl)
result5 = model5.fit()

pal = pd.DataFrame({"Abdullah": result4.params, "Ghani": result5.params})

# Fit a model for the log ratio Ghani/Abdullah in round 2 based on
# the log round 1 vote totals for all candidates

dl["Ghani_Abdullah"] = dl.Ghani_2 - dl.Abdullah_2

fml6 = "Ghani_Abdullah ~ " + " + ".join(candidates1)
model6 = sm.OLS.from_formula(fml6, data=dl)
result6 = model6.fit()

