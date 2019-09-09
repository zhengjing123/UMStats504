import numpy as np
import pandas as pd
import statsmodels.api as sm
from data import df

# Candidate names for round 1 and round 2
candidates1 = ['Hilal', 'Abdullah', 'Rassoul', 'Wardak', 'Karzai', 'Sayyaf',
               'Ghani', 'Sultanzoy', 'Sherzai', 'Naeem', 'Arsala']
candidates2 = ['Abdullah_2', 'Ghani_2']

# Total votes per round
totals = ["Total_1", "Total_2"]

if True:
    # Treat missing values as zero and aggregate at the coarser geographic
    # resolution (i.e. polling centers)
    dx = df.copy()
    vx = candidates1 + candidates2 + totals
    dx.loc[:, vx] = dx.loc[:, vx].fillna(0)
    dx = dx.groupby("PC_number").agg(np.sum)
    dx = dx.loc[dx.Total_1 > 0, :]
else:
    # Analyze the data at the level of polling stations (the finer level
    # of resolution), and drop polling stations that do not appear in
    # both rounds
    dx = df.dropna()

# Fit models for the round 2 turnout in terms
# of the round 1 vote totals for all candidates.

fml0 = "Total_2 ~ " + " + ".join(candidates1)
model0 = sm.OLS.from_formula(fml0, data=dx)
result0 = model0.fit()

# Fit models for the log round 2 turnout in terms
# of the log round 1 vote totals for all candidates.

dl = dx.copy()
for c in candidates1, candidates2, totals:
    dl[c] = np.log(1 + dl[c])

fml1 = "Total_2 ~ " + " + ".join(candidates1)
model1 = sm.OLS.from_formula(fml1, data=dl)
result1 = model1.fit()

# Fit models for each round 2 candidate's vote totals in round 2 in terms
# of the round 1 vote totals for all candidates.

fml2 = candidates2[0] + " ~ " + " + ".join(candidates1)
model2 = sm.OLS.from_formula(fml2, data=dx)
result2 = model2.fit()

fml3 = candidates2[1] + " ~ " + " + ".join(candidates1)
model3 = sm.OLS.from_formula(fml3, data=dx)
result3 = model3.fit()

pa = pd.DataFrame({"Abdullah": result2.params, "Ghani": result3.params})

# Fit models for each round 2 candidate's log vote totals in round 2 in terms
# of the log round 1 vote totals for all candidates.

dl = dx.copy()
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

mnx = model6.exog.mean(0)
mnx0 = mnx.copy()
mnx0[model6.exog_names.index("Abdullah")] = 0
mnx0[model6.exog_names.index("Ghani")] = 0
c = np.dot(mnx0, result6.params)
a = result6.params[model6.exog_names.index("Abdullah")]
g = result6.params[model6.exog_names.index("Ghani")]
# c + a*A + g*G is the fitted log ratio of Ghani support
# to Abdullah support in round 2, and c + (a-1)*A + (g-1)*G
# is the log "ratio of ratios" comparing round 2 to round 1.
