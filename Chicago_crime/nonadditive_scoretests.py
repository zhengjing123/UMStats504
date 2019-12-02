import pandas as pd
import numpy as np
import statsmodels.api as sm
from chicago_data import cdat

# 2001 and 2002 data look incorrect
cdat = cdat.loc[cdat.Year >= 2003, :]

# Recode day of week with text labels
cdat.loc[:, "DayOfWeek"] = cdat.DayOfWeek.replace({0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr", 5: "Sa", 6: "Su"})

# Construct a Fourier basis for day of year
def fb(xm):
    for q in 1, 2:
        xm.loc[:, "DayOfYear_sin_%d" % q] = np.sin(2 * np.pi * q * xm.DayOfYear / 365.25)
        xm.loc[:, "DayOfYear_cos_%d" % q] = np.cos(2 * np.pi * q * xm.DayOfYear / 365.25)

# Build the Fourier basis functions
fb(cdat)

# A few values of CommunityArea are missing.
cdat = cdat.dropna()

cov_struct = sm.cov_struct.Independence

# Model terms for the day of year Fourier basis
doy = "DayOfYear_sin_1 + DayOfYear_cos_1 + DayOfYear_sin_2 + DayOfYear_cos_2"

# The basis functions for the three time scales, in decreasing scale.
terms = ["bs(Year, 4)", doy, "C(DayOfWeek)"]

# The additive model
fml0 = "Num ~ %s + %s + %s" % (terms[0], terms[1], terms[2])

# Models with one two-way interaction
fml1 = "Num ~ (%s)*(%s) + %s" % (terms[0], terms[1], terms[2])
fml2 = "Num ~ (%s)*(%s) + %s" % (terms[0], terms[2], terms[1])
fml3 = "Num ~ (%s)*(%s) + %s" % (terms[1], terms[2], terms[0])

# Models with two two-way interactions
fml4 = "Num ~ (%s)*(%s) + (%s)*(%s)" % (terms[0], terms[1],
          terms[0], terms[2])
fml5 = "Num ~ (%s)*(%s) + (%s)*(%s)" % (terms[0], terms[1],
          terms[1], terms[2])
fml6 = "Num ~ (%s)*(%s) + (%s)*(%s)" % (terms[0], terms[2],
          terms[1], terms[2])

# Model with all two-way interactions
fml7 = "Num ~ (%s)*(%s) + (%s)*(%s) + (%s)*(%s)" % (terms[0], terms[1],
          terms[0], terms[2], terms[1], terms[2])

# Model with all two-way and three way interactions
fml8 = "Num ~ (%s)*(%s)*(%s)" % (terms[0], terms[1], terms[2])

# Loop over primary crime types, create a Poisson model for each type
rslt = []
for pt, dz in cdat.groupby("PrimaryType"):

    if pt not in ("BATTERY", "THEFT", "NARCOTICS", "CRIMINAL DAMAGE"):
        continue

    mr = []
    for fml in (fml0, fml1, fml2, fml3, fml4, fml5, fml6, fml7, fml8):

        m = sm.GEE.from_formula(fml, groups="CommunityArea", family=sm.families.Poisson(),
                   cov_struct=cov_struct(), data=dz)
        r = m.fit()
        mr.append((m, r))

    kw = ["---", "+--", "-+-", "--+", "++-", "+-+", "-++", "+++", "*"]

    def cst(p, q):
        r = mr[p][0].compare_score_test(mr[q][1])
        r = [pt, r["statistic"], r["p-value"], r["df"], mr[p][0].exog.shape[1], mr[q][1].model.exog.shape[1],
             kw[p], kw[q]]
        rslt.append(r)

    # Compare each model with one pairwise interaction to the additive model
    cst(1, 0)
    cst(2, 0)
    cst(3, 0)

    # Compare models with two pairwise interactions to submodels with one
    # pairwise interaction
    cst(4, 1)
    cst(5, 1)
    cst(4, 2)
    cst(6, 2)
    cst(5, 3)
    cst(6, 3)

    # Compare models with three pairwise interactions to submodels
    # with one pairwise interaction
    cst(7, 4)
    cst(7, 5)
    cst(7, 6)

    # Compare the model with a three-way interaction to the
    # submodels with all two-way interactions
    cst(8, 7)

    print(pt)

rslt = pd.DataFrame(rslt)
