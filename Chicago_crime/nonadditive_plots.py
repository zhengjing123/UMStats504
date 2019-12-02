import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import patsy
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

# A non-additive model
fml = "Num ~ (%s) * (%s) + (%s) * (%s)" % (terms[0], terms[1], terms[1], terms[2])

# A grid of values spanning the day of year values
doyp = np.linspace(1, 365, 100).astype(np.int64)

pdf = PdfPages("nonadditive_plots.pdf")

# Loop over primary crime types, create a Poisson model for each type
rslt = []
for pt, dz in cdat.groupby("PrimaryType"):

    if pt not in ("BATTERY", "THEFT", "NARCOTICS", "CRIMINAL DAMAGE"):
        continue

    m = sm.GEE.from_formula(fml, groups="CommunityArea", family=sm.families.Poisson(),
               cov_struct=cov_struct(), data=dz)
    r = m.fit()

    dp = dz.iloc[0:200, :].copy()
    dp.iloc[0:100, :]["Year"] = 2010
    dp.iloc[0:100, :]["DayOfYear"] = doyp
    dp.iloc[0:100, :]["DayOfWeek"] = "Sa"
    dp.iloc[100:200, :]["Year"] = 2010
    dp.iloc[100:200, :]["DayOfYear"] = doyp
    dp.iloc[100:200, :]["DayOfWeek"] = "We"

    fb(dp)

    y = r.predict(exog=dp)
    y[0:100] -= y[0:100].mean()
    y[100:200] -= y[100:200].mean()

    plt.clf()
    plt.axes([0.12, 0.1, 0.75, 0.8])
    plt.grid(True)
    plt.title(pt)
    plt.plot(doyp, y[0:100], '-', label="Sa")
    plt.plot(doyp, y[100:200], '-', label="We")
    plt.xlabel("Day of year", size=15)
    plt.ylabel("Rate", size=15)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
    plt.gca().set_xticks([0+15, 90+15, 180+15, 270+15])
    plt.gca().set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
    pdf.savefig()
    plt.savefig(pt + ".png")

pdf.close()


import os
os.system("cp nonadditive_plots.pdf ~kshedden")