"""
The data come from the IPUMS/NHGIS site: data2.nhgis.org
"""

import numpy as np
import pandas as pd

# Household income data
pa1 = "nhgis0001_csv/nhgis0001_ts_nominal_tract.csv"

# Age x sex population counts
pa2 = "nhgis0002_csv/nhgis0002_ts_nominal_tract.csv"

df1 = pd.read_csv(pa1, encoding="latin1")
df2 = pd.read_csv(pa2, encoding="latin1")

# We loose a lot of data here, but OK for our purposes
dx1 = df1.dropna()
dx2 = df2.dropna()

dx = pd.merge(dx2, dx1, left_on="NHGISCODE", right_on="NHGISCODE")

incvars = [c for c in dx.columns if c.startswith("A88") and not c.endswith("M")]
popvars = [c for c in dx.columns if c.startswith("B58")]

# Alternative ordering for income variables
io = []
for year in 1970, 1980, 1990, 2000, 125:
    for vn in "AA", "AB", "AC", "AD", "AE":
        io.append("A88%s%d" % (vn, year))
