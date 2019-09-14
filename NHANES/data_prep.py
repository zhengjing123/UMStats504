# This script merges a few NHANES files, so that they can then
# be analyzed, e.g. using regression analysis
#
# First, download some of the NHANES data files.  You can do
# this through the NHANES web site, or on the command line
# using the commands below:
#
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.XPT
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.XPT
#
# See the NHANES web site for code books describing what each
# variable means.

import pandas as pd
import numpy as np

# List of file names to merge
xpt_files = ["DEMO_I.XPT", "BPX_I.XPT", "BMX_I.XPT"]

# Retain these variables.
vars = [
    ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1"],
    ["SEQN", "BPXSY1", "BPXDI1"],
    ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXLEG", "BMXARMC"],
]

# Load each individual file and keep only the variables of interest
da = []
for idf, fn in enumerate(xpt_files):
    df = pd.read_sas(fn)
    df = df.loc[:, vars[idf]]
    da.append(df)

# SEQN is a common subject ID that can be used to merge the files.
# These are cross sectional (wide form) files, so there is at most
# one row per subject.  Subjects may be missing from a file if they
# did not participate in those assessments.  All subjects should be
# present in the demog file.
dx = pd.merge(da[0], da[1], left_on="SEQN", right_on="SEQN")
dx = pd.merge(dx, da[2], left_on="SEQN", right_on="SEQN")

# Recode sex as an indicator for female sex
dx["Female"] = (dx.RIAGENDR == 2).astype(np.int)

# Recode the ethnic groups
dx["RIDRETH1"] = dx.RIDRETH1.replace({1: "MA", 2: "OH", 3: "NHW", 4: "NHB", 5: "OR"})

# Drop rows with any missing data in the variables of interest
dx = dx.dropna()
