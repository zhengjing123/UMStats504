import pandas as pd
import numpy as np

# Download some of the NHANES data files
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.XPT
# wget https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.XPT

xpt_files = ["DEMO_I.XPT", "BPX_I.XPT", "BMX_I.XPT"]

vars = [
    ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1"],
    ["SEQN", "BPXSY1", "BPXDI1"],
    ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXLEG", "BMXARMC"],
]

da = []

for idf, fn in enumerate(xpt_files):

    df = pd.read_sas(fn)
    df = df.loc[:, vars[idf]]
    da.append(df)


dx = pd.merge(da[0], da[1], left_on="SEQN", right_on="SEQN")
dx = pd.merge(dx, da[2], left_on="SEQN", right_on="SEQN")

dx["Female"] = (dx.RIAGENDR == 2).astype(np.int)

dx["RIDRETH1"] = dx.RIDRETH1.replace({1: "MA", 2: "OH", 3: "NHW", 4: "NHB", 5: "OR"})

dx = dx.dropna()