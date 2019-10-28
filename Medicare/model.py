import numpy as np
import statsmodels.api as sm
import pandas as pd
import json

df = pd.read_csv("2017_utilization_reduced.csv.gz", dtype={"nppes_provider_zip": str})

df = df.loc[df.nppes_entity_code=="I", :]

# Include this as a covariate to proxy for practice size
df["total_npi_cnt"] = df.groupby("npi")["line_srvc_cnt"].transform(np.sum)
df["log_total_npi_cnt"] = np.log(df.total_npi_cnt)

# Include this as a covariate to proxy for zip code attributes
# related to population demographics
df["total_zip_cnt"] = df.groupby("nppes_provider_zip")["line_srvc_cnt"].transform(np.sum)
df["log_total_zip_cnt"] = np.log(df.total_zip_cnt)

fml = [
    "line_srvc_cnt ~ provider_type + log_total_npi_cnt + log_total_zip_cnt",
    "line_srvc_cnt ~ nppes_provider_state + log_total_npi_cnt + log_total_zip_cnt",
    "line_srvc_cnt ~ provider_type + nppes_provider_state + log_total_npi_cnt + log_total_zip_cnt"
]

with open("hcpcs_description.json") as fr:
    hcpcs = json.load(fr)

scale = []
icc = []
codes = []

mx = 2

for code, dg in df.groupby("hcpcs_code"):

    # Focus on the most common codes
    if dg.shape[0] < 25000:
        continue

    codes.append(code)
    print(code)

    model = sm.GEE.from_formula(fml[mx], family=sm.families.Poisson(), groups="nppes_provider_zip",
        cov_struct=sm.cov_struct.Exchangeable(), data=dg)
    result = model.fit(maxiter=1, first_dep_update=0)

    # Estimate the scale parameter as if this were a quasi-Poisson model
    scale.append(np.mean(result.resid_pearson**2))

    icc.append(result.cov_struct.dep_params)

hd = [hcpcs[c] for c in codes]

rslt = pd.DataFrame({"code": codes, "icc": icc, "scale": scale, "description": hd})
rslt = rslt.sort_values(by="icc")
rslt.to_csv("model.csv", index=None)
