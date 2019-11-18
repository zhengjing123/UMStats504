import pandas as pd
import patsy
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import lars_path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data import df

resp = ["d3kcal", "d3carbo", "d3fat", "d3protn"]
coefsx = []

pdf = PdfPages("ols_reg.pdf")

for bars in False, True:
    for rv in resp:

        fml = rv + " ~ 0 + bs(age, 5)*(female + urban) + indinc + educ + wave"

        scale = sm.OLS.from_formula(fml, data=df).fit().scale
        usd = np.sqrt(scale)
        print(usd)

        y, x = patsy.dmatrices(fml, df, return_type='dataframe')
        di = x.design_info

        ymean = y.mean()
        y -= y.mean()
        xmean = x.mean(0)
        x -= xmean
        xsd = x.std(0)
        x /= xsd
        xnames = x.columns

        y = np.asarray(y)[:, 0]
        x = np.asarray(x)

        a, b, coefs = lars_path(x, y)

        coefs = pd.DataFrame(coefs, index=xnames)
        s = (coefs != 0).sum(1)
        ii = np.argsort(-s)
        coefsx.append(coefs.iloc[ii, :])

        dz = df.iloc[0:400, :].copy()
        age = np.linspace(18, 80, 100)
        dz["age"] = np.concatenate((age, age, age, age))
        a = np.ones(100)
        b = np.zeros(100)
        dz["female"] = np.concatenate((a, b, a, b))
        dz["urban"] = np.concatenate((a, a, b, b))
        dz["indinc"] = df.indinc.mean()
        dz["educ"] = df.educ.mean()
        dz["wave"] = df.wave.mean()

        dd = patsy.dmatrix(di, dz, return_type="dataframe")

        dd = (np.asarray(dd) - xmean.values)/ xsd.values
        pr = ymean.iloc[0] + np.dot(dd, coefs)

        labels = ["Urban female", "Urban male", "Rural female", "Rural male"]

        plt.figure(figsize=(7, 5))

        for j in range(1, 20):

            ii = np.abs(coefs.iloc[:, j]) > 1e-5
            xx = x[:, ii]
            u, s, vt = np.linalg.svd(xx, 0)
            #print(max(s)/min(s))

            plt.clf()
            plt.axes([0.12, 0.1, 0.65, 0.8])
            plt.grid(True)
            plt.plot(age, pr[0:100, j], '-', color='orange', label=labels[0])
            if bars:
                plt.plot([age[5], age[5]], [pr[5, j]-usd, pr[5, j]+usd], '-', color='grey')
                plt.plot([age[5]]*3, [pr[5, j]-usd, pr[5, j], pr[5, j]+usd], 'o', ms=4, color='orange')
            plt.plot(age, pr[100:200, j], '-', label=labels[1], color='purple')
            if bars:
                plt.plot([age[10], age[10]], [pr[110, j]-usd, pr[110, j]+usd], '-', color='grey')
                plt.plot([age[10]]*3, [pr[110, j]-usd, pr[110, j], pr[110, j]+usd], 'o', ms=4, color='purple')
            plt.plot(age, pr[200:300, j], '-', label=labels[2], color='lime')
            if bars:
                plt.plot([age[15], age[15]], [pr[215, j]-usd, pr[215, j]+usd], '-', color='grey')
                plt.plot([age[15]]*3, [pr[215, j]-usd, pr[215, j], pr[215, j]+usd], 'o', ms=4, color='lime')
            plt.plot(age, pr[300:400, j], '-', label=labels[3], color='blue')
            if bars:
                plt.plot([age[20], age[20]], [pr[320, j]-usd, pr[320, j]+usd], '-', color='grey')
                plt.plot([age[20]]*3, [pr[320, j]-usd, pr[320, j], pr[320, j]+usd], 'o', ms=4, color='blue')

            ha, lb = plt.gca().get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.draw_frame(False)
            plt.xlabel("Age", size=15)
            plt.ylabel(rv, size=15)

            pdf.savefig()

pdf.close()

import os
os.system("cp ols_reg.pdf ~kshedden")
