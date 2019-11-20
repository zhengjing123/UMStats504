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

df = df.loc[df.indinc>=0, :]
df["logindinc"] = np.log(1 + df.indinc)

pdf = PdfPages("ols_reg.pdf")

for bars in False, True:
    for rv in resp:

        fml = rv + " ~ 0 + bs(age, 5)*(female + urban) + logindinc + educ + wave"

        # Get the scale (dispersion) parameter estimate, used for plotting
        # vertical bars showing the range of the data around the mean.
        scale = sm.OLS.from_formula(fml, data=df).fit().scale
        usd = np.sqrt(scale)
        print(usd)

        y, x = patsy.dmatrices(fml, df, return_type='dataframe')
        di = x.design_info

        # Standardize the data before using LARS
        ymean = y.mean()
        y -= y.mean()
        xmean = x.mean(0)
        x -= xmean
        xsd = x.std(0)
        x /= xsd
        xnames = x.columns

        # Convert from dataframe to ndarray
        y = np.asarray(y)[:, 0]
        x = np.asarray(x)

        # Get the LARS paths
        a, b, coefs = lars_path(x, y)

        # Sort the rows according to the order that the variables enter
        coefs = pd.DataFrame(coefs, index=xnames)
        s = (coefs != 0).sum(1)
        ii = np.argsort(-s)
        coefsx.append(coefs.iloc[ii, :])

        # Prepare a dataframe for prediction
        dz = df.iloc[0:400, :].copy()
        age = np.linspace(18, 80, 100)
        dz["age"] = np.concatenate((age, age, age, age))
        a = np.ones(100)
        b = np.zeros(100)
        dz["female"] = np.concatenate((a, b, a, b))
        dz["urban"] = np.concatenate((a, a, b, b))
        dz["logindinc"] = df.logindinc.mean()
        dz["educ"] = df.educ.mean()
        dz["wave"] = df.wave.mean()
        dd = patsy.dmatrix(di, dz, return_type="dataframe")

        # Standardize the prediction data frame so it matches
        # the data in the model
        dd = (np.asarray(dd) - xmean.values)/ xsd.values

        # Get the predicted values
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

            for k, c in enumerate(["orange", "purple", "lime", "blue"]):

                plt.plot(age, pr[100*k:100*(k+1), j], '-', label=labels[k], color=c)
                if bars:
                    # Plot bars showing +/- 1*SD of the data
                    q = 5*(k+1)
                    qq = 100*k + q
                    plt.plot([age[q], age[q]], [pr[qq, j]-usd, pr[qq, j]+usd], '-', color='grey')
                    plt.plot([age[q]]*3, [pr[qq, j]-usd, pr[qq, j], pr[qq, j]+usd], 'o', ms=4, color=c)

            # Draw a legend
            ha, lb = plt.gca().get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.draw_frame(False)

            plt.xlabel("Age", size=15)
            plt.ylabel(rv, size=15)

            pdf.savefig()

pdf.close()

import os
os.system("cp ols_reg.pdf ~kshedden")
