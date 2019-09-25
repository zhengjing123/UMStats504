# Canonical Correlation Analysis of the US census data for
# household income and age/sex structure.  The unit of
# analysis is a census tract.

import numpy as np
import pandas as pd
from get_data import dx, incvars, popvars, io, plot_pop, plot_inc
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("census_cancorr.pdf")

# Do a PC reduction to focus the CCA on the more variable
# directions.
q_i = 5
q_p = 5

dy = dx.copy()
dy = dy.dropna()

# Log transform and standardize within years
for vars in incvars, popvars:
    dy.loc[:, vars] = np.log(0.01 + dy.loc[:, vars])
    dy.loc[:, vars] -= dy.loc[:, vars].mean(0)

inc_dat = dy.loc[:, incvars]
pop_dat = dy.loc[:, popvars]

u_i, s_i, vt_i = np.linalg.svd(inc_dat, 0)
inc_dat1 = u_i[:, 0:q_i]

u_p, s_p, vt_p = np.linalg.svd(pop_dat, 0)
pop_dat1 = u_p[:, 0:q_p]

cc = sm.multivariate.CanCorr(inc_dat1, pop_dat1)

# Check that canonical correlation is doing what it is supposed to do
for j in range(5):
    y1 = np.dot(inc_dat1, cc.y_cancoef[:, j])
    x1 = np.dot(pop_dat1, cc.x_cancoef[:, j])
    assert(np.abs(np.corrcoef(x1, y1)[0, 1] - cc.cancorr[j]) < 1e-5)


inc_coef = np.dot(vt_i.T[:, 0:q_i], cc.y_cancoef / s_i[0:q_i])
pop_coef = np.dot(vt_p.T[:, 0:q_p], cc.x_cancoef / s_p[0:q_p])

inc_coef = pd.DataFrame(inc_coef, index=incvars)
pop_coef = pd.DataFrame(pop_coef, index=popvars)

inc_coef = inc_coef.div(np.sqrt(np.sum(inc_coef**2, 0)), axis=1)
pop_coef = pop_coef.div(np.sqrt(np.sum(pop_coef**2, 0)), axis=1)


for j in range(3):
    for k in range(5):
        ylabel = "Component %d loading" % (j + 1)
        title = str(1970 + 10*k) + " population structure"
        plot_pop(pop_coef.iloc[k:180:5, j], ylabel, title, (-0.3, 0.3))
        pdf.savefig()
    ylabel = "Component %d loading" % (j + 1)
    plot_inc(inc_coef.iloc[:, j], ylabel, "Household income loadings")
    pdf.savefig()

pdf.close()

import os
os.system("cp census_cancorr.pdf ~kshedden")
