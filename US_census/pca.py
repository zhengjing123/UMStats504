# Principal Component Analysis of the US census data for
# household income and age/sex structure.  The unit of
# analysis is a census tract.  See the IPUMS/NHGIS
# codebooks for more information about the variables.

import numpy as np
import pandas as pd
from get_data import dx, incvars, popvars, io, age_bands, plot_inc, plot_pop
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("census_pca.pdf")

dy = dx.copy()
dy = dy.dropna()

def do_pca(x, vars):

    dz = x.loc[:, vars]
    ldz = np.log(dz + 0.01)

    ldz_mean = ldz.mean(0)
    ldz_c = ldz - ldz_mean

    u, s, vt = np.linalg.svd(ldz_c, 0)
    v = vt.T

    v = pd.DataFrame(v, index=vars)

    return u, v, ldz_mean

# Do the PCA for the population data.
u_pop, v_pop, mn_pop = do_pca(dy, popvars)

# Plot the centroid and loadings for the population data.
for cx in range(4):
    for k in range(5):

        if cx == 0:
            vp = mn_pop.iloc[k:180:5]
            ylabel = "Mean"
        else:
            vp = v_pop.iloc[k:180:5, cx - 1]
            ylabel = "Component %d loading" % cx

        ylim = (-0.2, 0.2) if cx > 0 else None

        plot_pop(vp, ylabel, "%d population structure" % (1970 + k*10), ylim)
        pdf.savefig()

# Do the PCA for the income data.
u_inc, v_inc, mn_inc = do_pca(dy, incvars)

# Plot the loadings for the income data.
for cx in range(3):
    vp = v_inc.loc[io, cx].values
    title = "Income component loadings"
    ylabel = "Component %d loading" % (cx + 1)
    plot_inc(vp, ylabel, title)
    pdf.savefig()

pdf.close()

# Remove this if you aren't me
import os
os.system("cp census_pca.pdf ~kshedden")
