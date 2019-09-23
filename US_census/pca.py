import numpy as np
import pandas as pd
from get_data import dx, incvars, popvars, io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("census_pca.pdf")

dy = dx.copy()

# Convert to proportions, since census tracts are not all the same size
for vars in popvars, incvars:
    dy.loc[:, vars] = dy.loc[:, vars].div(dy.loc[:, vars].sum(1), axis=0)

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

u_pop, v_pop, mn_pop = do_pca(dy, popvars)

xl = ["%.0f" % x for x in np.linspace(5, 85, 18)]
xl = xl + xl

for cx in range(4):
    for k in range(5):

        if cx == 0:
            vp = mn_pop.iloc[k:180:5]
            ylabel = "Mean"
        else:
            vp = v_pop.iloc[k:180:5, cx - 1]
            ylabel = "Component %d" % cx

        plt.clf()
        plt.figure(figsize=(8, 5))
        plt.axes([0.12, 0.15, 0.7, 0.8])
        plt.grid(True)
        m = vp.shape[0]
        plt.plot(vp.iloc[0:m//2], label="Female")
        plt.plot(vp.iloc[m//2:], label="Male")
        plt.title("%d" % (1970 + k*10))
        g = plt.gca().set_xticklabels(xl)
        for u in g:
            u.set_size(10)
            u.set_rotation(-90)
        ha, lb = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, loc="center right")
        leg.draw_frame(False)
        plt.xlabel("Age", size=15)
        plt.ylabel(ylabel, size=15)
        pdf.savefig()


u_inc, v_inc, mn_inc = do_pca(dy, incvars)

xl = []
for y in range(1970, 2011, 10):
    xl.extend(["", ""])
    xl.append(str(y))
    xl.extend(["", ""])

for cx in range(3):
    plt.clf()
    plt.grid(True)
    plt.title("Income component %d" % (cx + 1))
    vp = v_inc.loc[io, cx].values
    for j in range(5):
        plt.plot(range(5*j, 5*(j+1)), vp[5*j:5*(j+1)], '-o', color='black')
    plt.gca().set_xticks(range(25))
    g = plt.gca().set_xticklabels(xl)
    for u in g:
        u.set_size(10)
        u.set_rotation(-90)
    pdf.savefig()

pdf.close()

import os
os.system("cp census_pca.pdf ~kshedden")
