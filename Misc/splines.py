# Construct graphs of various spline basis sets.

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats.distributions import norm

n = 1000

pdf = PdfPages("splines.pdf")

def plot(fml, x, title):
    x.sort()
    y = np.random.normal(size=n) # Needed, but not used
    df = pd.DataFrame({"x": x, "y": y})
    m = sm.OLS.from_formula("y ~ 0 + " + fml, data=df)
    xm = m.exog

    # Check that the anolamlous looking final basis function
    # has a partner (via reflection) that is already in the span
    # of the other variables.
    u, _, _ = np.linalg.svd(xm, 0)
    z = xm[:, -1]
    v = z - np.dot(u, np.dot(u.T, z))
    assert(np.max(np.abs(v)) < 1e-10)

    plt.clf()
    plt.grid(True)
    for j in range(xm.shape[1]):
        plt.plot(x, xm[:, j], '-')
    plt.xlabel("x", size=15)
    plt.ylabel("Spline value", size=15)
    plt.title(title)
    pdf.savefig()

x = np.linspace(-1, 1, n)
for k in range(3, 11):
    fml = "bs(x, %d)" % k
    title = "df=%d, uniform" % k
    plot(fml, x, title)

p = np.linspace(0.001, 0.999, n)
x = -np.log(1 - p)
for k in range(3, 11):
    fml = "bs(x, %d)" % k
    title = "df=%d, exponential" % k
    plot(fml, x, title)

x = norm.ppf(p)
for k in range(3, 11):
    fml = "bs(x, %d)" % k
    title = "df=%d, Gaussian" % k
    plot(fml, x, title)

pdf.close()
