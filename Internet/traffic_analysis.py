import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
from scipy.special import gammaln
from scipy.stats import kendalltau

# The date to process
dt = "2012-04-01"

# Create a set of graphs in one pdf file.
pdf = PdfPages("internet_%s.pdf" % dt)

# Load the traffic statistics (one record per minute
# within a day).
df = pd.read_csv("traffic_stats_%s.csv" % dt)

# Rename the columns
cname = {"Traffic": "Total traffic", "UDP": "UDP", "TCP": "TCP",
         "Sources": "Unique sources"}


# Plot each column of data (total traffic, distinct sources,
# UDP traffic, TCP traffic).
for i in range(4):
    plt.clf()
    plt.axes([0.17, 0.1, 0.75, 0.8])
    plt.plot(df.iloc[:, 2+i])
    plt.grid(True)
    plt.xlabel("Minutes", size=15)
    plt.ylabel(cname[df.columns[2+i]], size=15)
    pdf.savefig()

# Plot the quantile functions for each traffic statistic.
for i in range(4):
    plt.clf()
    plt.axes([0.17, 0.1, 0.75, 0.8])
    pp = np.linspace(0, 1, df.shape[0])
    plt.plot(pp, np.sort(df.iloc[:, 2+i]))
    plt.grid(True)
    plt.xlabel("Probability point", size=15)
    plt.ylabel(cname[df.columns[2+i]] + " quantile", size=15)
    pdf.savefig()

# Plot the entropy data
ent = pd.read_csv("entropy_minute_%s.csv" % dt, header=None)
plt.clf()
plt.plot(ent)
plt.grid(True)
plt.xlabel("Minutes", size=15)
plt.ylabel("Destination port entropy", size=15)
pdf.savefig()


# Return ICC for blocks of length m minutes.
def anova(x, m):
    y = np.reshape(x, (-1, m)) # hours by minutes
    w = y.var(1).mean()
    b = y.mean(1).var()
    t = y.var()
    return b/t

print("            ICC")
print("       60m       240m")
for x in df.Traffic, df.Sources, df.TCP, df.UDP:
    x = np.asarray(x)
    print("%10.3f %10.3f" % (anova(x, 60), anova(x, 240)))


# Calculate the Kendall's tau autocorrelation for z at lag k.
def kt(z, k):
    n = len(z)
    x = z[0:n-k]
    y = z[k:n]
    return kendalltau(x, y).correlation

# Calculate all the Kendall's tau autocorrelation values (for
# each data series, at different lags).
# f[j, k, t] is the tau-autocorrelations for series j (e.g.
# total traffic) differenced to order k at lag t.
f = []
for j in range(4):
    v = []
    # Difference the series at various orders
    for diffo in 0, 1, 2:
        z = df.iloc[:, 2+j]
        if diffo > 0:
            z = np.diff(z, diffo)
        u = []
        for k in range(1, 241):
            u.append(kt(z, k))
        u = np.asarray(u)
        v.append(u)
    f.append(v)
f = np.asarray(f)

# Plot the tau autocorrelations as time series
# k = order of differencing
for k in range(3):
    plt.clf()
    plt.grid(True)
    for j in range(4):
        x = np.arange(k, 240)
        plt.plot(x, f[j, k, k:], label=cname[df.columns[2+j]])
    plt.xlabel("Lag minutes", size=15)
    plt.ylabel("Tau-autocorrelation", size=15)
    plt.title("Order %d differences" % k)
    ha,lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "upper center", ncol=4)
    leg.draw_frame(False)
    pdf.savefig()

# Get null quantiles for Kendall's tau
def kt0():
    z = []
    for j in range(100):
        y = np.random.normal(size=24*60)
        v = [kt(y, k) for k in range(1, 241)]
        v = np.asarray(v)
        z.append(v)
    z = np.asarray(z)
    z.sort(1)
    z.sort(0)
    return z

z0 = kt0()


# Plot the tau autocorrelations as quantile functions relative to null
# k = order of differencing
for k in range(3):

    for j in range(4):
        y = f[j, k, k:]
        plt.clf()
        plt.grid(True)
        mn = y.min()
        mx = y.max()
        plt.plot([mn, mx], [mn, mx], '-', color='orange')
        plt.plot(z0[5, k:], np.sort(y), color='grey')
        plt.plot(z0[50, k:], np.sort(y), color='blue')
        plt.plot(z0[95, k:], np.sort(y), color='grey')
        plt.ylabel("Observed tau autocorrelation", size=15)
        plt.xlabel("Null tau autocorrelation", size=15)
        plt.title(cname[df.columns[2+j]] + " order %d differences" % k)
        pdf.savefig()

# Calculate the binomial coefficient "n choose m".
def bincoeff(n, m):
    return np.exp(gammaln(n + 1) - gammaln(n - m + 1) - gammaln(m + 1))


# Calculate the l-moment of order k, for the data vector x.
# The order (k) must be equal to 1, 2, 3, or 4.
def lmoment(x, k):

    n = len(x)
    x = np.sort(x)
    ii = np.arange(1, n + 1, dtype=np.float64)

    if k == 1:
        return np.mean(x)
    elif k == 2:
        c = 2*ii - n - 1
        c /= bincoeff(n, 2)
        c /= 2
        return np.dot(x, c)
    elif k == 3:
        c = bincoeff(ii-1, 2) - 2*(ii-1)*(n-ii) + bincoeff(n-i, 2)
        c /= bincoeff(n, 3)
        c /= 3
        return np.dot(x, c)
    elif k == 4:
        c = bincoeff(ii-1, 3) - 3*bincoeff(ii-1, 2)*(n-ii) + 3*(ii-1)*bincoeff(n-ii, 2) - bincoeff(n-ii, 3)
        c /= bincoeff(n, 4)
        c /= 4
        return np.dot(x, c)

# Calculate the first four L-moments of all outcomes,
# for each hour within a day.
lmom = np.zeros((24, 16))
ii = 0
for hour in range(24):
    x = df.iloc[ii:ii+60, 2:].values
    ii += 60
    u = []
    for k in range(4):
        for j in range(1, 5):
            u.append(lmoment(x[:, k], j))
    u = np.asarray(u)
    lmom[hour, :] = u

# Plot each L moment (calculated by hour) as a function of time.
jj = 0
for k in range(4):
    for j in range(4):
        plt.clf()
        plt.axes([0.2, 0.1, 0.75, 0.8])
        plt.grid(True)
        plt.plot(lmom[:, jj])
        plt.xlabel("Hour", size=15)
        plt.ylabel("L-moment of order %d" % (j + 1), size=15)
        plt.title(cname[df.columns[2 + k]])
        pdf.savefig()
        jj += 1

# Calculate the Hurst index using means and variances.
def hurst(x):
    z = []
    # m is the block size
    for m in 15, 30, 60:
        y = np.reshape(x, (-1, m))
        v = y.mean(1).var()
        z.append([m, v])
    z = np.log(np.asarray(z))
    c = np.cov(z.T)
    b = c[0, 1] / c[0, 0]
    return b/2 + 1

# Calculate the Hurst index using absolute values.
def hurstabs(x):
    z = []
    # m is the block size
    for m in 15, 30, 60:
        y = np.reshape(x, (-1, m))
        v = np.mean(np.abs(y.mean(1)))
        z.append([m, v])
    z = np.log(np.asarray(z))
    c = np.cov(z.T)
    b = c[0, 1] / c[0, 0]
    return b + 1

# Calculate the Hurst index two ways, for all four data series.
print("Hurst parameters:")
for j, x in enumerate([df.Traffic, df.Sources, df.UDP, df.TCP]):
    print(df.columns[2+j])
    for diffo in range(3):
        x = np.asarray(x, dtype=np.float64)
        z = np.diff(x, diffo)
        z = z[0:60*(len(z)//60)]
        z -= z.mean()
        print("    %4d %7.3f %7.3f" % (diffo, hurst(z), hurstabs(z)))

# Use regularized regression to study the autoregressive structure of the
# traffic and unique sources series.

from numpy.lib.stride_tricks import as_strided
import statsmodels.api as sm

labs = ["Traffic", "Sources", "UDP", "TCP"]
for j,x in enumerate([df.Traffic, df.Sources, df.UDP, df.TCP]):

    x = np.asarray(x)
    x = np.log(x)
    x -= x.mean()

    # This is a trick to lag the data without copying
    z = as_strided(x, shape=(len(x)-30, 30), strides=(8, 8))

    y = z[:, 0]
    x = z[:, 1:30]

    # Use four approaches to regression
    for jj in 0, 1, 2, 3:

        params = []

        if jj == 0:
            title = "OLS"
            result = sm.OLS(y, x).fit()
            params.append(result.params)
        elif jj == 1:
            title = "Ridge"
            for alpha in 0.001, 0.01, 0.1:
                result = sm.OLS(y, x).fit_regularized(alpha=alpha, L1_wt=0)
                params.append(result.params)
        elif jj == 2:
            title = "Lasso"
            for alpha in 0.001, 0.01, 0.1:
                result = sm.OLS(y, x).fit_regularized(alpha=alpha, L1_wt=1)
                params.append(result.params)
        elif jj == 3:
            title = "Elastic net"
            for alpha in 0.00001, 0.001, 0.1:
                result = sm.OLS(y, x).fit_regularized(alpha=alpha, L1_wt=0.1)
                params.append(result.params)

        plt.clf()
        plt.title(labs[j] + " " + title)
        for p in params:
            plt.plot(np.arange(1, len(p)+1), p)
        plt.xlabel("Lag", size=15)
        plt.ylabel("Coefficient", size=15)
        plt.grid(True)
        plt.ylim(-1, 1)
        pdf.savefig()

pdf.close()
