"""
Use Poisson regression to decompose the crime rate data into
a long-term time trend, annual, and weekly cycles.  The crime
rates are modeled as the number of crimes per day of a given
type.  In the "spacetime" model the data are disaggregated
by spatial region.  In the non-spacetime models, the data are
aggregated to the whole city of Chicago.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import patsy
from chicago_data import cdat

# If False, sum over the communities to get a city-wide total for each
# crime type x day.
spacetime = False

# 2001 and 2002 data look incorrect
cdat = cdat.loc[cdat.Year >= 2003, :]

# Recode day of week with text labels
cdat.loc[:, "DayOfWeek"] = cdat.DayOfWeek.replace({0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr", 5: "Sa", 6: "Su"})

# Construct a Fourier basis for day of year
def fb(xm):
    for q in 1, 2:
        xm.loc[:, "DayOfYear_sin_%d" % q] = np.sin(2 * np.pi * q * xm.DayOfYear / 365.25)
        xm.loc[:, "DayOfYear_cos_%d" % q] = np.cos(2 * np.pi * q * xm.DayOfYear / 365.25)

if not spacetime:
    # Collapse over spatial units
    a = {"Num": np.sum, "DayOfWeek": "first"}
    cdat = cdat.groupby(["PrimaryType", "Year", "DayOfYear"]).agg(a)
    cdat = cdat.reset_index()

# Build the Fourier basis functions
fb(cdat)

# A few values of CommunityArea are missing.
cdat = cdat.dropna()

# Model terms for the day of year Fourier basis
doy = "DayOfYear_sin_1 + DayOfYear_cos_1 + DayOfYear_sin_2 + DayOfYear_cos_2"

if spacetime:
    pdf = PdfPages("chicago_timetrends_spacetime.pdf")
    fml = "Num ~ bs(Year, 4) + C(DayOfWeek) + C(CommunityArea) + " + doy
else:
    pdf = PdfPages("chicago_timetrends_time.pdf")
    fml = "Num ~ bs(Year, 4) + C(DayOfWeek) + " + doy

opts = {"DayOfWeek": {"lw": 3}, "CommunityArea": {"color": "grey", "lw": 2, "alpha": 0.5}}

# Get the empirical mean and variance of the response variable
# in a series of fitted value strata.
def get_meanvar(model, result):
    c = pd.qcut(result.fittedvalues, np.linspace(0.1, 0.9, 9))
    dd = pd.DataFrame({"c": c, "y": model.endog})
    mv = []
    for k,v in dd.groupby("c"):
        mv.append([v.y.mean(), v.y.var()])
    mv = np.asarray(mv)
    return mv

# Histogram of counts
def plot_count_hist(model):
    plt.clf()
    plt.axes([0.15, 0.1, 0.8, 0.8])
    plt.hist(model.endog)
    plt.xlabel(pt, size=15)
    plt.ylabel("Frequency", size=15)
    pdf.savefig()

# Plot the empirical mean/variance relationship
def plot_empirical_meanvar(pt, mv):
    plt.clf()
    plt.title(pt)
    plt.grid(True)
    plt.plot(mv[:, 0], mv[:, 1], 'o', color='orange')
    mx = mv.max()
    mx *= 1.04
    b = np.dot(mv[:, 0], mv[:, 1]) / np.dot(mv[:, 0], mv[:, 0])
    plt.plot([0, mx], [0, b*mx], color='purple')
    plt.plot([0, mx], [0, mx], '-', color='black')
    plt.xlim(0, mx)
    plt.ylim(0, mx)
    plt.xlabel("Mean", size=15)
    plt.ylabel("Variance", size=15)
    pdf.savefig()

# Plot the fitted means curves for two variables, holding the others fixed.
def plot_fit(pt, cdat, dz):

    for tp in "Year", "DayOfYear":
        for vn in "DayOfWeek", "CommunityArea":

            if vn == "CommunityArea" and not spacetime:
                continue

            # Create a data set so we can plot the fitted regression function
            # E[Y|X=x] where certain components of x are held fixed and others
            # are varied systematically.
            p = 100
            dp = cdat.iloc[0:p, :].copy()
            dp.Year = 2015
            dp.DayOfWeek = "Su"
            dp.CommunityArea = 1
            dp.DayOfYear = 180
            fb(dp)

            if tp == "Year":
                dp.Year = np.linspace(2003, 2018, p)
            elif tp == "DayOfYear":
                dp.DayOfYear = np.linspace(1, 365, p)
                fb(dp)

            plt.clf()

            if vn == "DayOfWeek":
                plt.axes([0.15, 0.1, 0.72, 0.8])

            plt.grid(True)

            for u in dz[vn].unique():
                dp[vn] = u
                pr = result.predict(exog=dp)
                if vn == "DayOfWeek":
                    plt.plot(dp[tp], pr, '-', label=u, **opts[vn])
                else:
                    plt.plot(dp[tp], pr, '-', **opts[vn])

            plt.xlabel(tp, size=14)
            plt.ylabel("Expected number of reports per day", size=16)
            plt.title(pt)

            if vn == "DayOfWeek":
                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "center right")
                leg.draw_frame(False)

            pdf.savefig()

            # Plot with confidence bands
            di = model.data.design_info
            cm = result.scale * result.cov_params()

            if vn == "CommunityArea":
                continue

            # Plot with error bands
            plt.clf()
            plt.axes([0.15, 0.1, 0.72, 0.8])
            plt.grid(True)
            for u in dz[vn].unique():
                dp[vn] = u
                pr = result.predict(exog=dp)
                lpr = np.log(pr)
                xm = patsy.dmatrix(di, dp)
                dm = np.dot(xm, np.dot(cm, xm.T))
                se = np.sqrt(np.diag(dm))

                plt.plot(dp[tp], pr, '-', label=u, **opts[vn])
                lcb = np.exp(lpr - 2.8*se)
                ucb = np.exp(lpr + 2.8*se)
                plt.fill_between(dp[tp], lcb, ucb, color='lightgrey')

            plt.xlabel(tp, size=14)
            plt.ylabel("Expected number of reports per day", size=16)
            plt.title(pt)
            ha, lb = plt.gca().get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.draw_frame(False)
            pdf.savefig()

def plot_resid_acor(pt, result):

    if spacetime:
        # Get the average autocorrelation over community areas
        dm = result.model.data.frame
        da = pd.DataFrame({"resid": result.resid_pearson, "Date": dm.Date, "CommunityArea": dm.CommunityArea})
        da = da.sort_values(by=["CommunityArea", "Date"])
        a, m = 0, 0
        for k, dv in da.groupby("CommunityArea"):
            a += sm.tsa.acf(dv.resid)
            m += 1
        a /= m
    else:
        a = sm.tsa.acf(result.resid_pearson)

    plt.clf()
    plt.grid(True)
    plt.plot(a)
    plt.title(pt)
    plt.xlabel("Lag (days)", size=15)
    plt.ylabel("Autocorrelation", size=15)
    plt.ylim(-0.2, 1)
    pdf.savefig()

def plot_resid_week_pca(pt, result):

    if spacetime:
        dm = pd.DataFrame(result.model.data.frame)
        dm["resid"] = result.resid_pearson.values
        dm = dm.sort_values(by=["CommunityArea", "Date"])

        da = []
        for k, dv in dm.groupby("CommunityArea"):
            dr = pd.DataFrame({"resid": dv.resid,
                               "DayOfWeek": dv.DayOfWeek})
            ii = np.flatnonzero(dr.DayOfWeek == "Mo")[0]
            dr = dr.iloc[ii:, :]
            ii = np.flatnonzero(dr.DayOfWeek == "Su")[-1]
            dr = dr.iloc[:ii+1, :]
            rs = np.reshape(dr.resid.values, (-1, 7))
            da.append(rs)

        da = np.concatenate(da)
        u, s, vt = np.linalg.svd(da, 0)
        v = vt.T
    else:
        dm = result.model.data.frame
        dr = pd.DataFrame({"resid": result.resid_pearson.values,
                           "DayOfWeek": dm.DayOfWeek})

        ii = np.flatnonzero(dr.DayOfWeek == "Mo")[0]
        dr = dr.iloc[ii:, :]
        ii = np.flatnonzero(dr.DayOfWeek == "Su")[-1]
        dr = dr.iloc[:ii+1, :]

        rs = np.reshape(dr.resid.values, (-1, 7))

        u, s, vt = np.linalg.svd(rs, 0)
        v = vt.T

    plt.clf()
    plt.axes([0.14, 0.1, 0.75, 0.8])
    plt.grid(True)
    for k in range(3):
        plt.plot(v[:, k], label=str(k+1))
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc="center right")
    leg.draw_frame(False)
    plt.gca().set_xticks(range(7))
    plt.gca().set_xticklabels(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    plt.title(pt)
    plt.ylabel("Loading", size=15)
    pdf.savefig()

# Loop over primary crime types, create a Poisson model for each type
for pt, dz in cdat.groupby("PrimaryType"):

    # Create and fit the model
    model = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=dz)
    result = model.fit(scale='X2')

    # Estimate the scale as if this was a quasi-Poisson model
    print("%-20s %5.1f" % (pt, result.scale))

    mv = get_meanvar(model, result)

    plot_count_hist(model)
    plot_empirical_meanvar(pt, mv)
    plot_fit(pt, cdat, dz)
    plot_resid_acor(pt, result)
    plot_resid_week_pca(pt, result)

pdf.close()
