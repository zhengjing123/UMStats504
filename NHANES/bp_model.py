import numpy as np
import statsmodels.api as sm
from data_prep import dx
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_fit_by_age(result, fml):

    # Create a dataframe in which all variables are at the
    # reference level
    da = dx.iloc[0:100, :].copy()
    da["RIDAGEYR"] = np.linspace(18, 80, 100)
    da["RIDRETH1"] = "OH"

    plt.figure(figsize=(8, 5))
    plt.clf()
    plt.axes([0.1, 0.1, 0.66, 0.8])
    plt.grid(True)

    for female in 0, 1:
        for bmi in 22, 26, 31:

            db = da.copy()
            db.Female = female
            db.BMXBMI = bmi

            pr = result.predict(exog=db)

            la = "Female" if female == 1 else "Male"
            la += ", BMI=%.0f" % bmi
            plt.plot(da.RIDAGEYR, pr, '-', label=la)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.xlabel("Age (years)", size=15)
    plt.ylabel("BP (mm/Hg)", size=15)
    plt.title(fml, size=11)
    plt.title(fml, fontdict={"fontsize": 9})
    pdf.savefig()

def plot_fit_by_bmi(result, fml):

    # Create a dataframe in which all variables are at the
    # reference level
    da = dx.iloc[0:100, :].copy()
    da["BMXBMI"] = np.linspace(15, 35, 100)
    da["RIDRETH1"] = "OH"

    plt.figure(figsize=(8, 5))
    plt.clf()
    plt.axes([0.1, 0.1, 0.66, 0.8])
    plt.grid(True)

    for female in 0, 1:
        for age in 25, 50, 75:

            db = da.copy()
            db.Female = female
            db.RIDAGEYR = age

            pr = result.predict(exog=db)

            la = "Female" if female == 1 else "Male"
            la += ", age=%.0f" % age
            plt.plot(da.BMXBMI, pr, '-', label=la)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.xlabel("BMI (kg/m^2)", size=15)
    plt.ylabel("BP (mm/Hg)", size=15)
    plt.title(fml, fontdict={"fontsize": 9})
    pdf.savefig()

def plot_basis(xv, xmat, names, degf):

    ii = np.argsort(xv)
    xv = xv[ii]
    xmat = xmat[ii, :]

    plt.clf()
    plt.grid(True)

    for k in range(degf):
        vn = "bs(RIDAGEYR, 5)[%d]" % k
        pos = names.index(vn)
        plt.plot(xv, xmat[:, pos], '-', rasterized=True)

    plt.xlabel("Age (years)", size=15)
    plt.ylabel("Basis function value", size=15)
    plt.title("Spline basis for age (5 df)")
    pdf.savefig()


def plot_fit_by_age_race(result, fml):

    # Create a dataframe in which all variables are at the
    # reference level
    da = dx.iloc[0:100, :].copy()
    da["RIDAGEYR"] = np.linspace(18, 80, 100)
    da["BMXBMI"] = 22

    plt.figure(figsize=(8, 5))
    plt.clf()
    plt.axes([0.1, 0.1, 0.62, 0.8])
    plt.grid(True)

    for female in 0, 1:
        for race in "MA", "NHW", "NHB", "OR", "OH":

            db = da.copy()
            db.Female = female
            db.RIDRETH1 = race

            pr = result.predict(exog=db)

            la = "Female" if female == 1 else "Male"
            la += ", Ethnicity=%s" % race
            plt.plot(da.RIDAGEYR, pr, '-', label=la)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.xlabel("Age (years)", size=15)
    plt.ylabel("BP (mm/Hg)", size=15)
    plt.title(fml, size=11)
    plt.title(fml, fontdict={"fontsize": 9})
    pdf.savefig()



pdf = PdfPages("bp_model.pdf")

fml0 = "BPXSY1 ~ RIDAGEYR + BMXBMI + Female + RIDRETH1"
model0 = sm.OLS.from_formula(fml0, data=dx)
result0 = model0.fit()
plot_fit_by_age(result0, fml0)
plot_fit_by_bmi(result0, fml0)

fml1 = "BPXSY1 ~ (RIDAGEYR + BMXBMI) * Female + RIDRETH1"
model1 = sm.OLS.from_formula(fml1, data=dx)
result1 = model1.fit()
plot_fit_by_age(result1, fml1)
plot_fit_by_bmi(result1, fml1)

fml2 = "BPXSY1 ~ bs(RIDAGEYR, 5) + bs(BMXBMI, 5) + Female + RIDRETH1"
model2 = sm.OLS.from_formula(fml2, data=dx)
result2 = model2.fit()
plot_fit_by_age(result2, fml2)
plot_fit_by_bmi(result2, fml2)

plot_basis(dx.RIDAGEYR.values, result2.model.exog, result2.model.exog_names, 5)

aic = np.zeros((15, 15)) + np.nan
for age_df in range(3, 15):
    for bmi_df in range(3, 15):
        fml3 = "BPXSY1 ~ bs(RIDAGEYR, age_df) + bs(BMXBMI, bmi_df) + Female + RIDRETH1"
        model3 = sm.OLS.from_formula(fml3, data=dx)
        result3 = model3.fit()
        aic[age_df, bmi_df] = result3.aic

plt.clf()
plt.imshow(aic, interpolation="nearest")
plt.xlabel("BMI df", size=15)
plt.ylabel("Age df", size=15)
plt.colorbar()
plt.xlim(3, 14)
plt.ylim(3, 14)
pdf.savefig()

fml4 = "BPXSY1 ~ bs(RIDAGEYR, 6) + bs(BMXBMI, 5) + Female + RIDRETH1"
model4 = sm.OLS.from_formula(fml4, data=dx)
result4 = model4.fit()
plot_fit_by_age(result4, fml4)
plot_fit_by_bmi(result4, fml4)

fml5 = "BPXSY1 ~ RIDAGEYR + I(RIDAGEYR**2) + I(RIDAGEYR**3) + BMXBMI + I(BMXBMI**2) + I(BMXBMI**3) + Female + RIDRETH1"
model5 = sm.OLS.from_formula(fml5, data=dx)
result5 = model5.fit()
plot_fit_by_age(result5, fml5)
plot_fit_by_bmi(result5, fml5)

fml6 = "BPXSY1 ~ RIDAGEYR + I(RIDAGEYR**2) + I(RIDAGEYR**3) + I(RIDAGEYR**4) + I(RIDAGEYR**5) + BMXBMI + I(BMXBMI**2) + I(BMXBMI**3) + I(BMXBMI**4) + I(BMXBMI**5) + Female + RIDRETH1"
model6 = sm.OLS.from_formula(fml6, data=dx)
result6 = model6.fit()
plot_fit_by_age(result6, fml6)
plot_fit_by_bmi(result6, fml6)

fml7 = "BPXSY1 ~ (bs(RIDAGEYR, 6) + bs(BMXBMI, 5)) * Female + RIDRETH1"
model7 = sm.OLS.from_formula(fml7, data=dx)
result7 = model7.fit()
plot_fit_by_age(result7, fml7)
plot_fit_by_bmi(result7, fml7)

fml8 = "BPXSY1 ~ bs(RIDAGEYR, 5) * BMXBMI * Female + RIDRETH1"
model8 = sm.OLS.from_formula(fml8, data=dx)
result8 = model8.fit()
plot_fit_by_age(result8, fml8)
plot_fit_by_bmi(result8, fml8)

fml9 = "BPXSY1 ~ bs(RIDAGEYR, 5) * bs(BMXBMI, 4) * Female + RIDRETH1"
model9 = sm.OLS.from_formula(fml9, data=dx)
result9 = model9.fit()
plot_fit_by_age(result9, fml9)
plot_fit_by_bmi(result9, fml9)

fml10 = "BPXSY1 ~ bs(RIDAGEYR, 5) + bs(BMXBMI, 4) + Female * RIDRETH1"
model10 = sm.OLS.from_formula(fml10, data=dx)
result10 = model10.fit()
plot_fit_by_age_race(result10, fml10)

fml11 = "BPXSY1 ~ bs(BMXBMI, 4) + bs(RIDAGEYR, 5) + RIDAGEYR * Female * RIDRETH1"
model11 = sm.OLS.from_formula(fml11, data=dx)
result11 = model11.fit()
plot_fit_by_age_race(result11, fml11)

pdf.close()

import os
os.system("cp bp_model.pdf ~kshedden")
