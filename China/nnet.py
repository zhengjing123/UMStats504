from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data import df

# Create a log income variable
df = df.loc[df.indinc>=0, :]
df["logindinc"] = np.log(1 + df.indinc)

# Response variables
resp = ["d3kcal", "d3carbo", "d3fat", "d3protn"]

# Drop variables that we don't need
xv = ["age", "female", "urban", "logindinc", "educ"]
dx = df.loc[:, resp + xv]

# Center the data
xmean = dx.loc[:, xv].mean(0)
xmat = dx.loc[:, xv] - xmean
xsd = xmat.std(0)
xmat /= xsd
xmat = xmat.values

pdf = PdfPages("nnet.pdf")

ages = np.linspace(18, 80, 100)

def runvar(rv):

    yvec = dx.loc[:, rv].values
    ymean = yvec.mean()
    yvec = yvec - ymean
    ysd = yvec.std()
    yvec /= ysd

    model = Sequential()
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss="mse", optimizer="sgd")

    model.fit(xmat, yvec, epochs=5, batch_size=32)

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.axes([0.11, 0.12, 0.68, 0.8])
    plt.grid(True)

    for female in 0, 1:
        for urban in 0, 1:

            xmatp = np.zeros((100, 5))
            xmatp[:, 0] = ages
            xmatp[:, 1] = female
            xmatp[:, 2] = urban
            xmatp[:, 3] = xmean[3]
            xmatp[:, 4] = xmean[4]
            xmatp = (xmatp - xmean.values) / xsd.values

            yp = model.predict(xmatp)
            yp = ymean + ysd * yp

            # A label for the line we will add here
            label = [["rural", "urban"][urban], ["male", "female"][female]]
            label = "%s %s" % tuple(label)

            plt.plot(ages, yp, label=label)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.ylabel(rv, size=15)
    plt.xlabel("Age", size=15)

    pdf.savefig()

for rv in resp:
    runvar(rv)

pdf.close()

import os
os.system("cp nnet.pdf ~kshedden")
