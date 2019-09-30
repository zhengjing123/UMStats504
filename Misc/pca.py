import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

m = 100

means = [
    0.5*np.ones(m),
    0.5*np.linspace(-1, 1, m),
    0.5*np.linspace(-1, 1, m)**2,
    0.2*np.sin(np.linspace(-7, 7, m)),
]

pca = [
    np.ones(m),
    np.linspace(-1, 1, m),
    np.linspace(-1, 1, m)**2,
    np.linspace(-1, 1, m)**2 - 0.5,
    np.sin(np.linspace(-7, 7, m)),
]

for j in range(len(pca)):
    pca[j] /= np.sqrt(np.sum(pca[j]**2))

pdf = PdfPages("pca.pdf")

for j in range(len(means)):
    for k in range(len(pca)):

        plt.clf()
        plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.grid(True)
        plt.plot(means[j], '-', color='orange', label='Mean')
        plt.plot(pca[k], '-', color='purple', label='Loadings')
        ha, lb = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(ha, lb, "center right")
        leg.draw_frame(False)
        plt.ylim(-1, 1)
        pdf.savefig()

        plt.clf()
        plt.grid(True)
        plt.plot(means[j], '-', lw=4, color='orange')
        for i in range(10):
            u = np.random.uniform(-1, 1)
            plt.plot(means[j] + u*pca[k], '-', color='grey')
        pdf.savefig()

pdf.close()

import os
os.system("cp pca.pdf ~kshedden")