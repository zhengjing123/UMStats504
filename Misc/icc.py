# Plot to demonstrate ICC (intraclass correlation)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("icc.pdf")

# Number of groups to show
ng = 20

# True ICC
for icc in 0, 0.2, 0.4, 0.6, 0.8:

    # Sample size
    for n in 10000, 20:

        # The ICC is t / (t + 1), where t is the between-group variance
        # and 1 is the within-group variance.  Here we solve for t.
        t = icc / (1 - icc)

        x = np.random.normal(size=(n, ng))
        x += np.sqrt(t) * np.random.normal(size=ng)

        # Rescale so the marginal variance is 1.
        x /= np.sqrt(1 + t)

        plt.clf()
        plt.boxplot(x, showfliers=False)
        plt.xlabel("Group", size=15)
        plt.ylabel("Residual", size=15)
        plt.title("ICC=%.1f, n=%d" % (icc, n))
        pdf.savefig()

pdf.close()

import os
os.system("cp icc.pdf ~kshedden")