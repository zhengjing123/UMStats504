library(lme4)

# Number of people
n = 2000

# Number of measurements per person
m = 4

# Subject id's
a = array(m, n)
idx = rep(1:n, a)

# Generate a binary covariate
grp0 = sample(1:2, n, replace=T)
grp = rep(grp0, a)

# Generate random effects for individuals.  The variance of
# the random effects for individuals in group 1 is 4, the
# variance of the random effects for individuals in group 2 is 1.
re = rnorm(n)
re[grp0==1] = 2*re[grp0==1]
re = rep(re, a)

# Simulate the outcome, with error SD 0.25.
y = re + 0.25*rnorm(m*n)

# Put the data into a dataframe
da = data.frame(idx=idx, grp=grp, y=y)

# Create group indicators
da$grp1 = 1*(da$grp == 1)
da$grp2 = 1*(da$grp == 2)

# Fit the model
m = lmer(y ~ (0 + grp1 | idx) + (0 + grp2 | idx), data=da)
