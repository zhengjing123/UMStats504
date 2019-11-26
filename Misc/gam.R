# Check GAM using simulated data

library(mgcv)
library(dplyr)

# Number of people
n = 10000

# Number of observations per person
m = 1 + rpois(n, 5)

# Total number of observations
nm = sum(m)

# Subject id
idx = rep(seq(n), m)

age = rnorm(nm)
df = data.frame(age=age)

df$idx = idx
df = ungroup(mutate(group_by(df, idx), age=sort(age)))

# Covariates that don't vary in time
df$marst = as.factor(floor(rep(4*runif(n), m)))
df$region = as.factor(floor(rep(4*runif(n), m)))
df$status = as.factor(floor(rep(4*runif(n), m)))
df$year = df$age + rep(rnorm(n), m)
df$educ = rep(rnorm(n), m)
df$popul = rep(rnorm(n), m)
df$occup = as.factor(floor(rep(10*runif(n), m)))

# Satisfaction (outcome variable)
u = 0.5 * (0.2*as.integer(df$marst) + 0.3*as.integer(df$region) + df$year - df$age + df$educ - df$popul)
u = 3 + u + rnorm(length(u))
u = floor(u)
u[u < 1] = 1
u[u > 5] = 5
df$satis = u


res = gam(satis ~ region + s(year, k=5) + marst + s(age, k=5) + occup +
                  s(educ, k=5, by=marst:region) + s(popul, k=5, by=status),
                  family=ocat(R=5), data=df)
