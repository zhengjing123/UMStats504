library(lme4)
library(dplyr)
library(readr)
library(ggplot2)

df = read_csv("chns.csv.gz")

pdf("mixedlm_r_plots.pdf", width=8, height=6)

df = df[df$indinc >= 0,]
df$logindinc = log(1 + df$indinc)
df$sex = factor(df$female, c(0, 1), c("male", "female"))
df$urban = factor(df$urban, c(0, 1), c("rural", "urban"))

df$Idind = as.factor(df$Idind)

df = mutate(group_by(df, Idind), age_cen=age-mean(age))

do_fit = function(vname) {

    # The response variable
    df$y = df[[vname]]

    # Fit the mixed model
    m = lmer(y ~ sex*urban + age + wave + educ + logindinc + (1+age_cen|Idind), data=df)
    print(summary(m))

    # Extract the variance and correlation parameters from the fitted model
    s = attr(VarCorr(m)[[1]], "stddev")
    c = attr(VarCorr(m)[[1]], "correlation")
    r = c[1, 2]

    # For illustration, generate a sample from the random effects distribution.
    x = NULL
    for (i in 1:20) {
        e = rnorm(2)
        e[2] = r*e[1] + sqrt(1-r^2)*e[2]
        e = e * s
        z = e[1] + e[2] * c(-5, 5)
        x = rbind(x, c(i, -1, z[1]))
        x = rbind(x, c(i, 1, z[2]))
    }
    x = data.frame(x)
    names(x) = c("g", "x", "y")

    # Plot the sampled random effects trajectories
    p = ggplot(x, aes(x=x, y=y, group=g)) + geom_line()
    p = p + ylab(vname)
    p = p + xlab("Age")
    print(p)
}

for (vname in c("d3kcal", "d3carbo", "d3fat", "d3protn")) {
    do_fit(vname)
}

dev.off()

system("cp mixedlm_r_plots.pdf ~kshedden")
