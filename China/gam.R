library(mgcv)
library(dplyr)
library(readr)
library(ggplot2)

df = read_csv("chns.csv.gz")

pdf("gam_plots.pdf", width=8, height=6)

df = df[df$indinc >= 0,]
df$logindinc = log(1 + df$indinc)
df$sex = factor(df$female, c(0, 1), c("male", "female"))
df$urban = factor(df$urban, c(0, 1), c("rural", "urban"))

# Fit an additive model with the variable "vname" used as the
# dependent variable.
make_plot = function(vname) {

    # The response variable
    df$y = df[[vname]]

    # Fit an additive model
    m = gam(y ~ sex*urban*age + wave + educ + s(age, k=5, by=sex:urban) + s(wave, k=5) + s(educ, k=5) +
            s(logindinc, k=5), data=df, family=gaussian)

    # Create a dataframe for prediction
    pd = data.frame(sex=NULL, age=NULL, y=NULL)
    age = seq(18, 80, length.out=100)
    for (sex in c("female", "male")) {
        for (loc in c("rural", "urban")) {
            dx = df[1:100,]
            dx$age = age
            dx$sex = sex
            dx$urban = loc
            dx$wave = 2000
            dx$educ = mean(df$educ)
            dx$logindinc = mean(df$logindinc)
            dx = select(dx, wave, sex, urban, logindinc, age, educ)
            pd = rbind(pd, dx)
        }
    }
    dx$sex = as.factor(dx$sex)
    dx$urban = as.factor(dx$urban)

    # Get the predictedvalues
    pd$y = predict(m, newdata=pd)

    # Plot the predicted values
    p = ggplot(pd, aes(x=age, y=y, group=interaction(sex, urban), col=interaction(sex, urban)))
    p = p + geom_line()
    p = p + ylab(vname)
    print(p)
}

for (vname in c("d3kcal", "d3carbo", "d3fat", "d3protn")) {
    make_plot(vname)
}

dev.off()

system("cp gam_plots.pdf ~kshedden")
