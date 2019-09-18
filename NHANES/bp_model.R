library(dplyr)
library(splines)
library(ggplot2)

source("bp_data.R")

# Fit a model
rslt1 = lm(BPXSY1 ~ (bs(RIDAGEYR, 6) + bs(BMXBMI, 6)) * Female + C(RIDRETH1), dx)

# Build a dataframe for constructing the mean values to plot
da = dx[1:100,]
da$RIDRETH1 = "OH"
da$RIDRETH1 = factor(da$RIDRETH1, levels=levels(dx$RIDRETH1))
da$RIDAGEYR = seq(from=18, to=80, length.out=100)

pl = ggplot()

# Get the fitted mean values for several subgroups
dz = NULL
for (female in c(0, 1)) {
    for (bmi in c(22, 25, 28)) {

        da$Female = female
        da$BMXBMI = bmi

        if (female == 1) {
            label = sprintf("Female, BMI=%d", bmi)
        } else {
            label = sprintf("Male, BMI=%d", bmi)
        }

        sbp = sbp=predict(rslt1, newdata=da)
        du = data.frame(Female=female, BMXBMI=bmi, Age=da$RIDAGEYR, label=label, sbp=sbp)
        dz = rbind(dz, du)
    }
}

pl = pl + geom_line(data=dz, aes(x=Age, y=sbp, group=label, color=label))

pl = pl + guides(color=guide_legend(title=NULL))

pl = pl + xlab("Age (years)")
pl = pl + ylab("SBP (mm/Hg)")

ggsave("bp_model_r.pdf", height=6, width=8)
