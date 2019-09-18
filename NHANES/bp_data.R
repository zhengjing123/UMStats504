library(haven)
library(dplyr)

# This is an simplified version of data_prep.py, translated to R.

# List of file names to merge
xpt_files = c("DEMO_I.XPT", "BPX_I.XPT", "BMX_I.XPT")

# Retain these variables.
vax = list(c("SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1"),
           c("SEQN", "BPXSY1", "BPXDI1"),
           c("SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXLEG", "BMXARMC"))

dfs = list()
for (j in 1:length(xpt_files)) {
    fna = sprintf("data/2015/%s", xpt_files[j])
    df = read_xpt(fna)
    df = select(df, vax[[j]])
    dfs[[j]] = df
}

dx = left_join(dfs[[1]], dfs[[2]])
dx = left_join(dx, dfs[[3]])

# Relace the gender variable with a female indicator
dx = dx[complete.cases(dx), ]
dx$Female = 1*(dx$RIAGENDR == 2)
dx = select(dx, -RIAGENDR)

# Recode the ethnic groups
dx = mutate(dx, RIDRETH1=recode(RIDRETH1, "1"="MA", "2"="OH", "3"="NHW", "4"="NHB", "5"="OR"))
dx$RIDRETH1 = as.factor(dx$RIDRETH1)
