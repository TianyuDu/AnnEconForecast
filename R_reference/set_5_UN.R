
# load package nlme for autocorrelation functions
library(stats)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data (Monthly data on U.S. unemployed people, in thousands)
data_un <- read.csv(file="Unemployed.csv", header=TRUE, sep=",")

# plot, ACF, PACF
tsdata_un <- ts(data_un$Unemployed,start=c(1989,1),frequency=12)
plot.ts(tsdata_un, ylab ="U.S. Unemployed people (thousands)")
abline(h=c(0,length(data_un$Unemployed)))

ACF <- acf(data_un$Unemployed, lag.max = 150, plot = FALSE, demean = TRUE)
plot(ACF[1:150])

PACF <- pacf(data_un$Unemployed, lag.max = 150, plot = FALSE, demean = TRUE)
plot(PACF[1:150])
