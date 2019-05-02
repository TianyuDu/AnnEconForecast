
# load package nlme for autocorrelation functions
library(stats)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data (Per capita personal income growth in California)
data <- read.csv(file="g_Cal.csv", header=TRUE, sep=",")

# plot, ACF, PACF
tsdata <- ts(data$g_pci_california,start=c(1969))
plot.ts(tsdata, ylab ="pc income growth in California")

ACF <- acf(data$g_pci_california, lag.max = 15, plot = FALSE, demean = TRUE)
plot(ACF[1:15], main="ACF")
ACF[1]

PACF <- pacf(data$g_pci_california, lag.max = 15, plot = FALSE, demean = TRUE)
plot(PACF[1:15], main="PACF")
PACF[1]
