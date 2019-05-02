
# load package nlme for autocorrelation functions
library(stats)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data
data <- read.csv(file="Figure3_10_HoursWorkedUSA.csv", header=TRUE, sep=",")

#-------------------------------------------------------------------
# calculate autocorrelation function
ACF <- acf(data$USA, lag.max = NULL, plot = FALSE, demean = TRUE)

# plot acf
plot(ACF)

# write out acf table
ACF

#-------------------------------------------------------------------
# calculate partial autocorrelation function
PACF <- pacf(data$USA, lag.max = NULL, plot = FALSE, demean = TRUE)

# plot pacf
plot(PACF)

# write out acf table
PACF

#-------------------------------------------------------------------
# Q-statistic
for (k in 1:12) {
  Q = Box.test(data$USA, lag = k, type = "Ljung-Box", fitdf = 0)
  print(paste("lag = ",k))
  print(Q)
}
 
