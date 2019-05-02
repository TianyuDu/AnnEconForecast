
# load package nlme for autocorrelation functions
library(stats)

# download data from https://fred.stlouisfed.org/series/DGS5
# max range, monthly, % change

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data
data <- read.csv(file="DGS5.csv", header=TRUE, sep=",")

# plot, ACF, PACF
plot(data$DGS5_CHG, type="l")
abline(h=c(0,length(data$DGS5_CHG)))

ACF <- acf(data$DGS5_CHG, lag.max = NULL, plot = FALSE, demean = TRUE)
plot(ACF[1:20])

PACF <- pacf(data$DGS5_CHG, lag.max = NULL, plot = FALSE, demean = TRUE)
plot(PACF[1:20])

#----------------------------------------------------------------------
# forecasting with MA(1)

MA1 <- arima(data$DGS5_CHG, order = c(0,0,1), include.mean=TRUE)
MA1

forecast <- predict(MA1)
forecast 
