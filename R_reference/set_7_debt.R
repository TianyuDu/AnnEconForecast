library(stats)
library(forecast)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data (Home Mortgage Outstanding Debt, billions US$)
data <- read.csv(file="debt.csv", header=TRUE, sep=",")

# plot, ACF, PACF
tsdata <- ts(data$debt,start=c(1992,1),frequency=4)
plot.ts(tsdata, ylab ="Home Mortgage Outstanding Debt, billions US$")

# specify a polynomial trend regression model
t <- seq(from=1, to=length(data$debt), by=1)
t2 <- t^2
t3 <- t^3
t4 <- t^4
trendMod <- lm(data$debt ~ t + t2 + t3 + t4)
summary(trendMod)


res <- residuals(trendMod)
plot(res, type="l")

ACF <- acf(res, plot = FALSE, demean = TRUE)
plot(ACF[1:20], main="Residual ACF")

PACF <- pacf(res, plot = FALSE, demean = TRUE)
plot(PACF[1:20], main="Residual PACF")

#----------------------------------------------------------------------
# specify an ARMA model for the residuals
ARMA <- arima(res, order = c(1,0,2), include.mean=TRUE)
resres <- residuals(ARMA)

ACF2 <- acf(resres, plot = FALSE, demean = TRUE)
plot(ACF2[1:20], main="ARMA Residual ACF")

PACF2 <- pacf(resres, plot = FALSE, demean = TRUE)
plot(PACF2[1:20], main="ARMA Residual PACF")


