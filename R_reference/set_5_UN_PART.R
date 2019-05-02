# install.packages("forecast")

# load package nlme for autocorrelation functions
library(stats)
library(forecast)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data (Monthly data on U.S. unemployed people, in thousands)
data <- read.csv(file="unemp_part.csv", header=TRUE, sep=",")

# plot, ACF, PACF
tsdata <- ts(data$unempl_part,start=c(1989,1),frequency=12)
plot.ts(tsdata, ylab ="U.S. Unemployed, looking for part time work")

ACF <- acf(data$unempl_part, lag.max = 150, plot = FALSE, demean = TRUE)
plot(ACF[1:20], main="ACF")

PACF <- pacf(data$unempl_part, lag.max = 150, plot = FALSE, demean = TRUE)
plot(PACF[1:10], main="PACF")

#----------------------------------------------------------------------
# forecasting with AR(p)

AR <- arima(data$unempl_part, order = c(4,0,0), include.mean=TRUE)
AR
ARfcast <- forecast(AR,h=10)
plot(ARfcast)

ARfct = predict(AR,n.ahead=10)
plot(ARfct$pred, main="AR(4) forecast", ylab="forecast")

ARfct = predict(AR,n.ahead=30)
plot(ARfct$pred, main="AR(4) forecast", ylab="forecast")

