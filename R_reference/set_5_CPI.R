
# load package nlme for autocorrelation functions
library(stats)
library(forecast)

# Download data from https://fred.stlouisfed.org/series/CPIAUCNS
# Edit Graph - annual frequency, percentage change
# Download CSV: CPIAUCNS.csv

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data
data_cpi <- read.csv(file="CPIAUCNS.csv", header=TRUE, sep=",")

# plot, ACF, PACF
tsdata <- ts(data_cpi$CPIAUCNS_PCH,start=c(1914))
plot.ts(tsdata, ylab ="U.S. CPI pct change")
abline(h=c(0,length(data$CPIAUCNS_PCH)))

ACF <- acf(data_cpi$CPIAUCNS_PCH, lag.max = 50, plot = FALSE, demean = TRUE)
plot(ACF[1:50], main="ACF")

PACF <- pacf(data_cpi$CPIAUCNS_PCH, lag.max = 50, plot = FALSE, demean = TRUE, main="ACF")
plot(PACF[1:50], main="PACF")

#----------------------------------------------------------------------
# forecasting with AR(2)

AR2 <- arima(data_cpi$CPIAUCNS_PCH, order = c(2,0,0), include.mean=TRUE)
AR2
AR2fcast <- forecast(AR2,h=10)
plot(AR2fcast)

AR2fct = predict(AR2,n.ahead=10)
plot(AR2fct$pred, main="AR(2) forecast", ylab="forecast")

AR2fct = predict(AR2,n.ahead=30)
plot(AR2fct$pred, main="AR(2) forecast", ylab="forecast")

