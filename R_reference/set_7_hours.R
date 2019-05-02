
library(stats)
library(forecast)
library(aTSA)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# load data (Total Hours Worked in Spain and U.S.)
data <- read.csv(file="hours_worked.csv", header=TRUE, sep=",")

#------------------------------------------------------------
# Data for Spain (non-stationary AR(p))
tsdata <- ts(data$Spain,start=c(1970,1),frequency=1)
plot.ts(tsdata, ylab ="Total Hours Worked in Spain")

# test for unit root
adf.test(data$Spain)

# difference the data
D.data.Spain <- diff(data$Spain, lag = 1, differences = 1)
plot(D.data.Spain, type="l")

# test for unit root
adf.test(D.data.Spain)

# ACF, PACF
ACF <- acf(D.data.Spain, lag.max = 10 , plot = FALSE, demean = TRUE)
plot(ACF[1:10], main="ACF")

PACF <- pacf(D.data.Spain, lag.max = 10, plot = FALSE, demean = TRUE, main="ACF")
plot(PACF[1:10], main="PACF")

# model estimation and forecast (original data)
mSpain <- arima(data$Spain, order = c(2,1,2), include.mean=TRUE)
fcast <- forecast::forecast(mSpain, lead=5)
plot(fcast)

#------------------------------------------------------------
# Data for U.S. (stationary AR(p) with a trend)
tsdata <- ts(data$US,start=c(1970,1),frequency=1)
plot.ts(tsdata, ylab ="Total Hours Worked in U.S.")

# test for unit root
adf.test(data$US)

# adf.test rejected unit root and hence the upward trend is due to a deterministic linear trend
# detrend the data and select model for the residuals based on ACF, PACF
t <- seq(from=1, to=length(data$US), by=1)
trendMod <- lm(data$US ~ t) 
summary(trendMod)
detrended.data.US <- residuals(trendMod)
plot(detrended.data.US, type="l")

ACF <- acf(detrended.data.US, lag.max = 10 , plot = FALSE, demean = TRUE)
plot(ACF[1:10], main="ACF")

PACF <- pacf(detrended.data.US, lag.max = 10, plot = FALSE, demean = TRUE, main="ACF")
plot(PACF[1:10], main="PACF")

# fit the model (note that the xreg option specifies a linear trend)
time <- seq(from=1, to=length(data$US), by=1)
mUS <- arima(data$US,order=c(2,0,1),xreg = time)
mUS

# verify residuals
ACF <- acf(residuals(mUS), lag.max = 10 , plot = FALSE, demean = TRUE)
plot(ACF[1:10], main="residual ACF")

PACF <- pacf(residuals(mUS), lag.max = 10, plot = FALSE, demean = TRUE, main="ACF")
plot(PACF[1:10], main="PACF")

# forecast 
s = 5
ftime <- seq(from=length(time)+1, to=length(time)+s, by=1)
fcast <- forecast(mUS,h=5,xreg=ftime)
plot(fcast)

