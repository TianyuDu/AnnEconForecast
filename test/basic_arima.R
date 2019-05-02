# May. 1, 2019
# Basic ARIMA model for arbitrary dataset.
# Example on sunspot dataset.
library(forecast)
library(aTSA)
library(fGarch)
library(ggfortify)

data <- na.omit(read.csv("./data/sunspots.csv", header=TRUE, sep=","))
tsdata <- ts(data$Sunspot.Numbers., start=c(1700), frequency=1)

# ==== Setup Hyperparameters ====
TEST_SIZE <- 30
TRAIN_SIZE <- length(tsdata) - TEST_SIZE

ts_train <- head(tsdata, TRAIN_SIZE)
ts_test <- tail(tsdata, TEST_SIZE)
plot.ts(tsdata, ylab="Sunspot Numbers")


# Basic look at the structure of time series.
ACF <- acf(ts_train, lag.max=150, plot=FALSE, demean=TRUE)
plot(ACF[1:20], main="ACF")

PACF <- pacf(ts_train, lag.max=150, plot=FALSE, demean=TRUE)
plot(PACF[1:20], main="PACF")

ARIMA <- arima(ts_train, order=c(8, 1, 1), include.mean=TRUE)

ARIMAfct <- predict(ARIMA, n.ahead=30, plot=TRUE)
plot(ARIMAfct$pred, main="ARIMA Forecast", ylab="forecast")

autoplot(ts(
    cbind(c(ts_train, ARIMAfct$pred), tsdata), start=c(1700), frequency=1
    ), facets=FALSE)

error <- (ARIMAfct$pred - ts_test)**2
rmse <- sqrt(mean(error))
print(rmse)

# GARCH <- garchFit(~arma(1, 1)+garch(1, 1), data=ts_train, trace=TRUE)
# f <- predict(GARCH, plot=TRUE, n.ahead=TEST_SIZE)
