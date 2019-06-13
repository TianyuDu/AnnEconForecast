# Jun. 10, 2019
# Exchange rate prediction using ARIMA models

library(fpp2)
library(ggfortify)
library(forecast)
library(stats)
library(aTSA)
library(lmtest)
library(xts)
library(anytime)

setwd("/Users/tianyudu/Documents/Academics/EconForecasting/AnnEconForecast")
df <- read.csv("./data/DEXCAUS_cleaned.csv", header=TRUE, sep=",", col.names=c("Date", "Exchange"))

# ts_all <- ts(df, start=c(1971, 1, 04),frequency=30)

ts_all <- xts(x=df[c("Exchange")], order.by=anytime(df$Date))

# Train and test spliting
train_size <- as.integer(0.8 * length(ts_all))
test_size <- length(ts_all) - train_size

ts_train <- head(ts_all, train_size)
ts_test <- tail(ts_all, test_size)

# Forecasting of naive preddictor, as a baseline model.
baseline_error <- mean(
    na.omit(diff(ts_test)) ** 2
)
cat(baseline_error)
# ==== End ====

autoplot(ts_train) +
    ggtitle("Number of sunspots") +
    xlab("Date") +
    ylab("Exchange Rate") +
    theme(plot.title = element_text(size=8, face="bold"))

# Determine the d number.
ndiffs(ts_train)

# Inspect the transformed sequence
transformed <- diff(ts_train, lag=1, differences=1)

ACF <- Acf(transformed, plot=TRUE, lag.max=34) # q=3
PACF <-Pacf(transformed, plot=TRUE, lag.max=34) # p=3

autoplot(transformed) +
    ggtitle("Differenced Series") +
    xlab("Date") +
    ylab("Transformed")

# Test stationarity
adf.test(transformed)

# Find the model minimizing AIC w/ correction.
# auto.arima(ts_train, max.p=30, max.P=30, max.q=30, max.Q=30, max.d=3, max.D=3, stationary=TRUE, trace=TRUE)
auto.arima(ts_train, max.order=30, stationary=FALSE, trace=TRUE)

# ==== Fit and Evaluate the Model ====
model <- arima(ts_train, order=c(0, 1, 2))
res <- residuals(model)
mse <- mean(res**2)
rmse <- sqrt(mse)

err_abspec <- abs(res) / as.matrix(ts_train)
mape <- mean(
    err_abspec[err_abspec != Inf]
)
cat("MSE:", mse, "RMSE:", rmse, "MAPE:", mape)

coeftest(model)

# Simple forecast
fore <- forecast::forecast(model, h=1, level=c(99))
autoplot(fore)

# Rolling forecast
rf <- c()
for (i in c(1: test_size)){
    # Refit model
    model <- arima(
        head(ts_all, train_size+i-1),
        # To predict the first element of the test set, we use the whole training set.
        # To predict the k-th element of the test set (i.e. train_size+k th element), use all previouse train_size+k-1 elements.
        order=c(3,1,3)
    )
    # Produce one step forecast
    f <- forecast::forecast(model, h=1, level=c(99))
    rf <- c(rf, f$mean)  # Add rolling forecast.
}
ts_rf <- ts(rf, start=time(ts_test)[1])
combined <- cbind("Rolling Forecast"=ts_rf, "Actual"=ts_test)
autoplot(combined) +
    ggtitle("One Step Rolling Forecasting on Test Set") +
    xlab("Date") +
    ylab("Sunspot")
# Performance Metrics
mse_test <- mean((ts_rf - ts_test) ** 2)
rmse_test <- sqrt(mse_test)
cat("Test MSE:", mse, "RMSE:", rmse)
