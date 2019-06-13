# May 16 2019
# Modify: Jun. 13 2019
library(fpp2)
library(ggfortify)
library(forecast)
library(stats)
library(aTSA)
library(lmtest)
library(xts)
library(anytime)

df <- read.csv("./data/CPIAUCSL.csv", header=TRUE, sep=",")

# ts_data <- ts(df$CPIAUCSL,start=c(1947,1),frequency=12)
ts_all <- xts(x=df[c("CPIAUCSL")], order.by=anytime(df$DATE))

autoplot(ts_all) +
    ggtitle("Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)") +
    xlab("Date") +
    ylab("CPI") +
    theme(plot.title = element_text(size=8, face="bold"))

# Train and test spliting
train_size <- as.integer(0.8 * length(ts_all))
test_size <- length(ts_all) - train_size

ts_train <- head(ts_all, train_size)
ts_test <- tail(ts_all, test_size)

# Forecasting of naive preddictor, as a baseline model.
baseline_error <- mean(
    na.omit(diff(ts_test)) ** 2
)
cat("Base line MSE: ", baseline_error)

# Operations on the Training Set
# Determine coefficient of integration.
cat("Order of Integration: ", ndiffs(ts_train))
transformed <- diff(ts_train, lag=1, differences=2)

ACF <- Acf(transformed, plot=TRUE)
PACF <-Pacf(transformed, plot=TRUE)


autoplot(transformed) +
    ggtitle("Differenced Series") +
    xlab("Date") +
    ylab("Transformed")

adf.test(transformed)

# Find the Best ARIMA Profile
auto.arima(ts_train$CPIAUCSL, max.p=10, max.P=10, max.q=10, max.Q=10, max.d=10, max.D=10, stationary=FALSE, trace=TRUE)

# ==== Fit and Evaluate the Model ====
profile <- c(2, 2, 2)
model <- arima(ts_train, order=profile)
res <- residuals(model)
mse_train <- mean(res**2)
rmse_train <- sqrt(mse_train)

err_abspec <- abs(res) / as.matrix(ts_train)
mape_train <- mean(
    err_abspec[err_abspec != Inf]
)
cat("Training Set MSE:", mse_train, "RMSE:", rmse_train, "MAPE:", mape_train)

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
        order=profile
    )
    # Produce one step forecast
    f <- forecast::forecast(model, h=1, level=c(99))
    rf <- c(rf, f$mean)  # Add rolling forecast.
}
ts_rf <- xts(x=rf, order.by=anytime(time(ts_test)))
combined <- cbind("Rolling Forecast"=ts_rf, "Actual"=ts_test)
autoplot(combined, facets=FALSE, alpha=0.6) +
    ggtitle("One Step Rolling Forecasting on Test Set") +
    xlab("Date") +
    ylab("Sunspot")
# Performance Metrics
mse_test <- mean((ts_rf - ts_test) ** 2)
rmse_test <- sqrt(mse_test)
cat("Test MSE:", mse_test, "RMSE:", rmse_test)
