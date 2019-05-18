# May 17 2019
# Benchmark ARIMA for sunspot data.

library(fpp2)
library(ggfortify)
library(forecast)
library(stats)
library(aTSA)

df <- read.csv("./data/sunspots.csv", header=TRUE, sep=",", col.names=c("Year", "Sunspots"))

ts <- ts(df$Sunspots,start=c(1700),frequency=1)
# plot.ts(tsdata_un, ylab="Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)")

autoplot(ts) +
    ggtitle("Number of sunspots") +
    xlab("Year") +
    ylab("Sunspot") +
    theme(plot.title = element_text(size=8, face="bold"))

transformed <- diff(ts, lag=1, differences=1)

ACF <- Acf(transformed, plot=TRUE)
PACF <-Pacf(transformed, plot=TRUE)

# AR=4, MA=Inf -> ARIMA(4,1,0)

autoplot(transformed) + 
    ggtitle("Differenced Series") +
    xlab("Date") +
    ylab("Transformed")

adf.test(transformed)

# ==== Fit and Evaluate the Model ====
model <- arima(ts, order = c(8,1,1))
res <- residuals(model)
mse <- mean(res**2)
print(mse)
fore <- predict(model, n.ahead=10)
