# May 16 2019
library(fpp2)
library(ggfortify)
library(forecast)
library(stats)
library(aTSA)

df <- read.csv("./data/CPIAUCSL.csv", header=TRUE, sep=",")

ts_data <- ts(df$CPIAUCSL,start=c(1947,1),frequency=12)

autoplot(ts_data) +
    ggtitle("Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)") +
    xlab("Date") +
    ylab("CPI") +
    theme(plot.title = element_text(size=8, face="bold"))

# Classical decomposition of target series.
components.ts <- decompose(ts_data)
plot(components.ts)

# The seasonal plot
ggseasonplot(ts_data,polar=TRUE)+
    ggtitle("CPIAUCSL (Season plot)")+
    theme(plot.title = element_text(size=8, face="bold"))

transformed <- diff(ts, lag=1, differences=1)

ggsubseriesplot(ts_data)+
    ggtitle("CPIAUCSL (Subseries plot)")+
    ylab("number of deaths")+
    theme(plot.title = element_text(size=8, face="bold"))

ACF <- Acf(transformed, plot=TRUE)
PACF <-Pacf(transformed, plot=TRUE)

# AR=4, MA=Inf -> ARIMA(4,1,0)

autoplot(transformed) +
    ggtitle("Differenced Series") +
    xlab("Date") +
    ylab("Transformed")

adf.test(transformed)

# ==== Fit and Evaluate the Model ====
model <- arima(ts, order = c(4,1,0))
res <- residuals(model)
mse <- mean(res**2)
print(mse)
fore <- predict(model, n.ahead=10)
