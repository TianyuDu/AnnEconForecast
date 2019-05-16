# May 16 2019
library(fpp2)
library(ggfortify)
library(forecast)
library(stats)
library(aTSA)

df <- read.csv("./data/CPIAUCSL.csv", header=TRUE, sep=",")

ts <- ts(df$CPIAUCSL,start=c(1947,1),frequency=12)
# plot.ts(tsdata_un, ylab="Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)")

autoplot(ts) +
    ggtitle("Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)") +
    xlab("Date") +
    ylab("CPI") +
    theme(plot.title = element_text(size=8, face="bold"))

transformed <- diff(ts, lag=1, differences=1)

acf <- Acf(transformed, plot=TRUE)
pacf <- Pacf(transformed, plot=TRUE)

autoplot(transformed) + 
    ggtitle("Differenced Series") +
    xlab("Date") +
    ylab("Transformed")

adf.test(transformed)
