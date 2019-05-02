
#install.packages("forecast")

# load packages
library(stats)
library(forecast)

# change your data directory path here
setwd("C:/Teaching/Undergrad/ECO374 19 winter/R")

# download data from https://fred.stlouisfed.org/series/TLRESCON#0
# U.S. Total construction spending: residential
# modify graph: Changes in millions of dollars, save data
# load data 
data <- read.csv(file="TLRESCON.csv", header=TRUE, sep=",")

# plot, ACF, PACF
tsdata <- ts(data$TLRESCON_CHG,start=c(2002,2),frequency=12)
plot.ts(tsdata, ylab ="U.S. Total Residential Spending, change")
abline(h=c(0,0))

ACF <- acf(data$TLRESCON_CHG, lag.max = 150, plot = FALSE, demean = TRUE)
plot(ACF[1:36], main="ACF")

PACF <- pacf(data$TLRESCON_CHG, lag.max = 150, plot = FALSE, demean = TRUE)
plot(PACF[1:36], main="PACF")

#----------------------------------------------------------------------
# forecasting with S-AR AR

AR <- arima(data$TLRESCON_CHG, order = c(1,0,0), seasonal = list(order=c(1,0,0),period=12), include.mean=TRUE)
AR

ARfcast <- forecast(AR,h=10)
plot(ARfcast)

ARfct = predict(AR,n.ahead=10)
plot(ARfct$pred, main="S-AR(1) AR(1) forecast", ylab="forecast")

ARfct = predict(AR,n.ahead=30)
plot(ARfct$pred, main="S-AR(1) AR(1) forecast", ylab="forecast")

