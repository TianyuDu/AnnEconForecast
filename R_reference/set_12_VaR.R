

# Install and load required packages
library(aTSA)
library(xts)
library(anytime)
library(fGarch)

# Dowload data on S&P500 index, daily frequency
# Source: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC

# Load, extract Close price, convert to xts format, and plot
# setwd("C:/Teaching/Undergrad/data ECO374")
data_SP500_full <- read.csv(file="SP500_daily.csv", header=TRUE, sep=",")
# Select column Frame[c(...)]
xtsdata_SP500 <- xts(x=data_SP500_full[c("Close")], order.by=anytime(data_SP500_full$Date))
plot.xts(xtsdata_SP500, plot.type = c("single"), main ="S&P 500 index, close price" , legend.loc = "topleft", grid.col="lightgray")

# Plot of returns
# log(1+r) ~ r, taylor's polynomial of degree 1.
xtsdata_SP500_returns <- diff(log(xtsdata_SP500))
xtsdata_SP500_returns <- na.omit(xtsdata_SP500_returns)
plot.xts(xtsdata_SP500_returns, plot.type = c("single"), main ="S&P 500 index, returns" , legend.loc = "topleft", grid.col="lightgray", col="blue")

# Model
model <- garchFit(
    formula=~arma(1,1)+garch(1,1),
    data=xtsdata_SP500_returns,
    trace=FALSE)
summary(model)

forecast <- predict(model, plot=FALSE, n.ahead=10)
summary(forecast)

# Calculate VaR at alpha = 1%
f_mean <- forecast$meanForecast
f_std <- forecast$standardDeviation 
VaR <- f_mean - 2.33*f_std  
VaR

# VaR for a $1,000,000 portfolio
VaR*1e06

# Calculate Expected Shortrall at alpha = 1%
ES <- f_mean - 2.64*f_std
ES

# ES for a $1,000,000 portfolio
ES*1e06

