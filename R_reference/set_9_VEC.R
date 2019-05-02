
#Set data directory path
setwd("C:/Teaching/Undergrad/data ECO374")
 
#Install and load required packages
chooseCRANmirror(graphics=FALSE, ind=33)
if (!require("urca")) install.packages("urca")
if (!require("tsDyn")) install.packages("tsDyn")

library(aTSA)
library(xts)
library(anytime)
library(urca)
library(tsDyn)

#Dowload data on quarterly Personal Consumption Expenditure
#Source: https://fred.stlouisfed.org/series/PCEC, download file PCEC.csv

#Dowload data on quarterly GDP
#Source: https://fred.stlouisfed.org/series/GDP, download file GDP.csv

#Load, log, convert to xts format, and merge data on quarterly Personal Consumption Expenditure and GDP
data_PCEC <- read.csv(file="PCEC.csv", header=TRUE, sep=",")
data_GDP <- read.csv(file="GDP.csv", header=TRUE, sep=",")

data_PCEC$PCEC <- log(data_PCEC$PCEC)
data_GDP$GDP <- log(data_GDP$GDP)

xtsdata_PCEC <- xts(x=data_PCEC[c("PCEC")], order.by=anytime(data_PCEC$DATE))
xtsdata_GDP <- xts(x=data_GDP[c("GDP")], order.by=anytime(data_GDP$DATE))

xtsdata <- merge.xts(xtsdata_PCEC,xtsdata_GDP)

#Data time series plot
plot.xts(xtsdata, plot.type = c("single"), main ="log Consumption and log Production (GDP)" , legend.loc = "topleft", grid.col="lightgray")

#Plot of log Consumption vs log Production (GDP)
par(pty="s")
plot(x=data_PCEC$PCEC, y=data_GDP$GDP, xlab=("log Consumption"), ylab=("log Production (GDP)"))

#Unit root test for log GDP
adf.test(xtsdata$GDP)

#Unit root test for log Consumption
adf.test(xtsdata$PCEC)

#Run Johansen Cointegration Test
#The output does not provide p-values, therefore the value of the test statistic "test" needs to be compared
#for each hypothesized values of r with the critical values at 10pct, 5pct, and 1pct 
#to determine the outcome of the test. 

#Here, for r = 0, "test" > (10pct or 5pct or 1pct) and hence we reject the null of no cointegration. 
#We conclude that log Production (GDP) and log Consumption (PCEC) are cointegrated.
Y <- as.matrix(as.data.frame(xtsdata))
jotest <- ca.jo(Y)
summary(jotest)

#Plot of deviations from equilibrium
reg <- lm(data_GDP$GDP ~ data_PCEC$PCEC)
z <- residuals(reg)
plot(z, main ="Deviations from equilibrium", type="l")
abline(h=0)

#Estimate VEC model with one lag
vecm <- lineVar(data=Y, model = c("VECM"), r=1, lag=1)
summary(vecm)
AIC(vecm)

#Choose lag order based on AIC minimization
maxlag=8
lag <- seq(0,0,length.out=maxlag)
aic <- seq(0,0,length.out=maxlag)

for (k in 1:maxlag) {
  vecm <- lineVar(data=Y, model = c("VECM"), r=1, lag=k)
  lag[k] <- k
  aic[k] <- AIC(vecm)
} 
plot(lag,aic)
lines(lag,aic)

Forecast
vecm <- ca.jo(Y, K=3)
vecm.levels <- vec2var(vecm, r=1)
fc <- predict(vecm.levels, n.head = 8)
plot(fc)
