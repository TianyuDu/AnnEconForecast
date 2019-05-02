
# Set data directory path
setwd("C:/Teaching/Undergrad/data ECO374")
 
#Install and load required packages
chooseCRANmirror(graphics=FALSE, ind=33)
if (!require("xts")) install.packages("xts")
if (!require("anytime")) install.packages("anytime")
if (!require("vars")) install.packages("vars")

library(forecast)
library(xts)
library(anytime)
library(vars)

# Dowload data from http://www.freddiemac.com/research/indices/house-price-index.html, Non-Seasonally Adjusted Index Values, file MSAs_NSA.xls.
# Extract data for Los Angeles and Riverside, save in comma separated format as LA_Riverside_hpi.csv.
# Load data on MONTHLY house price index for LA and Riverside
data <- read.csv(file="LA_Riverside_hpi.csv", header=TRUE, sep=",")

# Data plot (uses package xts and anytime)
xtsdata <- xts(x=data[c("LA", "Riverside")], order.by=anytime(data$Month))
plot.xts(xtsdata, plot.type = c("single"), main ="House Price Index" , legend.loc = "topleft", grid.col="lightgray")

# Differenced data plot (uses package xts)
Dxtsdata <- diff(xtsdata, lag = 1)
plot.xts(Dxtsdata, plot.type = c("single"), main ="Differenced House Price Index" , legend.loc = "topleft", grid.col="lightgray")

# VAR model estimation (uses package vars)
Ddata <- diff.xts(xtsdata, lag=1)
DY <- data.matrix(as.data.frame(Ddata))
DY <- na.omit(DY)

v <- VAR(y=DY, p=1)
summary(v)

# VAR model lag selection based on AIC (uses package vars)
maxlag=24
lag <- seq(0,0,length.out=maxlag)
aic <- seq(0,0,length.out=maxlag)

for (k in 1:maxlag) {
  v <- VAR(y=DY, p=k)
  lag[k] <- k
  aic[k] <- AIC(v)
} 
plot(lag,aic)
lines(lag,aic)

print(aic)

# Granger causality test
v <- VAR(y=DY, p=15)
causality(v, cause = "LA")
causality(v, cause = "Riverside")

# Impulse-response function: LA -> Riverside
irf1 <- irf(v, impulse = "LA", response = "Riverside", n.ahead = 24)
plot(irf1)

# Impulse-response function: Riverside -> LA  
irf2 <- irf(v, impulse = "Riverside", response = "LA", n.ahead = 24)
plot(irf2)

# Forecast
fc <- predict(v,n.ahead = 24)
plot(fc, names="LA")
plot(fc, names="Riverside")




