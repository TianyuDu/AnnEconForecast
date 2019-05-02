 
# Install and load required packages
library(aTSA)
library(xts)
library(anytime)
library(fGarch)

# Dowload data on Apple stock (AAPL) for the last five years, daily frequency
# Source: https://finance.yahoo.com/quote/AAPL/history/

# Load, extract Close price, convert to xts format, and plot
# setwd("C:/Teaching/Undergrad/data ECO374")
data_AAPL_full <- read.csv(file="AAPL.csv", header=TRUE, sep=",")
xtsdata_AAPL <- xts(x=data_AAPL_full[c("Close")], order.by=anytime(data_AAPL_full$Date)) 
plot.xts(xtsdata_AAPL, plot.type = c("single"), main ="Apple stock price (Close)" , legend.loc = "topleft", grid.col="lightgray")

# Plot of returns
xtsdata_AAPL_returns <- diff(log(xtsdata_AAPL))
xtsdata_AAPL_returns <- na.omit(xtsdata_AAPL_returns)
plot.xts(
    xtsdata_AAPL_returns,
    plot.type=c("single"),
    main ="Apple stock returns",
    legend.loc = "topleft",
    grid.col="lightgray",
    col="blue"
)

# Dowload data on Freeport-McMoRan Copper stock (FCX) for the last five years, daily frequency
# Source: https://finance.yahoo.com/quote/FCX/history/

# Load, extract Close price, convert to xts format, and plot
# setwd("C:/Teaching/Undergrad/data ECO374")
data_FCX_full <- read.csv(file="FCX.csv", header=TRUE, sep=",")
xtsdata_FCX <- xts(x=data_FCX_full[c("Close")], order.by=anytime(data_FCX_full$Date))
plot.xts(xtsdata_FCX, plot.type = c("single"), main ="FCX stock price (Close)" , legend.loc = "topleft", grid.col="lightgray")

# Plot of returns
xtsdata_FCX_returns <- diff(log(xtsdata_FCX))
xtsdata_FCX_returns <- na.omit(xtsdata_FCX_returns)
plot.xts(xtsdata_FCX_returns, plot.type = c("single"), main ="FCX stock returns" , legend.loc = "topleft", grid.col="lightgray", col="blue")

# Apple returns model
model_AAPL <- garchFit(~arma(1,0)+garch(1,1), data = xtsdata_AAPL_returns, trace = FALSE)
forecast_AAPL <- predict(model_AAPL, plot=FALSE, n.ahead=1)
forecast_AAPL

# FCX returns model
model_FCX <- garchFit(~arma(1,0)+garch(1,1), data = xtsdata_FCX_returns, trace = FALSE)
forecast_FCX <- predict(model_FCX, plot=FALSE, n.ahead=1)
forecast_FCX

# Optimal portfolio weights for daily return of 0.1%
m_set <- 0.1
m1 <- forecast_AAPL$meanForecast
v1 <- forecast_AAPL$standardDeviation^2
m2 <- forecast_FCX$meanForecast
v2 <- forecast_FCX$standardDeviation^2
denominator <- m1^2/v1 + m2^2/v2
w1u <- m_set*(m1^2/v1)/denominator
w2u <- m_set*(m2^2/v2)/denominator
w1 <- w1u/(w1u+w2u)
w2 <- w2u/(w1u+w2u)
w1
w2

# Asset Pricing

# Approximate market returns with S&P500 index
setwd("C:/Teaching/Undergrad/data ECO374")
data_SP500_full <- read.csv(file="SP500_daily.csv", header=TRUE, sep=",")
xtsdata_SP500 <- xts(x=data_SP500_full[c("Close")], order.by=anytime(data_SP500_full$Date))
xtsdata_SP500_returns <- diff(log(xtsdata_SP500))
xtsdata_SP500_returns <- na.omit(xtsdata_SP500_returns)
model_SP <- garchFit(~arma(1,0)+garch(1,1), data = xtsdata_SP500_returns, trace = FALSE)
forecast_SP <- predict(model_SP, plot=FALSE, n.ahead=1)
s_m <- forecast_SP$standardDeviation

# Calculate beta of Apple
s_AAPL <- forecast_AAPL$standardDeviation
beta_AAPL <- 0.45*s_AAPL/s_m
beta_AAPL

# Calculate beta of FCX
s_FCX <- forecast_FCX$standardDeviation
beta_FCX <- 0.28*s_FCX/s_m
beta_FCX
