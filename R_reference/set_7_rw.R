
library(aTSA)

set.seed(9847354)

n = 500
eps <- rnorm(n)
c <- 0.5
phi1 <- 1
wn <- ts(rnorm(n))

#---------------------------------------------------------------------
# simulate random walk without a drift using AR(1)

y.sim1 <- as.vector(seq(0,0,length.out=n))
y.sim1[1] <- 0
for(t in 2:n){
   y.sim1[t] <- phi1*y.sim1[t-1] + wn[t]
}
plot(y.sim1, type="l", main="Random Walk without a Drift")
abline(h=c(0,n))

#---------------------------------------------------------------------
# simulate random walk with a drift using AR(1)

y.sim2 <- as.vector(seq(0,0,length.out=n))
y.sim2[1] <- 0
for(t in 2:n){
   y.sim2[t] <- c + phi1*y.sim2[t-1] + wn[t]
}
plot(y.sim2, type="l", main="Random Walk with a Drift")
abline(a=0, b=c, col = "red")

#----------------------------------------------------------------------
# simulate random walk without a drift using AR(2)

phi1 <- 0.5
phi2 <- 0.5
wn <- ts(rnorm(n))
y.sim3 <- as.vector(seq(0,0,length.out=n))
y.sim3[1] <- 0
y.sim3[2] <- c + phi1*y.sim3[t-1] + wn[2]
for(t in 3:n){
   y.sim3[t] <- phi1*y.sim3[t-1] + phi2*y.sim3[t-2] + wn[t]
}
plot(y.sim3, type="l", main="Random Walk without a Drift")
abline(h=c(0,n))

#----------------------------------------------------------------------
# autocorrelations
ACF1 <- acf(y.sim1, lag.max=100, plot = FALSE)
ACF2 <- acf(y.sim2, lag.max=100, plot = FALSE)
PACF1 <- pacf(y.sim1, plot = FALSE)
PACF2 <- pacf(y.sim2, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.2, 1)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(ACF1[1:100], main="ACF of random walk without a drift")
plot(ACF2[1:100], main="ACF of random walk without a drift")
plot(PACF1[1:10], main="PACF of random walk without a drift")
plot(PACF2[1:10], main="PACF of random walk without a drift")

#----------------------------------------------------------------------
# unit root tests

adf.test(y.sim1)

adf.test(y.sim2)

adf.test(y.sim3)

#----------------------------------------------------------------------
# simulate a stationary time series and verify ADF unit root test

phi1 <- 0.5
y.sim4 <- as.vector(seq(0,0,length.out=n))
y.sim4[1] <- 0
for(t in 2:n){
   y.sim4[t] <- phi1*y.sim4[t-1] + wn[t]
}
plot(y.sim4, type="l", main="Stationary Time Series")
abline(h=c(0,n))

adf.test(y.sim4)


