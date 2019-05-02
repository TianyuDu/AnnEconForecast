
set.seed(2985472)

#---------------------------------------------------------------------
#simulations

n = 200
eps <- 0.5*rnorm(n)
y.sim1 <- 2 + arima.sim(list(ma=0.05), n, innov = eps)
y.sim2 <- 2 + arima.sim(list(ma=0.5), n, innov = eps)
y.sim3 <- 2 + arima.sim(list(ma=0.95), n, innov = eps)
y.sim4 <- 2 + arima.sim(list(ma=2), n, innov = eps)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-1, 5)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim1, ylim=ylims, main=expression(paste("MA(1) with ", theta, " = 0.05")))
abline(h=c(2,n))
plot(y.sim2, ylim=ylims, main=expression(paste("MA(1) with ", theta, " = 0.5")))
abline(h=c(2,n))
plot(y.sim3, ylim=ylims, main=expression(paste("MA(1) with ", theta, " = 0.95")))
abline(h=c(2,n))
plot(y.sim4, ylim=ylims, main=expression(paste("MA(1) with ", theta, " = 2")))
abline(h=c(2,n))

#---------------------------------------------------------------------
#Autocorrelations

ACF1 <- acf(y.sim1, plot = FALSE)
ACF2 <- acf(y.sim2, plot = FALSE)
ACF3 <- acf(y.sim3, plot = FALSE)
ACF4 <- acf(y.sim4, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.6, 0.6)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(ACF1[1:10], ylim=ylims, main=expression(paste("ACF: ", theta, " = 0.05")) )
plot(ACF2[1:10], ylim=ylims, main=expression(paste("ACF: ", theta, " = 0.5")) )
plot(ACF3[1:10], ylim=ylims, main=expression(paste("ACF: ", theta, " = 0.95")) )
plot(ACF4[1:10], ylim=ylims, main=expression(paste("ACF: ", theta, " = 2")) )

#---------------------------------------------------------------------
#Partial Autocorrelations

PACF1 <- pacf(y.sim1, plot = FALSE)
PACF2 <- pacf(y.sim2, plot = FALSE)
PACF3 <- pacf(y.sim3, plot = FALSE)
PACF4 <- pacf(y.sim4, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.6, 0.6)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(PACF1[1:10], ylim=ylims, main=expression(paste("PACF: ", theta, " = 0.05")) )
plot(PACF2[1:10], ylim=ylims, main=expression(paste("PACF: ", theta, " = 0.5")) )
plot(PACF3[1:10], ylim=ylims, main=expression(paste("PACF: ", theta, " = 0.95")) )
plot(PACF4[1:10], ylim=ylims, main=expression(paste("PACF: ", theta, " = 2")) )

