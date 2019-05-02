
set.seed(2985472)

#---------------------------------------------------------------------
#simulations

n = 200
eps <- 0.5*rnorm(n)
y.sim1 <- 2 + arima.sim(list(ma=c(1.7,0.72)), n, innov = eps)
y.sim2 <- 2 + arima.sim(list(ma=c(-1,0.25)), n, innov = eps)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-1, 5)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim1, ylim=ylims, main=expression(paste("MA(2) with ", theta[1], " = 1.7 and ", theta[2], " = 0.72")))
abline(h=c(2,n))
plot(y.sim2, ylim=ylims, main=expression(paste("MA(2) with ", theta[1], " = -1 and ", theta[2], " = 0.25")))
abline(h=c(2,n))


#---------------------------------------------------------------------
#Autocorrelations and Partial Autocorrelations

ACF1 <- acf(y.sim1, plot = FALSE)
ACF2 <- acf(y.sim2, plot = FALSE)

PACF1 <- pacf(y.sim1, plot = FALSE)
PACF2 <- pacf(y.sim2, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.7, 0.7)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(ACF1[1:10], ylim=ylims, main=expression(paste("ACF: ", theta[1], " = 1.7 and ", theta[2], " = 0.72")) )
plot(ACF2[1:10], ylim=ylims, main=expression(paste("ACF: ", theta[1], " = -1 and ", theta[2], " = 0.25")) )
plot(PACF1[1:10], ylim=ylims, main=expression(paste("PACF: ", theta[1], " = 1.7 and ", theta[2], " = 0.72")) )
plot(PACF2[1:10], ylim=ylims, main=expression(paste("PACF: ", theta[1], " = -1 and ", theta[2], " = 0.25")) )


