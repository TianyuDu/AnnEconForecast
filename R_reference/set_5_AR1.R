
set.seed(2985472)

#---------------------------------------------------------------------
#simulations

n = 200
eps <- 0.5*rnorm(n)
c <- 1
phi <- 0.4
y.sim1 <- c/(1-phi) + arima.sim(list(ar=phi), n, innov = eps)
phi <- 0.7
y.sim2 <- c/(1-phi) + arima.sim(list(ar=phi), n, innov = eps)
phi <- 0.95
y.sim3 <- c/(1-phi) + arima.sim(list(ar=phi), n, innov = eps)

y.sim4 <- as.vector(seq(0,0,length.out=n))
wn <- ts(rnorm(n))
y.sim4[1] <- wn[1]
for(i in 2:n){
   y.sim4[i] <- 1 + y.sim4[i-1] + wn[i]
}

#---------------------------------------------------------------------
# Plots

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim1, main=expression(paste("AR(1) with ", phi, " = 0.4")))
abline(h=c(1/(1-0.4),n))
plot(y.sim2, main=expression(paste("AR(1) with ", phi, " = 0.7")))
abline(h=c(1/(1-0.7),n))
plot(y.sim3, main=expression(paste("AR(1) with ", phi, " = 0.95")))
abline(h=c(1/(1-0.95),n))
plot(y.sim4, main=expression(paste("AR(1) with ", phi, " = 1")))

#---------------------------------------------------------------------
# Autocorrelations

ACF1 <- acf(y.sim1, plot = FALSE)
ACF2 <- acf(y.sim2, plot = FALSE)
ACF3 <- acf(y.sim3, plot = FALSE)
ACF4 <- acf(y.sim4, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.2, 1)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(ACF1[1:20], ylim=ylims, main=expression(paste("ACF: ", phi, " = 0.4")) )
plot(ACF2[1:20], ylim=ylims, main=expression(paste("ACF: ", phi, " = 0.7")) )
plot(ACF3[1:20], ylim=ylims, main=expression(paste("ACF: ", phi, " = 0.95")) )
plot(ACF4[1:20], ylim=ylims, main=expression(paste("ACF: ", phi, " = 1")) )

#---------------------------------------------------------------------
# Partial Autocorrelations

PACF1 <- pacf(y.sim1, plot = FALSE)
PACF2 <- pacf(y.sim2, plot = FALSE)
PACF3 <- pacf(y.sim3, plot = FALSE)
PACF4 <- pacf(y.sim4, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.2, 1)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(PACF1[1:20], ylim=ylims, main=expression(paste("PACF: ", phi, " = 0.4")) )
plot(PACF2[1:20], ylim=ylims, main=expression(paste("PACF: ", phi, " = 0.7")) )
plot(PACF3[1:20], ylim=ylims, main=expression(paste("PACF: ", phi, " = 0.95")) )
plot(PACF4[1:20], ylim=ylims, main=expression(paste("PACF: ", phi, " = 1")) )

#------------------------------------------------------------------------
# Negative phi

phi <- -0.95
y.sim5 <- c/(1-phi) + arima.sim(list(ar=phi), n, innov = eps)
ACF5 <- acf(y.sim5, plot = FALSE)
PACF5 <- pacf(y.sim5, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim5, main=expression(paste("AR(1) with ", phi, " = -0.95")))
abline(h=c(1/(1+0.95),n))
plot(ACF5[1:20], ylim=c(-1,1), main=expression(paste("ACF: ", phi, " = -0.95")) )
plot(PACF5[1:10], ylim=c(-1,1), main=expression(paste("PACF: ", phi, " = -0.95")) )




