
set.seed(2985472)

#---------------------------------------------------------------------
#simulations

n = 750
eps <- 0.5*rnorm(n)
c <- 1

phi1 <- 1
phi2 <- -0.5
wn <- ts(rnorm(n))
y.sim1 <- as.vector(seq(0,0,length.out=n))
y.sim1[1] <- wn[1]
y.sim1[2] <- c + phi1*y.sim1[t-1] + wn[2]
for(t in 3:n){
   y.sim1[t] <- c + phi1*y.sim1[t-1] + phi2*y.sim1[t-2] + wn[t]
}
y.sim1 <- y.sim1[601:n]

phi1 <- -0.5
phi2 <- 0.4
wn <- ts(rnorm(n))
y.sim2 <- as.vector(seq(0,0,length.out=n))
y.sim2[1] <- wn[1]
y.sim2[2] <- c + phi1*y.sim2[t-1] + wn[2]
for(t in 3:n){
   y.sim2[t] <- c + phi1*y.sim2[t-1] + phi2*y.sim2[t-2] + wn[t]
}
y.sim2 <- y.sim2[601:n]

phi1 <- 0.5
phi2 <- 0.3
wn <- ts(rnorm(n))
y.sim3 <- as.vector(seq(0,0,length.out=n))
y.sim3[1] <- wn[1]
y.sim3[2] <- c + phi1*y.sim3[t-1] + wn[2]
for(t in 3:n){
   y.sim3[t] <- c + phi1*y.sim3[t-1] + phi2*y.sim3[t-2] + wn[t]
}
y.sim3 <- y.sim3[601:n]

phi1 <- 0.5
phi2 <- 0.5
wn <- ts(rnorm(n))
y.sim4 <- as.vector(seq(0,0,length.out=n))
y.sim4[1] <- wn[1]
y.sim4[2] <- c + phi1*y.sim4[t-1] + wn[2]
for(t in 3:n){
   y.sim4[t] <- c + phi1*y.sim4[t-1] + phi2*y.sim4[t-2] + wn[t]
}
y.sim4 <- y.sim4[601:n]

#---------------------------------------------------------------------
# Plots

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim1, type="l", main=expression(paste("AR(2) with ", phi[1], " = 1, ", phi[2], " = -0.5;      ", phi[1],"+",phi[2],"= 0.5")))
abline(h=c(1/(1-1+0.5),n))
plot(y.sim2, type="l", main=expression(paste("AR(2) with ", phi[1], " = -0.5, ", phi[2], " = 0.4;      ", phi[1],"+",phi[2],"= -0.1")))
abline(h=c(1/(1+0.5-0.4),n))
plot(y.sim3, type="l", main=expression(paste("AR(2) with ", phi[1], " = 0.5, ", phi[2], " = 0.3;      ", phi[1],"+",phi[2],"= 0.8")))
abline(h=c(1/(1-0.5-0.3),n))
plot(y.sim4, type="l", main=expression(paste("AR(2) with ", phi[1], " = 0.5, ", phi[2], " = 0.5;      ", phi[1],"+",phi[2],"= 1")))

#---------------------------------------------------------------------
# Autocorrelations

ACF1 <- acf(y.sim1, plot = FALSE)
ACF2 <- acf(y.sim2, plot = FALSE)
ACF3 <- acf(y.sim3, plot = FALSE)
ACF4 <- acf(y.sim4, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
ylims = c(-0.2, 1)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(ACF1[1:20], main=expression(paste("AR(2) with ", phi[1], " = 1, ", phi[2], " = -0.5;      ", phi[1],"+",phi[2],"= 0.5")))
plot(ACF2[1:20], main=expression(paste("AR(2) with ", phi[1], " = -0.5, ", phi[2], " = 0.4;      ", phi[1],"+",phi[2],"= -0.1")))
plot(ACF3[1:20], main=expression(paste("AR(2) with ", phi[1], " = 0.5, ", phi[2], " = 0.3;      ", phi[1],"+",phi[2],"= 0.8")))
plot(ACF4[1:20], main=expression(paste("AR(2) with ", phi[1], " = 0.5, ", phi[2], " = 0.5;      ", phi[1],"+",phi[2],"= 1")))

#---------------------------------------------------------------------
# Partial Autocorrelations

PACF1 <- pacf(y.sim1, plot = FALSE)
PACF2 <- pacf(y.sim2, plot = FALSE)
PACF3 <- pacf(y.sim3, plot = FALSE)
PACF4 <- pacf(y.sim4, plot = FALSE)

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
#ylims = c(-0.2, 1)
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(PACF1[1:20], main=expression(paste("AR(2) with ", phi[1], " = 1, ", phi[2], " = -0.5;      ", phi[1],"+",phi[2],"= 0.5")))
plot(PACF2[1:20], main=expression(paste("AR(2) with ", phi[1], " = -0.5, ", phi[2], " = 0.4;      ", phi[1],"+",phi[2],"= -0.1")))
plot(PACF3[1:20], main=expression(paste("AR(2) with ", phi[1], " = 0.5, ", phi[2], " = 0.3;      ", phi[1],"+",phi[2],"= 0.8")))
plot(PACF4[1:20], main=expression(paste("AR(2) with ", phi[1], " = 0.5, ", phi[2], " = 0.5;      ", phi[1],"+",phi[2],"= 1")))



