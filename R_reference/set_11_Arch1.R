
# Set data directory path
setwd("C:/Teaching/Undergrad/data ECO374")
 
# Install and load required packages
chooseCRANmirror(graphics=FALSE, ind=33)
if (!require("fGarch")) install.packages("fGarch")
library(aTSA)
library(xts)
library(anytime)
library(MASS)
library(moments)
library(fGarch)

# Simulate ARCH(1) with several different \alpha coefficients
spec = garchSpec(model = list(omega = 2, alpha = 0.3, beta = 0))
ARCH1 <- garchSim(spec, n = 1000, extended = TRUE)

spec = garchSpec(model = list(omega = 2, alpha = 0.6, beta = 0))
ARCH2 <- garchSim(spec, n = 1000, extended = TRUE)

spec = garchSpec(model = list(omega = 2, alpha = 0.9, beta = 0))
ARCH3 <- garchSim(spec, n = 1000, extended = TRUE)

par(mfcol=c(1,3), mai=c(0.4, 0.4, 0.7, 0.1))
plot(ARCH1$garch, type="l", ylim=c(-20,20), main=expression(paste("ARCH(1): ",alpha," = 0.3")))
plot(ARCH2$garch, type="l", ylim=c(-20,20), main=expression(paste("ARCH(1): ",alpha," = 0.6")))
plot(ARCH3$garch, type="l", ylim=c(-20,20), main=expression(paste("ARCH(1): ",alpha," = 0.9")))


# Distributions of ARCH draws (alpha = 0.6)
# Note that the distribution is not Normal in a similar way to a typical financial time series.
r_t <- ARCH2$garch
fit <- fitdistr(r_t, "normal")
para <- fit$estimate
hist(r_t, prob = TRUE, breaks=100, xlim=c(-10,10), col="lightgrey", main="Histogram of ARCH draws", cex.main=0.85)
curve(dnorm(x, para[1], para[2]), col = 2, add = TRUE)

# Moments of the ARCH draws (alpha=0.6)
all.moments(r_t, central=TRUE, order.max=4)

# ACF and PACF of the ARCH draws and their square (variance) (alpha=0.6)
maxlag <- 30
ACF <- acf(r_t, main=NULL, lag.max=maxlag, plot=FALSE)
PACF <- pacf(r_t, main=NULL, lag.max=maxlag, plot=FALSE)

ACF_S <- acf(r_t^2, main=NULL, lag.max=maxlag, plot=FALSE)
PACF_S <- pacf(r_t^2, main=NULL, lag.max=maxlag, plot=FALSE)

par(mfcol=c(2,2), mai=c(0.4, 0.4, 0.7, 0.1), cex.main=0.9)
plot(ACF[1:maxlag], ylim=c(-0.1,0.5), main="ACF of ARCH draws")
plot(PACF[1:maxlag], ylim=c(-0.1,0.5),main="PACF of ARCH draws")
plot(ACF_S[1:maxlag], ylim=c(-0.1,0.5), main="ACF of ARCH draws SQUARED")
plot(PACF_S[1:maxlag], ylim=c(-0.1,0.5),main="PACF of ARCH draws SQUARED")

# ACF and PACF of the normalized ARCH draws (variance) (alpha=0.6)
# Note that all the information of the process is contained in the conditional variance.
# Once we normalize with it, there is no more information left.

maxlag <- 20
r_t_normalized = ARCH2$garch / ARCH2$sigma
ACF <- acf(r_t_normalized, main=NULL, lag.max=maxlag, plot=FALSE)
PACF <- pacf(r_t_normalized, main=NULL, lag.max=maxlag, plot=FALSE)

r_t_S_normalized = ARCH2$garch^2 / ARCH2$sigma^2
ACF_S <- acf(r_t_S_normalized, main=NULL, lag.max=maxlag, plot=FALSE)
PACF_S <- pacf(r_t_S_normalized, main=NULL, lag.max=maxlag, plot=FALSE)

par(mfcol=c(2,2), mai=c(0.4, 0.4, 0.7, 0.1), cex.main=0.9)
plot(ACF[1:maxlag], main="ACF of ARCH draws normalized")
plot(PACF[1:maxlag], main="PACF of ARCH draws normalized")
plot(ACF_S[1:maxlag], main="ACF of ARCH draws SQUARED normalized")
plot(PACF_S[1:maxlag], main="PACF of ARCH draws SQUARED normalized")
