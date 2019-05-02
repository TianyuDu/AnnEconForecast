# load package nlme for autocorrelation functions
library(stats)

# simulate white noise (100 draws from N(0,1))
wnoise <- rnorm(n = 100, mean = 0, sd = 1)
plot(wnoise)
lines(wnoise)

# obtain ACF and PACF
lagmax = 10
ACF <- acf(wnoise, lag.max = lagmax, plot = FALSE, demean = TRUE)
plot(ACF[1:lagmax])
ACF

PACF <- pacf(wnoise, lag.max = lagmax, plot = FALSE, demean = TRUE)
plot(PACF[1:lagmax])
PACF

