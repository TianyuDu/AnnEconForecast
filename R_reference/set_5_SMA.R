
set.seed(254896)

#---------------------------------------------------------------------
#simulations

n = 300
eps <- 0.5*rnorm(n)
c <- 0

theta <- 0.8
wn <- ts(rnorm(n))
y.sim1 <- as.vector(seq(0,0,length.out=n))
y.sim1[1:4] <- wn[1:4]
for(t in 5:n){
   y.sim1[t] <- c + phi1*wn[t-4] + wn[t]
}
y.sim1 <- y.sim1[101:n]

ACF <- acf(y.sim1, plot = FALSE)
PACF <- pacf(y.sim1, plot = FALSE)

#---------------------------------------------------------------------
# Plots, ACF, PACF

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim1, type="l", main=expression(paste("S-MA(1) with ", theta, " = 0.6")))
abline(h=c(0,100))
plot(ACF[1:20], main=expression(paste("ACF")))
plot(PACF[1:20], main=expression(paste("PACF")))

