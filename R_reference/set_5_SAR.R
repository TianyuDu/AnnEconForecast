
set.seed(254896)

#---------------------------------------------------------------------
#simulations

n = 140
eps <- rnorm(n)
c <- 0

phi1 <- 0.8
wn <- ts(rnorm(n))
y.sim1 <- as.vector(seq(0,0,length.out=n))
y.sim1[1:4] <- wn[1:4]
for(t in 5:n){
   y.sim1[t] <- c + phi1*y.sim1[t-4] + wn[t]
}
y.sim1 <- y.sim1[51:n]

ACF <- acf(y.sim1, plot = FALSE)
PACF <- pacf(y.sim1, plot = FALSE)

#---------------------------------------------------------------------
# Plots, ACF, PACF

layout(matrix(c(1,2,3,4), 2, 2, byrow=TRUE))
par(mai=c(0.4, 0.4, 0.5, 0.1))
plot(y.sim1, type="l", main=expression(paste("S-AR(1) with ", phi, " = 0.8")))
abline(h=c(0,100))
plot(ACF[1:20], main=expression(paste("ACF")))
plot(PACF[1:20], main=expression(paste("PACF")))

