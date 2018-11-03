## Plan

### Basic statistical models about time series

* ARMA (autoregression moving average)
* ARIMA (autoregression integrated moving average)
* VAR (vector autoregression)

### Univariate Models

* Villain neural net with lagged variables as input feed. i.e. use $(x_{t-k}, x_{t-k+1}, \dots ,x_{t-1})$ to predict $x_t$. 

* Recurrent neural net on univariate time series.

###Multivariate Models

* Add concurrent series (e.g. other economic indicators) (So that the time series dataset is a sequence of vectors, with dimension of total number of dimensions included.)

* Villain neural net with lagged variables as input feed.

* Recurrent neural net on vectors.

* Consider different loss metrics.

### Multivariate Models with Feature Extractors(Selection).

* Use importances of gradient boosting machines.

* Use CNN as feature extractors.

