# ANNEF

>  Artificial Neural Networks in Economic Forecasting

### About this Project

ANNEF is a project focusing on the interdisciplinary areas of economics, computer science, and statistics.

With recent advances in artificial neural networks, ANNs are are shown to be excellent in image recognition and translation tasks. But we found relatively rare work done on examining the power of artificial neural networks on time series tasks.

In this project, we aim to implement a wide range of models, both from structural and non-structural, to forecast various economic indicators, including unemployment and foreign exchange rates.

### Methodology

Artificial neural networks, typically recurrent neural networks, are inherently suitable for capturing the inter-temporal dependency of sequential data.

Also, special types of recurrent units like Long Short Term Memory Unit are designed to grasp relevant information across various lengths of lagged periods.

We use historical observations as the input feed to our neural nets and use the immediately following observation as the target.

#### I. Generating Supervised Learning Problem

To train our model, we firstly convert it into a typical supervised learning problem so we can train neural networks with it.

For a typical univariate time series dataset, $\{x_t\}_{t=0}^T$ 

**Definition** ***Lags** is a positive integer representing the number of time periods our model looks back while make prediction.*

