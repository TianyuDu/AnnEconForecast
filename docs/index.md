# ANNEF

> Artificial Neural Networks in Economic Forecasting

## About this Project

ANNEF is a project focusing on the interdisciplinary areas of economics, computer science, and statistics.

With recent advances in artificial neural networks, ANNs are shown to be excellent in image recognition and translation tasks. However, we found relatively rare work done on examining the power of artificial neural networks on time series tasks.

In this project, we aim to implement a wide range of models, both from structural and non-structural, to forecast various economic indicators, including unemployment and foreign exchange rates.



## Methodology

Artificial neural networks, typically recurrent neural networks, are inherently suitable for capturing the inter-temporal dependency of sequential data.

Also, particular types of recurrent units like Long Short-Term Memory Unit are designed to grasp relevant information across various lengths of lagged periods.

To predict the value of one period, we use historical observations as the feature and train model to fit the value in the target period.

We use various algorithms, typically Adam optimizer to train the model to minimize the mean squared error between predicted values and actual values. 

After each training session, an expanded set of metrics, including mean-squared-error (MSE), mean-absolute-percentage-error(MAPE) and root-mean-square-error(RMSE) are calculated based on predictions from the neural network and the actual series of data. To measure the performance, we also implemented various time series models as benchmarks and comparisons of accuracies are made.

We have implemented several baseline neural networks, including multi-layer LSTM, and detailed demonstrations are available on the demo page.



## Basic Workflow

### I. Data Preprocessing

#### i. Generating a Supervised Learning Problem

In our baseline neural network,  we use a univariate time series as our main dataset.

To train our model, we first convert it into a typical supervised learning problem so we can train neural networks with error minimization oriented algorithms.  With user specified *lag* variable, the supervised learning problem (SLP) generator loops over the entire dataset and for each period, *t*, it marks the series range from *t-lag* to *t-1* as training feature and value at period *t* as the label.

![eq1](http://latex.codecogs.com/svg.latex?3x+1=3)


<p align="center">
  <img width="460" height="300" src="http://latex.codecogs.com/svg.latex?\int 2x^2 dx">
</p>
<div style="text-align:center" markdown="1">
	![eq2](http://latex.codecogs.com/svg.latex?\int 2x^2 dx)
<\div>

By dropping the first few observations in the time series (since we don't have sufficient historical data to make predictions on them), we can generate roughly as many feature-label pairs, say, sample, as the length of time series. ![eq](http://latex.codecogs.com/svg.latex?2x) another sample

#### ii. Splitting SLP

After generating the collection of samples, we split them into three subsets for training, testing and validation purposes. Typically, ratios of 0.6:0.2:0.2 and 0.7:0.15:0.15 are chosen, depends on the total number of observations we have in raw dataset.



### II. Training the Model

In each training session, specific optimizer with given parameters trys to minimize the mean-squared-error between predictions and actual values on the training set only. 

Moreover, loss metrics are evaluated and recorded periodically to avoid over-fitting.

Model structure (graph) and weights are stored after training session finishes.



### III. Evaluating the Model and Visualize

After the training session is finished, we evaluate the performance of the model with various performance and compare them with benchmark models from time series analysis.

Also, a copy of TensorBoard source code is stored together with the model structure and weights after training, one can use the `tensorboard` to navigate the structure of model and detailed metrics on performance.

