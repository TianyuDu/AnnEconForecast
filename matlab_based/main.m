%{
May 10. 2018
LSTM Recurrent Neural Network for forcasting CPI
Consumer Price Index for All Urban Consumers: All Items.
Monthly data, seasonally adjusted
from 1953-04-01 to 2018-02-01
%}
clear all;
clc;

%% Loading data from source;
filename = "CPIAUCSL.csv";
data = csvread(filename, 1, 1);
data = data'; % Row time time series data.

%% Initialize figure;
figure
plot(data)
grid on
xlabel("Date")
ylabel("Consumer Price Index")
% Consumer Price Index for All Urban Consumers: All Items
title("Consumer Price Index for All Urban Consumers: All Items")

%% Creating sub-dataset;
numTimeStepsTrain = floor(0.9 * numel(data));
XTrain = data(1: numTimeStepsTrain);
YTrain = data(2: numTimeStepsTrain + 1);
% One time step forward forcesting.
XTest = data(numTimeStepsTrain + 1: end -1);
YTest = data(numTimeStepsTrain + 2: end);

%% Standardize Data;
% Mean normalization.
mu = mean(XTrain);
sig = std(XTrain);

XTrain = (XTrain - mu) ./ sig;
YTrain = (YTrain - mu) ./ sig;

XTest = (XTest - mu) ./ sig;

%% Setup LSTM;
inputSize = 1; % Dimension of input sequence.
numResponses = 1; % Dimension of output sequence.
numHiddenUnits.L1 = 64;
numHiddenUnits.L2 = 32;

layers = [...
	sequenceInputLayer(inputSize)
	lstmLayer(numHiddenUnits.L1)
	fullyConnectedLayer(numResponses)
	regressionLayer
	];

opts = trainingOptions(...
	"adam",...
	"MaxEpochs", 1500, ...
	"GradientThreshold", 1, ...
	"InitialLearnRate", 0.005, ...
	"LearnRateSchedule", "piecewise", ...
	"LearnRateDropPeriod", 125, ...
	"LearnRateDropFactor", 0.2, ...
	"Verbose", 0, ...
	"Plot", "training-progress");

%% Training;
net = trainNetwork(...
	XTrain, ...
	YTrain, ...
	layers, ...
	opts);

%% Update netowrk state with observed values
net = predictAndUpdateState(net, XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);

for i = 1:numTimeStepsTest
	[net, YPred(1, i)] = predictAndUpdateState(net, XTest(i));
end

%% Unstandardize result
YPred = sig * YPred + mu;
rmse = sqrt(mean((YPred - YTest) .^ 2));

%% Visualize
figure
subplot(2, 1, 1)
plot(YTest)
hold on
plot(YPred, ".-")
grid on
hold off
legend(["Observed" "Predicted"])
ylabel("CPI value")
title("Consumer Price Index for All Urban Consumers: All Items")

subplot(2, 1, 2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
