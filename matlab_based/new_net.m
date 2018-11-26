clear;
close all;
clc;
addpath("data");

load_data_from_csv;

%% Model Settings
% Consider transfer this section to an isolated parameter setting script

% Precentage of data used as training data.
modelParameter.trainingSize = 0.7;
% Index of RESPONSE variable in data array.
modelParameter.responseIndex = 1;

%% Initial visualization
num_plot = length(names);
plot_size = ceil(sqrt(num_plot));

disp(['Data range from ', ...
	datestr(data.tt.DATE(1)), ...
	' to ', ...
	datestr(data.tt.DATE(end))]);
for i = 1:num_plot
    subplot(plot_size, plot_size, i);
    plot(data.tt.DATE, (data.ar(:, i)), 'LineWidth', 1.2);
    legend(names{i});
end

%% Creating data sets
sampleSize = height(data.tt);
numFeatures = width(data.tt);
% Total number of time point values.

trainingSize = floor(...
    sampleSize * modelParameter.trainingSize ...
	);

% Training Data for X.
x_ar = data.ar';  % Traning data X
y_ar = data.tt.UNRATE';  % Response Data Y.

figure;
hold on;
title('Value of samples, normalized to mid values.');
mid_idx = floor(sampleSize / 2);
plot(data.tt.DATE, x_ar ./ x_ar(:, mid_idx), 'LineWidth', 1.2);
legend(names);
hold off;


%% Indexing, DATA(dataIndex, timeStep)

XTrain = x_ar(:, ...
    (1:trainingSize));

YTrain = y_ar(1, ...
    (2:trainingSize + 1));  % Label, one step forwards.

XTest = x_ar(:, ...
    (trainingSize + 1: end - 1));

YTest = y_ar(1, ...
    trainingSize + 2: end);

%% Standardize data (Mean Normalization)
mu = mean(XTrain, 2);
sig = std(XTrain, 0, 2);

XTrain = (XTrain - mu) ./ sig;
XTest = (XTest - mu) ./ sig;

%%
YTrain = (YTrain - mu(1)) ./ sig(1);


%% Setup neural network
inputSize = numFeatures;
numResponses = 1;

numHiddenUnits.lstm1 = 64;
numHiddenUnits.lstm2 = 32;
numHiddenUnits.fc1 = 32;
numHiddenUnits.fc2 = 32;

% layers = [...
% 	sequenceInputLayer(inputSize, "Name", "Input layer")
% 	lstmLayer(numHiddenUnits.lstm1, "Name", "LSTM layer 1")
% 	lstmLayer(numHiddenUnits.lstm2, "Name", "LSTM layer 2")
% 	fullyConnectedLayer(numHiddenUnits.fc1, "Name", "FC layer 1")
% 	fullyConnectedLayer(numResponses, "Name", "Output layer")
% 	regressionLayer
% 	];

layers = [...
	sequenceInputLayer(inputSize, 'Name', 'Input layer')
	lstmLayer(numHiddenUnits.lstm1, 'Name', 'LSTM layer 1')
%         lstmLayer(numHiddenUnits.lstm2, 'Name', 'LSTM layer 2')
	fullyConnectedLayer(numHiddenUnits.fc1, 'Name', 'FC layer 1')
% 	fullyConnectedLayer(numHiddenUnits.fc2, 'Name', 'FC layer 2')
 	fullyConnectedLayer(numResponses, 'Name', 'Output layer')
	regressionLayer
	];

testLayers = [...
	sequenceInputLayer(inputSize)
	lstmLayer(64)
	fullyConnectedLayer(numResponses)
	regressionLayer
	];

opts = trainingOptions(...
	'adam',...
	'MaxEpochs', 50000, ...
	'GradientThreshold', 10000, ...
	'InitialLearnRate', 0.5, ...
	'LearnRateSchedule', 'piecewise', ...
	'LearnRateDropPeriod', 2000, ...
	'LearnRateDropFactor', 0.5, ...
    'MiniBatchSize', 72, ...
	'Verbose', 1);

%% Training
net = trainNetwork(...
	XTrain, ...
	YTrain, ...
	layers, ...
	opts);

%% Predicting
net = predictAndUpdateState(net, XTrain);

YPred = [];

numTimeStepsTest = length(XTest);

for i = 1:numTimeStepsTest
	[net, YPred(1, i)] = predictAndUpdateState(net, XTest(:, i));
end

YPred = sig(1) * YPred + mu(1);

rmse = sqrt(mean(YPred - YTest) .^ 2);

YTrain = sig(1) * YTrain + mu(1);

%% Visualize
figure
subplot(2, 1, 1)
plot(YTest, "-", "LineWidth", 1.25)
hold on
plot(YPred, "-", "LineWidth", 1.25)
grid on
hold off
legend(["Observed" "Predicted"])
ylabel("UNRATE(%)")
title("Civilian Unemployment Rate")

subplot(2, 1, 2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
