plot([1:numTimeStepsTrain], XTrain(1,:), 'Color', 'r', 'LineWidth', 1.3);
hold on;
plot([numTimeStepsTrain + 1: numTimeStepsTrain + length(XTest(1,:))], XTest(1,:), 'Color', 'b', 'LineWidth', 1.3);
