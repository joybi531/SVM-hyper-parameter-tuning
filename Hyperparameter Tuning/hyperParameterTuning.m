clear all;
clc;

load trainTest

uniqueClass = unique(trainClass);
predictionClass = ones(size(testClass,1),1);

modelTrain = tic;
count = 0;
for i = 1:size(uniqueClass)
    eachLoop = tic;
    groups = ismember(trainClass, uniqueClass(i));
    mdl = fitcsvm(trainData, groups, 'KernelFunction', 'rbf',...
        'OptimizeHyperparameters', 'auto',....
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName',...
        'expected-improvement-plus', 'ShowPlots', false));
    count = count + 1;
    
    [predictionList, predictionScore] = predict(mdl, testData);
    
%     predictionScore = predictionScore*(-1);
    if (i == 1)
        score = predictionScore;
    else
        for j = 1:size(testClass,1)
            if (score(j,1) > predictionScore(j,1))
                score(j,1)  = predictionScore(j,1);
                predictionClass(j) = uniqueClass(i);
            end
        end
    end

    eachLooptakesTime = toc(eachLoop);
end
timeTakenForTraining = toc(modelTrain)

%         for i = 1:size(uniqueClass)
%         end

cp = classperf(testClass, predictionClass);
Accuracy = cp.CorrectRate;

plotTime = tic;

figure;
hgscatter = gscatter(trainData(:,1), trainData(:,2), trainClass);
hold on;

h_sv = plot(mdl.SupportVectors(:,1), mdl.SupportVectors(:,2), 'ko', 'markersize', 4);

timeTakenForPlotting = toc(plotTime)


save hyperParameters mdl predictionList predictionScore predictionClass Accuracy
