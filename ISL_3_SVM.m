%% Linear Discriminants for Breast Cancer Detection
clc; clear all; close all;
%% Load the Dataset P and T
load('P.mat');
load('T.mat');

%Partition the Dataset for Training and Testing the Linear Discriminants
trainRatio=0.7;
testRatio=0.3;
valRatio=0;

[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,trainRatio,valRatio,testRatio);
[trainInd,valInd,testInd] = divideind(T,trainInd,valInd,testInd);
%whos;
%% Task 1 RBF Linear Polynomial Kernel Function

% Changing Row and column for SVM
trainP = trainP';
testP = testP';
trainInd = trainInd';
testInd = testInd';


% Training the SVM model with RBF KernelFunction

Rbf = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','RBF','KernelScale','auto','BoxConstraint',1);
rbf_pre_train = predict(Rbf,trainP);
rbf_pre_test = predict(Rbf,testP);
% Train Data Confusion Matrix
cm_tr_rbf = confusionmat(rbf_pre_train,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_rbf,[-1 1]);title(['Train Data Confusion Matrix::RBF  BC 1']);

% Test Data Confusion Matrix
cm_tst_rbf = confusionmat(rbf_pre_test,testInd); % confusion matrix
figure;
confusionchart(cm_tst_rbf,[-1 1]);title(['Test Data Confusion Matrix::RBF  BC 1']);

% Training the SVM model with Linear KernelFunction with scale 2

linear_2 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','linear','KernelScale',2,'BoxConstraint',1);
linear_2_pre_train = predict(linear_2,trainP);
linear_2_pre_test = predict(linear_2,testP);
% Train Data Confusion Matrix
cm_trlinear_2 = confusionmat(linear_2_pre_train,trainInd); % confusion matrix
figure;
confusionchart(cm_trlinear_2,[-1 1]);title(['Train Data Confusion Matrix::linear 2 BC 1']);

% Test Data Confusion Matrix
cm_tst_linear_2 = confusionmat(linear_2_pre_test,testInd); % confusion matrix
figure;
confusionchart(cm_tst_linear_2,[-1 1]);title(['Test Data Confusion Matrix::linear 2  BC 1']);


% Training the SVM model with Linear KernelFunction with scale 3

linear_3 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','linear','KernelScale',3,'BoxConstraint',1);
linear_3_pre_train = predict(linear_3,trainP);
linear_3_pre_test = predict(linear_3,testP);
% Train Data Confusion Matrix
cm_trlinear_3 = confusionmat(linear_3_pre_train,trainInd); % confusion matrix
figure;
confusionchart(cm_trlinear_3,[-1 1]);title(['Train Data Confusion Matrix::linear 3  BC 1']);

% Test Data Confusion Matrix
cm_tst_linear_3 = confusionmat(linear_3_pre_test,testInd); % confusion matrix
figure;
confusionchart(cm_tst_linear_3,[-1 1]);title(['Test Data Confusion Matrix::linear 3  BC 1']);


% Training the SVM model with Polynomial KernelFunction with scale 2

Polynomial_2 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','polynomial','KernelScale',2,'BoxConstraint',1);
Polynomial_2_pre_train = predict(Polynomial_2,trainP);
Polynomial_2_pre_test = predict(Polynomial_2,testP);
% Train Data Confusion Matrix
cm_tr_Polynomial_2 = confusionmat(Polynomial_2_pre_train,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_Polynomial_2,[-1 1]);title(['Train Data Confusion Matrix::Polynomial 2  BC 1']);

% Test Data Confusion Matrix
cm_tst_Polynomial_2 = confusionmat(Polynomial_2_pre_test,testInd); % confusion matrix
figure;
confusionchart(cm_tst_Polynomial_2,[-1 1]);title(['Test Data Confusion Matrix::Polynomial 2  BC 1']);


% Training the SVM model with Linear KernelFunction with scale 3

Polynomial_3 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','polynomial','KernelScale',3,'BoxConstraint',1);
Polynomial_3_pre_train = predict(Polynomial_3,trainP);
Polynomial_3_pre_test = predict(Polynomial_3,testP);
% Train Data Confusion Matrix
cm_tr_Polynomial_3 = confusionmat(Polynomial_3_pre_train,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_Polynomial_3,[-1 1]);title(['Train Data Confusion Matrix::Polynomial 3']);

% Test Data Confusion Matrix
cm_tst_Polynomial_3 = confusionmat(Polynomial_3_pre_test,testInd); % confusion matrix
figure;
confusionchart(cm_tst_Polynomial_3,[-1 1]);title(['Test Data Confusion Matrix::Polynomial 3']);



%% Task 2  - Box Constraint for RBF Linear 2 and Polynomial 2

% Training the SVM model with RBF KernelFunction Box - 5
Rbf_BC5 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','RBF','KernelScale','auto','BoxConstraint',5);
rbf_pre_train_BC5 = predict(Rbf_BC5,trainP);
rbf_pre_test_BC5 = predict(Rbf_BC5,testP);
% Train Data Confusion Matrix
cm_tr_rbf_BC5 = confusionmat(rbf_pre_train_BC5,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_rbf_BC5,[-1 1]);title(['Train Data Confusion Matrix::RBF:BC5']);
% Test Data Confusion Matrix
cm_tst_rbf_BC5 = confusionmat(rbf_pre_test_BC5,testInd); % confusion matrix
figure;
confusionchart(cm_tst_rbf_BC5,[-1 1]);title(['Test Data Confusion Matrix::RBF:BC5']);



% Training the SVM model with RBF KernelFunction Box - 250

Rbf_BC250 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','RBF','KernelScale','auto','BoxConstraint',250);
rbf_pre_train_BC250 = predict(Rbf_BC250,trainP);
rbf_pre_test_BC250 = predict(Rbf_BC250,testP);
% Train Data Confusion Matrix
cm_tr_rbf_BC250 = confusionmat(rbf_pre_train_BC250,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_rbf_BC250,[-1 1]);title(['Train Data Confusion Matrix::RBF:BC250']);
% Test Data Confusion Matrix
cm_tst_rbf_BC250  = confusionmat(rbf_pre_test_BC250 ,testInd); % confusion matrix
figure;
confusionchart(cm_tst_rbf_BC250 ,[-1 1]);title(['Test Data Confusion Matrix::RBF:BC250 ']);

% Training the SVM model with Linear KernelFunction with scale 2 Box - 10

linear_2_BC10 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','linear','KernelScale',2,'BoxConstraint',10);
linear_2_pre_train_BC10 = predict(linear_2_BC10,trainP);
linear_2_pre_test_BC10 = predict(linear_2_BC10,testP);
% Train Data Confusion Matrix
cm_trlinear_2_BC10 = confusionmat(linear_2_pre_train_BC10,trainInd); % confusion matrix
figure;
confusionchart(cm_trlinear_2_BC10,[-1 1]);title(['Train Data Confusion Matrix::linear 2:BC10']);

% Test Data Confusion Matrix
cm_tst_linear_2_BC10 = confusionmat(linear_2_pre_test_BC10,testInd); % confusion matrix
figure;
confusionchart(cm_tst_linear_2_BC10,[-1 1]);title(['Test Data Confusion Matrix::linear 2:BC10']);

% Training the SVM model with Linear KernelFunction with scale 2 Box - 1000
linear_2_BC1000 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','linear','KernelScale',2,'BoxConstraint',1000);
linear_2_pre_train_BC1000 = predict(linear_2_BC1000,trainP);
linear_2_pre_test_BC1000 = predict(linear_2_BC1000,testP);
% Train Data Confusion Matrix
cm_trlinear_2_BC1000 = confusionmat(linear_2_pre_train_BC1000,trainInd); % confusion matrix
figure;
confusionchart(cm_trlinear_2_BC1000,[-1 1]);title(['Train Data Confusion Matrix::linear 2:BC1000']);

% Test Data Confusion Matrix
cm_tst_linear_2_BC1000 = confusionmat(linear_2_pre_test_BC1000,testInd); % confusion matrix
figure;
confusionchart(cm_tst_linear_2_BC1000,[-1 1]);title(['Test Data Confusion Matrix::linear 2:BC1000']);


% Training the SVM model with Polynomial KernelFunction with scale 2 Box 10

Polynomial_2_BC10 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','polynomial','KernelScale',2,'BoxConstraint',10);
Polynomial_2_pre_train_BC10 = predict(Polynomial_2_BC10,trainP);
Polynomial_2_pre_test_BC10 = predict(Polynomial_2_BC10,testP);
% Train Data Confusion Matrix
cm_tr_Polynomial_2_BC10 = confusionmat(Polynomial_2_pre_train_BC10,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_Polynomial_2_BC10,[-1 1]);title(['Train Data Confusion Matrix::Polynomial 2:BC10']);

% Test Data Confusion Matrix
cm_tst_Polynomial_2_BC10 = confusionmat(Polynomial_2_pre_test_BC10,testInd); % confusion matrix
figure;
confusionchart(cm_tst_Polynomial_2_BC10,[-1 1]);title(['Test Data Confusion Matrix::Polynomial 2:BC10']);

% Training the SVM model with Polynomial KernelFunction with scale 2 BC 1000

Polynomial_2_BC1000 = fitcsvm(trainP,trainInd,'Standardize',true,'KernelFunction','polynomial','KernelScale',2,'BoxConstraint',1000);
Polynomial_2_pre_train_BC1000 = predict(Polynomial_2_BC1000,trainP);
Polynomial_2_pre_test_BC1000 = predict(Polynomial_2_BC1000,testP);
% Train Data Confusion Matrix
cm_tr_Polynomial_2_BC1000 = confusionmat(Polynomial_2_pre_train_BC1000,trainInd); % confusion matrix
figure;
confusionchart(cm_tr_Polynomial_2_BC1000,[-1 1]);title(['Train Data Confusion Matrix::Polynomial 2:BC1000']);

% Test Data Confusion Matrix
cm_tst_Polynomial_2_BC1000 = confusionmat(Polynomial_2_pre_test_BC1000,testInd); % confusion matrix
figure;
confusionchart(cm_tst_Polynomial_2_BC1000,[-1 1]);title(['Test Data Confusion Matrix::Polynomial 2:BC1000']);
