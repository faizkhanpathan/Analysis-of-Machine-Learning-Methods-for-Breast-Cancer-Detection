%% Linear Discriminants for Breast Cancer Detection
clc; clear all; close all;
%% Load the Dataset P and T
load('P.mat');
load('T.mat');

%% Partition the Dataset for Training and Testing the Linear Discriminants
trainRatio=0.7;
testRatio=0.3;
valRatio=0;

[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,trainRatio,valRatio,testRatio);
[trainInd,valInd,testInd] = divideind(T,trainInd,valInd,testInd);
%whos;

%% Separate (-1) Healthy and (1) Affected Patient data
H_train=trainP(:,(find(trainInd==-1))); % Healthy (-1)
A_train=trainP(:,(find(trainInd==1))); % Affected
H_test=testP(:,(find(testInd==-1))); % Healthy (-1)
A_test=testP(:,(find(testInd==1))); % Affected
H_train_T=trainInd(:,(find(trainInd==-1))); % Healthy (-1)
A_train_T=trainInd(:,(find(trainInd==1))); % Affected
H_test_T=testInd(:,(find(testInd==-1))); % Healthy (-1)
A_test_T=testInd(:,(find(testInd==1))); % Affected

% Label 0 and 1
H_tr_len=numel(H_train_T); % Length of Healthy dataset
A_tr_len=numel(A_train_T); % Length of Affected dataset
H_tst_len=numel(H_test_T);
A_tst_len=numel(A_test_T);
L_tr = [zeros(1,H_tr_len) ones(1,A_tr_len)]; % Train data label
L_tst = [zeros(1,H_tst_len) ones(1,A_tst_len)]; % Test data label
X_train=[H_train,A_train]'; 
X_test=[H_test,A_test]';
%% Task 1- Different Classifier
% 'linear', ’quadratic’, ‘diagLinear’, and ‘diagQuadratic’,
reduc = 'No PCA';
discriminator = 'linear';
discriminant_analy(X_train,X_test,L_tr,L_tst,discriminator,reduc)
discriminator = 'quadratic';
discriminant_analy(X_train,X_test,L_tr,L_tst,discriminator,reduc)
discriminator = 'diagLinear';
discriminant_analy(X_train,X_test,L_tr,L_tst,discriminator,reduc)
discriminator = 'diagQuadratic';
discriminant_analy(X_train,X_test,L_tr,L_tst,discriminator,reduc)
disp('------------------------------------------------------');

%% Task 2 PCA with 99% Variance

[PCAcoeff_train,PCAscore_train,latent_train,~,explained_train,mu_train]=pca(X_train); 
figure, plot(cumsum(explained_train)), title('Scree Plot Train data');
Xpca_train_per_99=PCAscore_train(:,1:2); % 2 coulms 99 % 
Actual_len=size(X_train);
feature_length = size(Xpca_train_per_99);
fprintf('Reduced train data Dimensionality: %d x %d \n',feature_length(1), feature_length(2));
[PCAcoeff_test,PCAscore_test,latent_test,~,explained_test,mu_test]=pca(X_test); 
figure, plot(cumsum(explained_test)), title('Scree Plot Test data');
Xpca_test_per_99=PCAscore_test(:,1:2); %%

% 'linear', ’quadratic’, ‘diagLinear’, and ‘diagQuadratic’,
disp('----------PCA  with 99% Variance Explained-----------');
disp('------------------------------------------------------');
reduc = 'PCA 99';
discriminator = 'linear';
discriminant_analy(Xpca_train_per_99,Xpca_test_per_99,L_tr,L_tst,discriminator,reduc)
discriminator = 'quadratic';
discriminant_analy(Xpca_train_per_99,Xpca_test_per_99,L_tr,L_tst,discriminator,reduc)
discriminator = 'diagLinear';
discriminant_analy(Xpca_train_per_99,Xpca_test_per_99,L_tr,L_tst,discriminator,reduc)
discriminator = 'diagQuadratic';
discriminant_analy(Xpca_train_per_99,Xpca_test_per_99,L_tr,L_tst,discriminator,reduc)
disp('------------------------------------------------------');
disp('------------------------------------------------------');
Xpca_test_per_99_99=PCAscore_test(:,1:4); %%
Xpca_train_per_99_99=PCAscore_train(:,1:4); % 2 coulms 99 % 
feature_length = size(Xpca_train_per_99_99);
fprintf('Reduced train data Dimensionality: %d x %d \n',feature_length(1), feature_length(2));
% 'linear', ’quadratic’, ‘diagLinear’, and ‘diagQuadratic’,
disp('----------PCA  with 99.99% Variance Explained-----------');
reduc = 'PCA 99.99';
discriminator = 'linear';
discriminant_analy(Xpca_train_per_99_99,Xpca_test_per_99_99,L_tr,L_tst,discriminator,reduc)
discriminator = 'quadratic';
discriminant_analy(Xpca_train_per_99_99,Xpca_test_per_99_99,L_tr,L_tst,discriminator,reduc)
discriminator = 'diagLinear';
discriminant_analy(Xpca_train_per_99_99,Xpca_test_per_99_99,L_tr,L_tst,discriminator,reduc)
discriminator = 'diagQuadratic';
discriminant_analy(Xpca_train_per_99_99,Xpca_test_per_99_99,L_tr,L_tst,discriminator,reduc)



%%
function discriminant_analy(X_train,X_test,L_tr,L_tst,discriminator,reduc)


Mdl = fitcdiscr(X_train,L_tr,'DiscrimType',discriminator); % 
[label,score] = predict(Mdl,X_train);
score_tr=score(:,2)';
label_tr=label';



%% ROC for Train Data
[X,Y,~,AUC_tr]=perfcurve(L_tr,score_tr,1);
figure;
plot(X,Y);
xlabel('FAR'),ylabel('GAR'),title(['AUC=' num2str(AUC_tr) ' :: Classif.: ' discriminator ' : Dim. red.:' reduc] )

%% Train Data Confusion Matrix
cm_tr = confusionmat(L_tr,label_tr); % confusion matrix
figure, plotconfusion(L_tr,label_tr);
title(['Train Data Confusion Matrix::Classif.: ' discriminator ' : Dim. red.:' reduc]);
% TP TN FP FN
TP=cm_tr(1,1);
TN=cm_tr(2,2);
FP=cm_tr(1,2);
FN=cm_tr(2,1);
Acc_tr= ((TP + TN) / (TP + TN + FN + FP))*100; % Accuracy
fprintf(' Discriminator Type: %s \n',discriminator);
fprintf('Train data Accuracy %0.2f \n',Acc_tr);
data_cm_tr=table(TP,TN,FP,FN,Acc_tr);
disp(data_cm_tr);
%%

%Mdl = fitcdiscr(X_test,L_tst,'DiscrimType',discriminator); % 
[label,score] = predict(Mdl,X_test);
score_tst=score(:,2)';
label_tst=label';

%% ROC for Test Data
[X,Y,~,AUC_tst]=perfcurve(L_tst,score_tst,1);
figure;
plot(X,Y);
xlabel('FAR'),ylabel('GAR'),title(['AUC=' num2str(AUC_tst) ' :: Classif.: ' discriminator ' : Dim. red.:' reduc ] )

%% Test Data Confusion Matrix
cm_tst = confusionmat(L_tst,label_tst); % confusion matrix
figure, plotconfusion(L_tst,label_tst);
title(['Test Data Confusion Matrix:: Classif. ' discriminator '  : Dim. red.:' reduc]);
% TP TN FP FN
TP=cm_tst(1,1);
TN=cm_tst(2,2);
FP=cm_tst(1,2);
FN=cm_tst(2,1);
Acc_tst= ((TP + TN) / (TP + TN + FN + FP))*100; % Accuracy
fprintf(' Discriminator Type: %s \n',discriminator);
fprintf('Test data Accuracy %0.2f \n',Acc_tst);
data_cm_tst=table(TP,TN,FP,FN,Acc_tst);
disp(data_cm_tst);



end
