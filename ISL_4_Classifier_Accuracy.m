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

% Changing Row and column for SVM
trainP_svm = trainP';
testP_svm = testP';
trainInd_svm = trainInd';
testInd_svm = testInd';



%%  Getting Scores from different Classifiers
[w ,J, m ,reg]=LDA3(H_train,A_train);
H_tr_W=w'*H_train-w'*m;
A_tr_W=w'*A_train-w'*m;
H_tst_W=w'*H_test-w'*m;
A_tst_W=w'*A_test-w'*m;
score_tr_LDA=[H_tr_W A_tr_W];
score_tst_LDA=[H_tst_W A_tst_W];

discriminator = 'linear';
[score_tr_lin,score_tst_lin] = discriminant_analy(X_train,X_test,L_tr,discriminator);
discriminator = 'quadratic';
[score_tr_quad,score_tst_quad] =discriminant_analy(X_train,X_test,L_tr,discriminator);
discriminator = 'diagLinear';
[score_tr_diag_lin,score_tst_diag_lin] =discriminant_analy(X_train,X_test,L_tr,discriminator);
discriminator = 'diagQuadratic';
[score_tr_diag_q,score_tst_diag_q] =discriminant_analy(X_train,X_test,L_tr,discriminator);

% Training the SVM model with Polynomial KernelFunction with scale 2

Polynomial_2 = fitcsvm(trainP_svm,trainInd_svm,'Standardize',true,'KernelFunction','polynomial','KernelScale',2,'BoxConstraint',1);
[Polynomial_2_pre_train, score_tr_ply] = predict(Polynomial_2,trainP_svm);
[Polynomial_2_pre_test, score_tst_ply] = predict(Polynomial_2,testP_svm);

score_tr_svm = score_tr_ply(:,2)';
score_tst_svm = score_tst_ply(:,2)';
%% Normalizing Data
[Y,PS]=mapminmax(score_tr_LDA);

y_lda_tr=mapminmax('apply',score_tr_LDA,PS);
y_lda_tst=mapminmax('apply',score_tst_LDA,PS);

y_lin_tr=mapminmax('apply',score_tr_lin,PS);
y_lin_tst=mapminmax('apply',score_tst_lin,PS);

y_quad_tr=mapminmax('apply',score_tr_quad,PS);
y_quad_tst=mapminmax('apply',score_tst_quad,PS);

y_diag_lin_tr=mapminmax('apply',score_tr_diag_lin,PS);
y_diag_lin_tst=mapminmax('apply',score_tst_diag_lin,PS);

y_diag_q_tr=mapminmax('apply',score_tr_diag_q,PS);
y_diag_q_tst=mapminmax('apply',score_tst_diag_q,PS);

y_svm_tr=mapminmax('apply',score_tr_svm,PS);
y_svm_tst=mapminmax('apply',score_tst_svm,PS);

%% Building Ensemble sum min max and Prod Classifier: lin diag_lin diag_quadratic
scores_fus_tr = [y_quad_tr;y_diag_lin_tr;y_diag_q_tr];
scores_fus_tst = [y_quad_tst;y_diag_lin_tst;y_diag_q_tst];
sum_scores_fus_tr = sum(scores_fus_tr);
sum_scores_fus_tst = sum(scores_fus_tst);
min_scores_fus_tr = min(scores_fus_tr);
min_scores_fus_tst = min(scores_fus_tst);
max_scores_fus_tr = max(scores_fus_tr);
max_scores_fus_tst = max(scores_fus_tst);
prod_scores_fus_tr = prod(scores_fus_tr);
prod_scores_fus_tst = prod(scores_fus_tst);
%%

Type_tr = 'Train LDA';
Type_tst = 'Test LDA';
roc_plot(L_tr,score_tr_LDA,L_tst,score_tst_LDA,Type_tr,Type_tst);

Type_tr = 'Train Linear';
Type_tst = 'Test Linear';
roc_plot(L_tr,score_tr_lin,L_tst,score_tst_lin,Type_tr,Type_tst);

Type_tr = 'Train quadratic';
Type_tst = 'Test quadratic';
roc_plot(L_tr,score_tr_quad,L_tst,score_tst_quad,Type_tr,Type_tst);

Type_tr = 'Train diagLinear';
Type_tst = 'Test diagLinear';
roc_plot(L_tr,score_tr_diag_lin,L_tst,score_tst_diag_lin,Type_tr,Type_tst);

Type_tr = 'Train diagQuadratic';
Type_tst = 'Test diagQuadratic';
roc_plot(L_tr,score_tr_diag_q,L_tst,score_tst_diag_q,Type_tr,Type_tst);

Type_tr = 'Train polySVM';
Type_tst = 'Test polySVM';
roc_plot(L_tr,score_tr_svm,L_tst,score_tst_svm,Type_tr,Type_tst);


Type_tr = 'Train Fusion Sum';
Type_tst = 'Test Fusion Sum';
roc_plot(L_tr,sum_scores_fus_tr,L_tst,sum_scores_fus_tst,Type_tr,Type_tst);

Type_tr = 'Train Fusion min';
Type_tst = 'Test Fusion min';
roc_plot(L_tr,min_scores_fus_tr,L_tst,min_scores_fus_tst,Type_tr,Type_tst);

Type_tr = 'Train Fusion max';
Type_tst = 'Test Fusion max';
roc_plot(L_tr,max_scores_fus_tr,L_tst,max_scores_fus_tst,Type_tr,Type_tst);

Type_tr = 'Train Fusion prod';
Type_tst = 'Test Fusion prod';
roc_plot(L_tr,prod_scores_fus_tr,L_tst,prod_scores_fus_tst,Type_tr,Type_tst);


%%
function roc_plot(L_tr,score_train,L_tst,score_test,Type_tr,Type_tst)

[X,Y,~,AUC]=perfcurve(L_tr,score_train,1);
figure;
plot(X,Y);
xlabel('FAR'),ylabel('GAR'),title(['AUC=' num2str(AUC) '  :Type ' Type_tr]);

[X,Y,~,AUC]=perfcurve(L_tst,score_test,1);
figure;
plot(X,Y);
xlabel('FAR'),ylabel('GAR'),title(['AUC=' num2str(AUC) '  :Type ' Type_tst]);

end



%%
function [ score_tr,score_tst]= discriminant_analy(X_train,X_test,L_tr,discriminator)


Mdl = fitcdiscr(X_train,L_tr,'DiscrimType',discriminator); % 
[label,score] = predict(Mdl,X_train);
score_tr=score(:,2)';
%label_tr=label';


%Mdl = fitcdiscr(X_test,L_tst,'DiscrimType',discriminator); % 
[label,score] = predict(Mdl,X_test);
score_tst=score(:,2)';
%label_tst=label';



end



