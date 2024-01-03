% Auto-regularize Fisher Linear Discriminant two-class projection direction finder
% (c) 2009 Reza Derakhshani, v3.1, now returns regularization frequency
% Usage: [w J m reg]=LDA3(C1,C2)
% where C1 (d by N1) has the N1 samples of class C1. Ditto for class 2(C2 dxN2)
% Provides w (1xd), Fisher projection vector;J, Fisher criterion (inter- / intra-class variance ratio)
% use w'*Ci to find projected class i, etc. m is dataset mean
% Classification: based on the sign of w'*x-w'*m
%reg mentions how many times determinant was <1e-6 (unstability metric)

function [w J m reg]=LDA3(C1,C2)
reg=0;
%Implements w ~ Sw^-1 * (m2-m1). Seepage 189 of Bishop's 2006 Pat Rec text
m1=mean(C1')';  %Class 1 mean
m2=mean(C2')';  %Class 2 mean
m=mean([C1 C2]')'; %C1 if w'*(x-m)<0, C2 otherwise (check m1 and m2 projection for correct sidedness)
Sw=(C1-repmat(m1,1,size(C1,2)))*(C1-repmat(m1,1,size(C1,2)))'+(C2-repmat(m2,1,size(C2,2)))*(C2-repmat(m2,1,size(C2,2)))';   %Total within-class covariance matrix
while (abs(det(Sw))<1e-6),   %Regularize
    Regularizing_this_det=det(Sw)
    Sw=Sw+0.001*eye(size(Sw,1));
    reg=reg+1;
end
w=inv(Sw)*(m2-m1);
Sb=(m2-m1)*(m2-m1)';
J=(w'*Sb*w)/(w'*Sw*w);
