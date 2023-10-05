clc;
clear all;
close all;

load('HW3_data_benign','data_benign')
load('HW3_data_malignant','data_malignant')
ben_all=data_benign';
mal_all=data_malignant';

%Taking 10 datapoints for parameter estimation
data_benign_10 = ben_all(:,1:10);
data_malignant_10 = mal_all(:,1:10);

%Estimating the mean and covariance parameters
mu1 = mean(data_benign_10',1)'; % mean estimate of benign features
mu2 = mean(data_malignant_10',1)'; % mean estimate of malignant features
C1 = cov(data_benign_10');%covariance estimate of benign
C2 = cov(data_malignant_10');%covariance estimate of malignant

%ML estimate
W1=-.5*inv(C1);
w1=inv(C1)*mu1;
w10=-0.5*mu1'*inv(C1)*mu1-0.5*log(det(C1));

W2=-.5*inv(C2);
w2=inv(C2)*mu2;
w20=-0.5*mu2'*inv(C2)*mu2-0.5*log(det(C2));

% Classifying the remaining 90 datapoints from each class
ben = ben_all(:,11:100);%test data
mal = mal_all(:,11:100);%test data

yben=zeros(length(ben),1);
g=zeros(length(ben),1);
for i=1:length(ben)
    x=ben(:,i);
    g1=x'*W1*x+w1'*x+w10;
    g2=x'*W2*x+w2'*x+w20;
    g(i)=g1-g2;
    if g(i)<0
        yben(i)=1;
    else
        yben(i)=0;
    end
end

ymal=zeros(length(mal),1);
g=zeros(length(mal),1);
for i=1:length(mal)
    x=mal(:,i);
    g1=x'*W1*x+w1'*x+w10;
    g2=x'*W2*x+w2'*x+w20;
    g(i)=g1-g2;

    if g(i)>0
        ymal(i)=0;
    else
       ymal(i)=1;
    end
end
%Probability of Detection and Probability of False Alarm
disp("Probability of detection when true class in Benign=")
disp(1-nnz(yben)/length(yben))
disp("Probability of False Alarm when true class in Benign=")
disp(nnz(yben)/length(yben))       
disp("Probability of Detection when true class in Malignant=")
disp(nnz(ymal)/length(ymal))
disp("Probability of False alarm when true class in Malignant=")
disp(1-nnz(ymal)/length(ymal))

%Plotting classified data
fpr_ben=ben(:,find(yben==1)); %true is benign, classified as malignant (False Alarm)
tpr_ben=ben(:,find(yben==0)); %true is benign, classified as benign

fpr_mal=mal(:,find(ymal==0));%true is malignant, classified as benign
tpr_mal=mal(:,find(ymal==1));%true is malignant, classified as malignant (True Detection)

figure(1)
scatter3(fpr_ben(1,:),fpr_ben(2,:),fpr_ben(3,:),'r+')
hold on
scatter3(tpr_ben(1,:),tpr_ben(2,:),tpr_ben(3,:),'b+')
hold on
scatter3(fpr_mal(1,:),fpr_mal(2,:),fpr_mal(3,:),'c*')
hold on
scatter3(tpr_mal(1,:),tpr_mal(2,:),tpr_mal(3,:),'k*')
xlabel('Feature 1')
ylabel('Feature 2')
zlabel('Feature 3')
legend('true data is benign, classified as malignant', 'benign data classified correctly','true data is malignant, classified as benign','malignant data classified correctly')



