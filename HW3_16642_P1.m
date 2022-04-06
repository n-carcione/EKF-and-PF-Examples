%% General Clearing
close all;
clear;
clc;

%% Part b
close all;
clear;
clc;
%Load in data and separate into individual variables
cal_data = load('calibration.mat');
q_gt = cal_data.q_groundTruth;
t_gt = cal_data.t_groundTruth;
t_y = cal_data.t_y;
u = cal_data.u;
y = cal_data.y;

%Measurement covariance W
w = y - q_gt(1:2,10:10:2500);
% figure;
% histogram(w(1,:));
% hold on;
% histogram(w(2,:));
% legend('w1','w2')
% figure;
% plot(w(1,:),w(2,:),'.')
W = cov(w');

%Process covariance V
v = zeros(2,length(q_gt)-1);
T = 0.01;
for k=1:length(q_gt)-1
    F = [1 0 -T*sin(q_gt(3,k));
         0 1  T*cos(q_gt(3,k));
         0 0  1];
    G = [T*cos(q_gt(3,k)), 0;
         T*sin(q_gt(3,k)), 0;
         0               , T];
    gamma = [T*cos(q_gt(3,k)), 0;
             T*sin(q_gt(3,k)), 0;
             0               , T];
    v(:,k) = gamma \ (q_gt(:,k+1) - F*q_gt(:,k) - G*u(:,k));
end
% figure;
% histogram(v(1,:));
% hold on;
% histogram(v(2,:));
% legend('v1','v2')
% figure;
% plot(v(1,:),v(2,:),'.')
V = cov(v');

%% Part c
%Load in data and create given initializations
kfData = load('kfData.mat');
q_gtc = kfData.q_groundtruth;
tc = kfData.t;
t_yc = kfData.t_y;
uc = kfData.u;
yc = kfData.y;
qh(:,1) = [0.355; -1.590; 0.682];
% qh(:,1) = [0; 1; 0.3927];
P = [25 0 0; 0 25 0; 0 0 0.154];

%i) dead reckoner
for k=1:length(uc)
    v = mvnrnd([0;0],V);
    qh(1,k+1) = qh(1,k) + T*(uc(1,k) + v(1))*cos(qh(3,k));
    qh(2,k+1) = qh(2,k) + T*(uc(1,k) + v(1))*sin(qh(3,k));
    qh(3,k+1) = qh(3,k) + T*(uc(2,k) + v(2));
end

%Plot dead reckoner vs actual state
figure;
plot(qh(1,:),qh(2,:),'r')
hold on
plot(q_gtc(1,:),q_gtc(2,:),'b')
title('Dead Reckoner Performance')
xlabel('x_r [m]')
ylabel('y_r [m]')
legend('Dead Reckoner','Ground Truth','location','best')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ii) Full EKF
qh(:,1) = [0.355; -1.590; 0.682];
P = [25 0 0; 0 25 0; 0 0 0.154];
i = 1;

for k=1:length(uc)
    %Prediction Step
    v = mvnrnd([0;0],V);
    qh(1,k+1) = qh(1,k) + T*(uc(1,k) + v(1))*cos(qh(3,k));
    qh(2,k+1) = qh(2,k) + T*(uc(1,k) + v(1))*sin(qh(3,k));
    qh(3,k+1) = qh(3,k) + T*(uc(2,k) + v(2));
    F = [1 0 -T*(uc(1,k))*sin(qh(3,k));
         0 1  T*(uc(1,k))*cos(qh(3,k));
         0 0  1];
    gamma = [T*cos(qh(3,k)), 0;
             T*sin(qh(3,k)), 0;
             0             , T];
    P = F*P*F' + gamma * V * gamma';
    
    %Update Step
    if (mod(k+1,10) == 0)
        w = mvnrnd([0;0],W);
        H = [1 0 0; 0 1 0];
        S = H*P*H' + W;
        qh(:,k+1) = qh(:,k+1) + P*H'*inv(S)*(yc(:,i) - qh(1:2,k+1));
        P = P - P*H'*inv(S)*H*P;
        i = i + 1;
    end
end

%Plot requested results
figure;
plot(q_gtc(1,:),q_gtc(2,:),'b')
hold on
plot(yc(1,:),yc(2,:),'gx')
plot(qh(1,:),qh(2,:),'r')
title('Full EKF Implementation')
xlabel('x_r [m]')
ylabel('y_r [m]')
legend('Ground Truth','GPS Measurements','EKF Trajectory','location','best')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%iii) scaled down W
W = W ./ 100;
qh(:,1) = [0.355; -1.590; 0.682];
P = [25 0 0; 0 25 0; 0 0 0.154];
i = 1;

for k=1:length(uc)
    %*******Implement Full EKF here (need to fix F matrix?)
    %Prediction Step
    v = mvnrnd([0;0],V);
    qh(1,k+1) = qh(1,k) + T*(uc(1,k) + v(1))*cos(qh(3,k));
    qh(2,k+1) = qh(2,k) + T*(uc(1,k) + v(1))*sin(qh(3,k));
    qh(3,k+1) = qh(3,k) + T*(uc(2,k) + v(2));
    F = [1 0 -T*(uc(1,k))*sin(qh(3,k));
         0 1  T*(uc(1,k))*cos(qh(3,k));
         0 0  1];
    gamma = [T*cos(qh(3,k)), 0;
             T*sin(qh(3,k)), 0;
             0             , T];
    P = F*P*F' + gamma * V * gamma';
    
    %Update Step
    if (mod(k+1,10) == 0)
        H = [1 0 0; 0 1 0];
        S = H*P*H' + W;
        qh(:,k+1) = qh(:,k+1) + P*H'*inv(S)*(yc(:,i) - qh(1:2,k+1));
        P = P - P*H'*inv(S)*H*P;
        i = i + 1;
    end
end

% Plot requested results
figure;
plot(q_gtc(1,:),q_gtc(2,:),'b')
hold on
plot(yc(1,:),yc(2,:),'gx')
plot(qh(1,:),qh(2,:),'r')
title('Full EKF Implementation')
xlabel('x_r [m]')
ylabel('y_r [m]')
legend('Ground Truth','GPS Measurements','EKF Trajectory','location','best')
grid on