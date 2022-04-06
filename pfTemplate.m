function M = pfTemplate()
% template and helper functions for 16-642 PS3 problem 2
close all;
clear;
clc;

rng(0); % initialize random number generator

b1 = [5,5]; % position of beacon 1
b2 = [15,5]; % position of beacon 2

% load pfData.mat
load('pfData.mat')

% initialize movie array
numSteps = length(u);
M(numSteps) = struct('cdata',[],'colormap',[]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 tunables                                %
num_particles = 50;
V = [1 0; 0 1] .* [1.26 0; 0 1.75];
W = [1 0; 0 1] .* [0.75 0; 0 0.5];
T = 0.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         put particle filter initialization code here                    %
[x_min, x_max] = deal(0, 20);
[y_min, y_max] = deal(0, 10);
[theta_min, theta_max] = deal(0, 2*pi);
part_x = x_min + (x_max-x_min)*rand([1,num_particles]);
part_y = y_min + (y_max-y_min)*rand([1,num_particles]);
part_theta = theta_min + (theta_max-theta_min)*rand([1,num_particles]);
particles = [part_x; part_y; part_theta];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% here is some code to plot the initial scene
figure(1)
plotParticles(particles); % particle cloud plotting helper function
hold on
plot([b1(1),b2(1)],[b1(2),b2(2)],'s',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5]);
drawRobot(q_groundTruth(:,1), 'cyan'); % robot drawing helper function
axis equal
axis([0 20 0 10])
title('Particle Filter Performance')
xlabel('x position [m]')
ylabel('y position [m]')

M(1) = getframe; % capture current view as movie frame
% pause
% disp('hit return to continue')
hold off

% iterate particle filter in this loop
weights = ones(1,num_particles) * (1/num_particles);
new_particles = zeros(3,num_particles);
for k = 2:numSteps
    for j=1:num_particles
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %              put particle filter prediction step here               %
        %Draw sample noise from distribution for each particle
        v = mvnrnd([0;0],V);
        %Use the given state equation to predict particle movement
        particles(1,j) = particles(1,j) + 0.1*(u(1,k)+v(1))*cos(particles(3,j));
        particles(2,j) = particles(2,j) + 0.1*(u(1,k)+v(1))*sin(particles(3,j));
        particles(3,j) = particles(3,j) + 0.1*(u(2,k)+v(2));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                put particle filter update step here                 %
    % weight particles
    for j=1:num_particles
        dist1 = sqrt( (particles(1,j)-b1(1))^2 + (particles(2,j)-b1(2))^2 );
        dist2 = sqrt( (particles(1,j)-b2(1))^2 + (particles(2,j)-b2(2))^2 );
        weight1 = exp(-((y(1,k)-dist1)^2)/(2*W(1,1)^2)) / (sqrt(2*pi*W(1,1)^2));
        weight2 = exp(-((y(2,k)-dist2)^2)/(2*W(2,2)^2)) / (sqrt(2*pi*W(2,2)^2));
        weights(j) = weight1 * weight2;
    end
    % resample particles
    %Normalize weights so that their sum is equal to 1
    weights_norm = weights ./ sum(weights);
    for j=1:num_particles
        %Make cumulative weight vector
        weights_cum(j) = sum(weights_norm(1:j));
    end
    
    for j=1:num_particles
        z = rand(1);
        i = find((weights_cum>z),1);
        new_particles(:,j) = particles(:,i);
        weights(j) = 1;
    end
    particles = new_particles;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %               get single estimate from particle cloud               %
    if (k < 3)
        [temp, max_ind] = max(weights_norm);
        x_est(:,k) = particles(:,max_ind);
    else
        x_est(:,k) = sum(particles .* weights_norm, 2);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % plot particle cloud, robot, robot estimate, and robot trajectory here %
    h1 = plot(q_groundTruth(1,1:k), q_groundTruth(2,1:k), 'b');
    hold on
    h2 = plot(x_est(1,3:k), x_est(2,3:k), 'r');
    plotParticles(particles)
    plot([b1(1),b2(1)],[b1(2),b2(2)],'s',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','r',...
        'MarkerFaceColor',[0.5,0.5,0.5]);
    drawRobot(q_groundTruth(:,k), 'cyan'); % robot drawing helper function
    drawRobot(x_est(:,k), 'red')
    legend([h1, h2], {'Ground Truth','Predicted'},'location','southeast')
    axis equal
    axis([0 20 0 10])
    title('Particle Filter Performance')
    xlabel('x position [m]')
    ylabel('y position [m]')
    hold off
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % capture current figure and pause
    M(k) = getframe; % capture current view as movie frame
%     pause
%     disp('hit return to continue')
        
end

% when you're ready, the following block of code will export the created 
% movie to an mp4 file
% videoOut = VideoWriter('result.mp4','MPEG-4');
% videoOut.FrameRate=5;
% open(videoOut);
% for k=1:numSteps
%   writeVideo(videoOut,M(k));
% end
% close(videoOut);

end


% helper function to plot a particle cloud
function plotParticles(particles)
plot(particles(1, :), particles(2, :), 'go')
line_length = 0.1;
hold on
quiver(particles(1, :), particles(2, :), line_length * cos(particles(3, :)), line_length * sin(particles(3, :)))
end

% helper function to plot a differential drive robot
function drawRobot(pose, color)
    
% draws a SE2 robot at pose
x = pose(1);
y = pose(2);
th = pose(3);

% define robot shape
robot = [-1 .5 1 .5 -1 -1;
          1  1 0 -1  -1 1 ];
tmp = size(robot);
numPts = tmp(2);
% scale robot if desired
scale = 0.5;
robot = robot*scale;

% convert pose into SE2 matrix
H = [ cos(th)   -sin(th)  x;
      sin(th)    cos(th)  y;
      0          0        1];

% create robot in position
robotPose = H*[robot; ones(1,numPts)];

% plot robot
plot(robotPose(1,:),robotPose(2,:),'k','LineWidth',2);
rFill = fill(robotPose(1,:),robotPose(2,:), color);
alpha(rFill,.2); % make fill semi transparent
end
