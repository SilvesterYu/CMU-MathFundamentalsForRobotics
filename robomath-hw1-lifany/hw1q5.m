% HW1-Question5
clear all; close all
%% Generating the initial point set and the final point set after translating and rotating
% Randomly generating the rotation angles around the x-, y-, and z-axis
theta  = rand(3,1)*2*pi;
% Rotating Matrix around the X-axis
Rx = [1,0,0; ...
    0, cos(theta(1)), -sin(theta(1));...
    0, sin(theta(1)), cos(theta(1))];
% Rotating Matrix around the Y-axis
Ry = [cos(theta(2)),0, -sin(theta(2));...
    0,1,0; ...
    sin(theta(2)), 0, cos(theta(2))];
% Rotating Matrix around the Z-axis
Rz = [cos(theta(3)), -sin(theta(3)),0;...
    sin(theta(3)), cos(theta(3)),0;...
    0,0,1];
% Triaxial rotation matrix, rotates first around x, then around y, and finally around the z axis
R0 = Rz*Ry*Rx
DetofR0 = det(R0)

% Randomly generating translation vector
t0 = 8*(rand(3,1)-0.5)

%Randomly generating an initial point set of a rigid body
n = 4
P = rand(3,n)

% The final point set after translating and rotating
Q = R0 * P + t0*ones(1,n)

%% My  Scheme

% Compute the average centers of the initial point set and the final point set
p0 = mean(P,2);
q0 = mean(Q,2);

%  Coordinates of initial point set and the final point set Relative to the corresponding average centers
X = P - p0*ones(1,n);
Y = Q - q0*ones(1,n);

[U,S,V] = svd(X*(Y.'))
R = V*(U.')
t = q0 - R*p0

%  compute error to verify the scheme
difofR = R - R0
difoft = t - t0
