function [origin, extent] = compute3DBoxCorners(object)

% 3D bounding box dimensions
l = object.l; % extent.X
h = object.h; % extent.Y
w = object.w; % extent.Z
extent = [l, h, w];

object.t(2) = -object.t(2);
origin = object.t - 0.5 * extent;
% origin(2) = -1.0 * origin(2);




% % 3D bounding box corners
% x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2];
% % y_corners = [0,0,0,0,-h,-h,-h,-h];
% y_corners = [h/2, -h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2];
% % z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2];
% z_corners = [-w/2, -w/2, -w/2, -w/2, w/2, w/2, w/2, w/2];
% 
% % rotate and translate 3D bounding box
% % corners_3D = R*[x_corners;y_corners;z_corners];
% corners_3D = [x_corners;y_corners;z_corners];
% corners_3D(1,:) = corners_3D(1,:) + object.t(1);
% corners_3D(2,:) = corners_3D(2,:) + object.t(2);
% corners_3D(3,:) = corners_3D(3,:) + object.t(3);