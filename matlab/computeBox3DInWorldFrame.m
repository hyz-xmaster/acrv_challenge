function [corners_2D,face_idx] = computeBox3DInWorldFrame(object,P, image_size)
% takes an object and a projection matrix (P) and projects the 3D
% bounding box into the image plane.

% index for 3D bounding box faces
% face_idx = [ 1,2,6,5   % front face
%              2,3,7,6   % left face
%              3,4,8,7   % back face
%              4,1,5,8]; % right face
face_idx = [ 1,5,6,2  % front face
             2,6,7,3  % left face
             3,7,8,4  % back face
             1,5,8,4]; % right face

% compute rotational matrix around yaw axis
R = [+cos(object.ry), 0, +sin(object.ry);
                   0, 1,               0;
     -sin(object.ry), 0, +cos(object.ry)];

% 3D bounding box dimensions
l = object.l; % extent.X
h = object.h; % extent.Y
w = object.w; % extent.Z


% 3D bounding box corners
x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2];
% y_corners = [0,0,0,0,-h,-h,-h,-h];
y_corners = [h/2, -h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2];
% z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2];
z_corners = [-w/2, -w/2, -w/2, -w/2, w/2, w/2, w/2, w/2];

% rotate and translate 3D bounding box
% corners_3D = R*[x_corners;y_corners;z_corners];
corners_3D = [x_corners;y_corners;z_corners];
corners_3D(1,:) = corners_3D(1,:) + object.t(1);
corners_3D(2,:) = corners_3D(2,:) + object.t(2);
corners_3D(3,:) = corners_3D(3,:) + object.t(3);

% transform the world locations to camera locations
corners_3D = [corners_3D; ones(1, size(corners_3D, 2))];
y = object.view_rot(1); p = object.view_rot(2); r = object.view_rot(3);
dx = object.view_loc(1); dy = object.view_loc(2); dz = object.view_loc(3);
transform_matrix = generate_transform_matrix(y, p, r, dx, dy, dz);
corners_3D = corners_3D' * transform_matrix;
corners_3D = corners_3D(:, 1:3)';

% only draw 3D bounding box for objects in front of the camera
if any(corners_3D(3,:)<0.1) 
  corners_2D = [];
  return;
end

% project the 3D bounding box into the image plane
corners_2D = projectToImage_full(corners_3D, P);

if any(corners_2D(:) < 0) || any(corners_2D(1,:) > image_size(2)) || any(corners_2D(2,:) > image_size(1))
    corners_2D = [];
    return;
end
