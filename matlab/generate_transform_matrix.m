function transform_matrix = generate_transform_matrix(y, p, r, dx, dy, dz)
% this function compute the transform matrix that transforms a point
% location in the world coordinate to the camera corrdinate
% P_c = P_w * transform_matrix
% P_w = [X_w, Y_w, Z_w, 1];
% this function is written according to the function
% UpdateProjectionMatrix() in IsaacSimBoundingBox.cpp

% [y, p, r] = [yaw, pitch, roll] the rotation of the camera coordinate wrt
% the world coordinate
% [dx, dy, dz] = [X_w, Y_w, Z_w] the location of the origin of the camera
% coordinate in the world coordinate

y = y/180.0 * pi; p = -p/180.0 * pi; r = -r/180.0 * pi;
dx = -dx; dy = -dy; dz = -dz;
yaw = [cos(y), -sin(y), 0;
       sin(y),  cos(y), 0;
       0,      0,       1];
pitch = [ cos(p), 0, sin(p);
          0,      1,      0;
         -sin(p), 0, cos(p)];
roll = [1,      0,       0;
        0, cos(r), -sin(r);
        0, sin(r),  cos(r)];
    
rotation_matrix = yaw *  pitch * roll; 

rotation_matrix = [ rotation_matrix, [0,0,0]';
                    0, 0, 0,         1];

translation_matrix = [1,  0,   0,  0;
                      0,  1,   0,  0;
                      0,  0,   1,  0;
                      dx, dy,  dz, 1];
switch_matrix = [0, 0, 1, 0;
                 1, 0, 0, 0;
                 0, 1, 0, 0;
                 0, 0, 0, 1];
transform_matrix = translation_matrix * rotation_matrix * switch_matrix;
