function objects = readLabels(label_dir,img_idx)

% parse input file
json_text = fileread(sprintf('%s/%08d.json',label_dir,img_idx));
C = jsondecode(json_text);


% 
% for all objects do
objects = [];
for o = 1:numel(C)

  % extract label, truncation, occlusion
%   lbl = C{1}(o);                   % for converting: cell -> string
  objects(o).type       = C(o).class;  % 'Car', 'Pedestrian', ...
  objects(o).ID_name    = C(o).ID_name;
  objects(o).truncation = 0;%C(o).occlusion_ratio; % truncated pixel ratio ([0..1])
  objects(o).occlusion  = occlusion_rate(C(o).occlusion_ratio); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
  objects(o).alpha      = C(o).occlusion_ratio; % object observation angle ([-pi..pi])

  % extract 2D bounding box in 0-based coordinates
  objects(o).x1 = C(o).bounding_box(1) - 1; % left
  objects(o).y1 = C(o).bounding_box(2) - 1; % top
  objects(o).x2 = C(o).bounding_box(3) - 1; % right
  objects(o).y2 = C(o).bounding_box(4) - 1; % bottom

  % extract 3D bounding box information
  objects(o).l    = C(o).extent_in_world(1) * 2; % box length
  objects(o).h    = C(o).extent_in_world(2) * 2; % box width
  objects(o).w    = C(o).extent_in_world(3) * 2; % box height 
  objects(o).t(1) = C(o).origin_in_world(1); % center location (x)
  objects(o).t(2) = C(o).origin_in_world(2); % center location (y)
  objects(o).t(3) = C(o).origin_in_world(3); % center location (z)
  objects(o).l_c    = C(o).extent_in_camera(1) * 2; % box length
  objects(o).h_c    = C(o).extent_in_camera(2) * 2; % box height
  objects(o).w_c    = C(o).extent_in_camera(3) * 2; % box width 
  objects(o).t_c(1) = C(o).origin_in_camera(1); % center location (x)
  objects(o).t_c(2) = C(o).origin_in_camera(2); % center location (y)
  objects(o).t_c(3) = C(o).origin_in_camera(3); % center location (z)
  objects(o).ry   = 0; % yaw angle
%   objects(o).ry   = C{15}(o); % yaw angle

  % extract view_location and view_rotation to compute the transform matrix
  objects(o).view_loc(1) = C(o).pose_translation(1); % camera location in the world (x)
  objects(o).view_loc(2) = C(o).pose_translation(2); % camera location in the world (y)
  objects(o).view_loc(3) = C(o).pose_translation(3); % camera location in the world (z)
  objects(o).view_rot(1) = C(o).pose_rotation(1); % camera rotation wrt the world frame (yaw)
  objects(o).view_rot(2) = C(o).pose_rotation(2); % camera rotation wrt the world frame (pitch)
  objects(o).view_rot(3) = C(o).pose_rotation(3); % camera rotation wrt the world frame (roll)
  
  objects(o).pose = [reshape(C(o).pose, 4, 3)';
                     0,   0,            0, 1];
end

function occ_rate = occlusion_rate(occlusion_ratio)
    if occlusion_ratio <= 0.2
        occ_rate = 0;
    elseif occlusion_ratio <= 0.9
        occ_rate = 1;
    elseif occlusion_ratio <= 1.0
        occ_rate = 2;
    else
        occ_rate = 3;
    end

