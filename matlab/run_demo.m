% This tool displays the images and the object labels for the benchmark and
% provides an entry point for writing your own interface to the data set.
% Before running this tool, set root_dir to the directory where you have
% downloaded the dataset. 'root_dir' must contain the subdirectory
% 'rgb' and 'ground_truth' which in turn contains 'instance', 'label' and
% 'segmentation'.
% For more information about the data format, please look into readme.txt.
%
% Usage:
%   SPACE: next frame
%   '-':   last frame
%   'x':   +10 frames
%   'y':   -10 frames
%   q:     quit
%
% Occlusion Coding:
%   green:  not occluded
%   yellow: partly occluded
%   red:    fully occluded
%   white:  unknown
%


% clear and close everything
clear all; close all;
disp('======= ACRV DevKit Demo =======');

% options
root_dir = '/home/wind/Data/Isaac_sim/AIUE_V01_001/demo';
data_set = '000000';

% get sub-directories
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,'/rgb', data_set);
label_dir = fullfile(root_dir,'/ground_truth/label', data_set);

%show the difference between two sequences
label_dir_2 = fullfile(root_dir,'/ground_truth/label', '000001');
findSequencesDifference(label_dir, label_dir_2, 1);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

% set up figure
h = visualization('init',image_dir);

% find all the objects and plot them
all_objects = findAllOjbects(label_dir, nimages);
figure(2); h2 = subplot(1,1,1);axis equal;
title('The object map and the robot trajectory');
hold(h2, 'on');
object_keys = keys(all_objects);

for key_idx = 1:numel(object_keys)
    current_object = all_objects(object_keys{key_idx});
    [box_origin, box_extent] = compute3DBoxCorners(current_object);
    hh = plotcube(box_extent/100.0, box_origin/100.0, .1, [0, 1, 0], h2);
end
set(h2, 'Position', [0., 0., 1, 1]);

% main loop
img_idx=0;
T0 =  [1, 0, 0, 0;
       0, 1, 0, 0;
       0, 0, 1, 0;
       0, 0, 0, 1];

locations = [[]];
while 1

  % load projection matrix
%   P = readCalibration(calib_dir,img_idx,cam);
%   P = [1.0, 0.0,  0.0, 0.0;
%        0.0, 1.78, 0.0, 0.0;
%        0.0, 0.0,  0.0, 1.0;
%        0.0, 0.0,  10.0, 0.0];
   
  P_f = [480.0, 0.0,    480.0,  0.0;
         0.0,   -480.0, 270.0,  0.0;
         0.0,   0.0,    1.0,    0.0;
         0.0,   0.0,    0.0,    1.0];
  
  % load labels
  objects = readLabels(label_dir,img_idx);
  
  % visualization update for next frame
  [~, image_size] = visualization('update',image_dir,h,img_idx,nimages,data_set);
 
  % for all annotated objects doCVMdl.Trained{
  for obj_idx=1:numel(objects)
   
    % plot 2D bounding box
    drawBox2D(h,objects(obj_idx));
    
    % plot 3D bounding box
    [corners,face_idx] = computeBox3DInWorldFrame(objects(obj_idx),P_f, image_size);
    drawBox3D(h, objects(obj_idx),corners,face_idx,[]);
    
  end

  % draw the trajectory
  if ~isempty(objects)
      camera_loc = objects(1).view_loc/100.0;
      plot3(h2, camera_loc(1), -camera_loc(2), camera_loc(3), 'c*');
  end
  
  % force drawing and tiny user interface
  waitforbuttonpress; 
  key = get(gcf,'CurrentCharacter');

  switch lower(key)                         
    case 'q',  break;                                 % quit
    case '-',  img_idx = max(img_idx-1,  0);          % previous frame
    case 'x',  img_idx = min(img_idx+10, nimages-1); % +10 frames
    case 'y',  img_idx = max(img_idx-10, 0);         % -10 frames
    otherwise, img_idx = min(img_idx+1,  nimages-1);  % next frame
  end

end

% clean up
close all;
